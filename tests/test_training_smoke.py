"""CPU smoke tests for gaze predictor training loop.

Verifies that training runs, loss decreases, checkpoints save/load,
and inference methods work on a briefly-trained model.
All tests use synthetic data — no GECO download required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.geco import GazeDataset
from src.gaze_predictor import GazePredictor, GazePredictorConfig, load_trained_predictor
from tests.conftest import _make_synthetic_examples


def _train_one_config(examples, tokenizer, config, epochs=2):
    """Helper: train a GazePredictor and return per-epoch losses."""
    dataset = GazeDataset(examples, tokenizer, max_length=32, normalize=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = GazePredictor(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    epoch_losses = []
    for _epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            preds = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(preds, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

    model.eval()
    return model, dataset, epoch_losses


@pytest.mark.slow
class TestGazePredictorTrainingSmoke:
    """Smoke tests for the gaze predictor training pipeline."""

    def test_training_loss_decreases(self, bert_tokenizer):
        """Training for 4 epochs produces finite loss with overall downward trend."""
        torch.manual_seed(42)
        examples = _make_synthetic_examples(20)
        config = GazePredictorConfig(
            learning_rate=1e-4, batch_size=4, epochs=4, max_seq_length=32,
        )

        _, _, losses = _train_one_config(examples, bert_tokenizer, config, epochs=4)

        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
        assert all(l > 0 for l in losses)
        # Last epoch should be lower than first (overall trend, not monotonic)
        assert losses[-1] < losses[0], f"Loss did not decrease overall: {losses}"

    def test_checkpoint_save_and_load(self, bert_tokenizer, tmp_path):
        """Checkpoint round-trips: save → load produces identical predictions."""
        torch.manual_seed(42)
        examples = _make_synthetic_examples(12)
        config = GazePredictorConfig(
            learning_rate=5e-4, batch_size=4, epochs=1, max_seq_length=32,
        )

        model, dataset, _ = _train_one_config(examples, bert_tokenizer, config, epochs=1)
        norm_stats = dataset.get_normalization_stats()

        checkpoint_path = tmp_path / "test_ckpt.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "normalization": norm_stats,
            },
            checkpoint_path,
        )

        # Fixed input for comparison
        input_ids = torch.randint(0, 30522, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        with torch.no_grad():
            original_preds = model(input_ids, attention_mask)

        loaded_model, loaded_norm = load_trained_predictor(
            checkpoint_path, device=torch.device("cpu")
        )

        assert isinstance(loaded_model, GazePredictor)
        assert not loaded_model.training  # eval mode
        assert loaded_norm == norm_stats

        with torch.no_grad():
            loaded_preds = loaded_model(input_ids, attention_mask)

        assert torch.allclose(original_preds, loaded_preds, atol=1e-6)

    def test_score_tokens_after_training(self, bert_tokenizer):
        """score_tokens returns per-position scores with non-zero content."""
        torch.manual_seed(42)
        examples = _make_synthetic_examples(16)
        config = GazePredictorConfig(
            learning_rate=5e-4, batch_size=4, epochs=1, max_seq_length=32,
        )
        model, _, _ = _train_one_config(examples, bert_tokenizer, config, epochs=1)

        token_ids = bert_tokenizer.encode("The cat sat on the mat", return_tensors="pt").squeeze(0)
        scores = model.score_tokens(token_ids, bert_tokenizer, context_window=3)

        assert scores.shape[0] == len(token_ids)
        assert scores.dtype == torch.float32
        assert not torch.isnan(scores).any()
        # Special tokens (CLS=0, SEP=last) should be 0
        assert scores[0].item() == 0.0
        assert scores[-1].item() == 0.0
        # Content tokens should have some non-zero scores
        content_scores = scores[1:-1]
        assert (content_scores != 0).sum() >= 1

    def test_score_vocabulary_after_training(self, bert_tokenizer):
        """score_vocabulary returns per-candidate scores with variation."""
        torch.manual_seed(42)
        examples = _make_synthetic_examples(16)
        config = GazePredictorConfig(
            learning_rate=5e-4, batch_size=4, epochs=1, max_seq_length=32,
        )
        model, _, _ = _train_one_config(examples, bert_tokenizer, config, epochs=1)

        sequence = bert_tokenizer.encode("The big dog ran fast", return_tensors="pt").squeeze(0)
        candidates = torch.tensor([
            bert_tokenizer.encode(w, add_special_tokens=False)[0]
            for w in ["cat", "epistemological", "the", "running", "photosynthesis"]
        ])

        scores = model.score_vocabulary(
            sequence=sequence, position=3, vocab_candidates=candidates,
            tokenizer=bert_tokenizer,
        )

        assert scores.shape == (5,)
        assert torch.isfinite(scores).all()
        assert scores.std() > 1e-8, "All candidates scored identically"

    def test_validation_spearman_computable(self, bert_tokenizer):
        """After 2 epochs, validation predictions yield a valid Spearman r."""
        from scipy.stats import spearmanr

        torch.manual_seed(42)
        examples = _make_synthetic_examples(20)
        train_ex, val_ex = examples[:16], examples[4:]
        config = GazePredictorConfig(
            learning_rate=5e-4, batch_size=4, epochs=2, max_seq_length=32,
        )

        model, _, _ = _train_one_config(train_ex, bert_tokenizer, config, epochs=2)

        val_dataset = GazeDataset(val_ex, bert_tokenizer, max_length=32, normalize=True)
        val_loader = DataLoader(val_dataset, batch_size=4, num_workers=0)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch["input_ids"], batch["attention_mask"])
                all_preds.extend(preds.numpy().tolist())
                all_labels.extend(batch["labels"].numpy().tolist())

        r, p = spearmanr(all_preds, all_labels)
        assert not torch.isnan(torch.tensor(r)), "Spearman r is NaN"
        assert -1.0 <= r <= 1.0
        assert torch.isfinite(torch.tensor(p))
