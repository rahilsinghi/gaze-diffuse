"""Integration tests wiring real components together on CPU.

Tests data pipeline → predictor → guidance → metrics chains.
All use synthetic data + mocks — no GPU or model downloads beyond BERT.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ar_baseline import ARGazeConfig, ar_gaze_guided_generate
from src.data.geco import (
    GazeDataConfig,
    GazeDataset,
    create_subject_cv_splits,
    extract_gaze_examples,
)
from src.gaze_guidance import GazeDiffuseConfig, gaze_guided_diffuse
from src.gaze_predictor import GazePredictor, GazePredictorConfig
from src.metrics import evaluate_generations
from tests.conftest import _make_synthetic_examples


def _get_trained_predictor(tokenizer):
    """Train a GazePredictor for 1 epoch on synthetic data. Cached per call."""
    torch.manual_seed(42)
    examples = _make_synthetic_examples(20)
    dataset = GazeDataset(examples, tokenizer, max_length=32, normalize=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    config = GazePredictorConfig(
        learning_rate=5e-4, batch_size=4, epochs=1, max_seq_length=32,
    )
    model = GazePredictor(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for batch in loader:
        preds = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(preds, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# Module-level cache to avoid retraining per test
_trained_predictor_cache: dict[str, GazePredictor] = {}


def _cached_trained_predictor(tokenizer):
    if "model" not in _trained_predictor_cache:
        _trained_predictor_cache["model"] = _get_trained_predictor(tokenizer)
    return _trained_predictor_cache["model"]


# ---- Data Pipeline Tests ----


@pytest.mark.integration
class TestDataPipeline:

    def test_extract_gaze_examples_from_dataframe(self, synthetic_gaze_dataframe):
        """extract_gaze_examples converts DataFrame to GazeExample list."""
        config = GazeDataConfig(context_window=3, min_fixation_ms=0.0, max_fixation_ms=2000.0)
        examples = extract_gaze_examples(synthetic_gaze_dataframe, config)

        assert len(examples) > 0
        for ex in examples:
            assert len(ex.left_context) <= 3
            assert len(ex.right_context) <= 3
            assert 0.0 <= ex.fixation_duration_ms <= 2000.0

    @pytest.mark.slow
    def test_dataframe_to_dataset_to_dataloader(self, synthetic_gaze_dataframe, bert_tokenizer):
        """Full pipeline: DataFrame → examples → GazeDataset → DataLoader batch."""
        config = GazeDataConfig(context_window=3)
        examples = extract_gaze_examples(synthetic_gaze_dataframe, config)
        dataset = GazeDataset(examples, bert_tokenizer, max_length=32, normalize=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        batch = next(iter(loader))

        assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
        assert batch["input_ids"].shape == (4, 32)
        assert batch["attention_mask"].shape == (4, 32)
        assert batch["labels"].shape == (4,)
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.float32

    def test_subject_cv_splits_no_leakage(self, synthetic_gaze_examples):
        """CV splits have zero participant overlap between train and test."""
        splits = list(create_subject_cv_splits(synthetic_gaze_examples, n_folds=2))

        assert len(splits) == 2
        all_test_participants = set()

        for train, test in splits:
            train_parts = {ex.participant_id for ex in train}
            test_parts = {ex.participant_id for ex in test}
            assert train_parts & test_parts == set(), "Train/test participant overlap"
            assert len(train) + len(test) == len(synthetic_gaze_examples)
            all_test_participants |= test_parts

        all_participants = {ex.participant_id for ex in synthetic_gaze_examples}
        assert all_test_participants == all_participants


# ---- Gaze Predictor Integration Tests ----


@pytest.mark.integration
class TestGazePredictorIntegration:

    @pytest.mark.slow
    def test_gaze_dataset_to_predictor_forward(self, bert_tokenizer):
        """A GazeDataset batch feeds into GazePredictor.forward() with gradient flow."""
        examples = _make_synthetic_examples(8)
        dataset = GazeDataset(examples, bert_tokenizer, max_length=32)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        model = GazePredictor(GazePredictorConfig(max_seq_length=32))

        batch = next(iter(loader))
        preds = model(batch["input_ids"], batch["attention_mask"])
        loss = nn.functional.mse_loss(preds, batch["labels"])
        loss.backward()

        assert preds.shape == (4,)
        assert torch.isfinite(loss)
        assert loss.item() > 0
        # Gradients should flow through both BERT and regressor
        assert any(p.grad is not None for p in model.regressor.parameters())

    @pytest.mark.slow
    def test_predictor_score_tokens_chain(self, bert_tokenizer):
        """Trained predictor → score_tokens produces varying per-position scores."""
        model = _cached_trained_predictor(bert_tokenizer)
        token_ids = bert_tokenizer.encode(
            "Scientists discovered a new species in the deep ocean",
            return_tensors="pt",
        ).squeeze(0)

        scores = model.score_tokens(token_ids, bert_tokenizer, context_window=5)

        assert len(scores) == len(token_ids)
        assert scores[0].item() == 0.0   # CLS
        assert scores[-1].item() == 0.0  # SEP
        content_scores = scores[1:-1]
        assert (content_scores != 0).sum() >= 1
        assert content_scores.std() > 0, "All content tokens scored identically"


# ---- GazeDiffuse Integration Tests ----


@pytest.mark.integration
class TestGazeDiffuseIntegration:

    @pytest.mark.slow
    def test_with_mock_mdlm_and_trained_predictor(self, mock_mdlm, bert_tokenizer):
        """GazeDiffuse runs end-to-end with MockMDLM + real trained predictor."""
        model = _cached_trained_predictor(bert_tokenizer)
        prompt_ids = bert_tokenizer.encode("The discovery of", return_tensors="pt").squeeze(0)

        config = GazeDiffuseConfig(
            lam=-1.0, steps=3, gen_length=8, top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )

        output = gaze_guided_diffuse(mock_mdlm, model, bert_tokenizer, prompt_ids, config)

        n_prompt = len(prompt_ids)
        assert output.shape == (n_prompt + 8,)
        assert torch.equal(output[:n_prompt], prompt_ids)
        assert (output == bert_tokenizer.mask_token_id).sum() == 0
        assert (output >= 0).all() and (output < 30522).all()
        # Should decode without error
        bert_tokenizer.decode(output, skip_special_tokens=True)

    @pytest.mark.slow
    def test_unguided_vs_guided_differ(self, mock_mdlm, bert_tokenizer):
        """lam=0 (unguided) vs lam=-1 (guided) produce different outputs."""
        model = _cached_trained_predictor(bert_tokenizer)
        prompt_ids = bert_tokenizer.encode("The discovery of", return_tensors="pt").squeeze(0)
        n_prompt = len(prompt_ids)

        config_unguided = GazeDiffuseConfig(
            lam=0.0, steps=3, gen_length=12, top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )
        config_guided = GazeDiffuseConfig(
            lam=-1.0, steps=3, gen_length=12, top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )

        torch.manual_seed(42)
        out_unguided = gaze_guided_diffuse(mock_mdlm, model, bert_tokenizer, prompt_ids, config_unguided)

        torch.manual_seed(42)
        out_guided = gaze_guided_diffuse(mock_mdlm, model, bert_tokenizer, prompt_ids, config_guided)

        assert out_unguided.shape == out_guided.shape
        assert (out_unguided == bert_tokenizer.mask_token_id).sum() == 0
        assert (out_guided == bert_tokenizer.mask_token_id).sum() == 0
        # Gaze guidance should change at least some tokens
        assert not torch.equal(out_unguided[n_prompt:], out_guided[n_prompt:])


# ---- AR Baseline Integration Tests ----


@pytest.mark.integration
class TestARBaselineIntegration:

    @pytest.mark.slow
    def test_with_mock_lm_and_trained_predictor(self, mock_causal_lm, bert_tokenizer):
        """AR generation completes with mock LM + trained gaze predictor."""
        gaze_pred = _cached_trained_predictor(bert_tokenizer)
        prompt_ids = bert_tokenizer.encode("Scientists have long", return_tensors="pt").squeeze(0)

        config = ARGazeConfig(
            model_name="mock", lam=-1.0, top_k=10, max_new_tokens=5,
        )

        output = ar_gaze_guided_generate(
            mock_causal_lm, gaze_pred, bert_tokenizer, prompt_ids, config,
        )

        assert len(output) >= len(prompt_ids)
        assert len(output) <= len(prompt_ids) + 5
        assert torch.equal(output[:len(prompt_ids)], prompt_ids)
        assert (output >= 0).all()
        bert_tokenizer.decode(output, skip_special_tokens=True)

    def test_unguided_skips_gaze_scoring(self, mock_causal_lm, bert_tokenizer):
        """With lam=0, gaze_predictor.score_vocabulary is never called."""
        gaze_pred = MagicMock()
        prompt_ids = bert_tokenizer.encode("Hello world", return_tensors="pt").squeeze(0)

        config = ARGazeConfig(
            model_name="mock", lam=0.0, top_k=10, max_new_tokens=3,
        )

        output = ar_gaze_guided_generate(
            mock_causal_lm, gaze_pred, bert_tokenizer, prompt_ids, config,
        )

        assert gaze_pred.score_vocabulary.call_count == 0
        assert len(output) >= len(prompt_ids)


# ---- Metrics Integration Tests ----


@pytest.mark.integration
class TestMetricsIntegration:

    @pytest.mark.slow
    def test_metrics_on_generated_output(self, mock_mdlm, bert_tokenizer):
        """Metrics pipeline runs on decoded GazeDiffuse output."""
        model = _cached_trained_predictor(bert_tokenizer)
        prompts = [
            "The cat sat on the",
            "Scientists discovered a new",
            "Education is important for",
        ]

        config = GazeDiffuseConfig(
            lam=-1.0, steps=3, gen_length=16, top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )

        texts = []
        for p in prompts:
            prompt_ids = bert_tokenizer.encode(p, return_tensors="pt").squeeze(0)
            output = gaze_guided_diffuse(mock_mdlm, model, bert_tokenizer, prompt_ids, config)
            text = bert_tokenizer.decode(output, skip_special_tokens=True)
            texts.append(text)

        result = evaluate_generations(texts, compute_ppl=False)

        assert result.n_samples == 3
        assert isinstance(result.fkgl_mean, float)
        assert isinstance(result.ari_mean, float)
        assert result.fk_sentence_variance >= 0
        assert result.mauve_score is None


# ---- End-to-End Pipeline Test ----


@pytest.mark.integration
class TestEndToEnd:

    @pytest.mark.slow
    def test_full_pipeline_train_generate_evaluate(self, bert_tokenizer, tmp_path, mock_mdlm):
        """Complete pipeline: train → checkpoint → load → generate → evaluate."""
        from src.gaze_predictor import load_trained_predictor

        # Phase 1: Train
        torch.manual_seed(42)
        examples = _make_synthetic_examples(20)
        dataset = GazeDataset(examples, bert_tokenizer, max_length=32, normalize=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        config = GazePredictorConfig(
            learning_rate=1e-4, batch_size=4, epochs=4, max_seq_length=32,
        )
        model = GazePredictor(config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        losses = []
        for _epoch in range(4):
            epoch_loss = 0.0
            n = 0
            for batch in loader:
                preds = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(preds, batch["labels"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            losses.append(epoch_loss / n)

        assert losses[-1] < losses[0], f"Loss did not decrease overall: {losses}"

        # Phase 2: Save and load checkpoint
        model.eval()
        ckpt_path = tmp_path / "e2e_test.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "normalization": dataset.get_normalization_stats(),
            },
            ckpt_path,
        )

        loaded_model, norm_stats = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        assert isinstance(loaded_model, GazePredictor)

        # Phase 3: Generate
        prompt_ids = bert_tokenizer.encode("The discovery of a new", return_tensors="pt").squeeze(0)
        gen_config = GazeDiffuseConfig(
            lam=-1.0, steps=3, gen_length=8, top_k=5,
            mask_token_id=bert_tokenizer.mask_token_id,
        )

        output = gaze_guided_diffuse(mock_mdlm, loaded_model, bert_tokenizer, prompt_ids, gen_config)
        assert output.shape == (len(prompt_ids) + 8,)
        assert torch.equal(output[:len(prompt_ids)], prompt_ids)
        assert (output == bert_tokenizer.mask_token_id).sum() == 0

        # Phase 4: Decode and evaluate
        text = bert_tokenizer.decode(output, skip_special_tokens=True)
        assert len(text) > 0

        result = evaluate_generations([text], compute_ppl=False)
        assert result.n_samples == 1
        assert isinstance(result.fkgl_mean, float)
        assert result.fk_sentence_variance >= 0
