"""Tests for gaze predictor model (src/gaze_predictor.py)."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from src.gaze_predictor import GazePredictor, GazePredictorConfig, load_trained_predictor


class TestGazePredictorConfig:
    """Test GazePredictorConfig defaults and immutability."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        config = GazePredictorConfig()
        assert config.bert_model == "bert-base-uncased"
        assert config.hidden_size == 768
        assert config.dropout == 0.1
        assert config.epochs == 3

    @pytest.mark.unit
    def test_frozen(self) -> None:
        config = GazePredictorConfig()
        with pytest.raises(AttributeError):
            config.epochs = 10  # type: ignore[misc]


class TestGazePredictor:
    """Test GazePredictor model forward pass."""

    @pytest.mark.slow
    def test_forward_shape(self, gaze_predictor_config) -> None:
        """Forward pass returns correct shape [batch]."""
        model = GazePredictor(gaze_predictor_config)
        batch_size = 4
        seq_len = 32

        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        output = model(input_ids, attention_mask)
        assert output.shape == (batch_size,)

    @pytest.mark.slow
    def test_forward_no_nan(self, gaze_predictor_config) -> None:
        """Output should not contain NaN values."""
        model = GazePredictor(gaze_predictor_config)
        input_ids = torch.randint(0, 30522, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        output = model(input_ids, attention_mask)
        assert not torch.isnan(output).any()

    @pytest.mark.slow
    def test_gradient_flows(self, gaze_predictor_config) -> None:
        """Gradients should flow through the model."""
        model = GazePredictor(gaze_predictor_config)
        input_ids = torch.randint(0, 30522, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        labels = torch.tensor([250.0, 180.0])

        output = model(input_ids, attention_mask)
        loss = torch.nn.functional.mse_loss(output, labels)
        loss.backward()

        # Check that BERT parameters received gradients
        for param in model.bert.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break


class TestGazeDataset:
    """Test GazeDataset tokenization and output format."""

    @pytest.mark.unit
    def test_dataset_item_keys(self) -> None:
        """Each item should have input_ids, attention_mask, labels."""
        from src.data.geco import GazeDataset, GazeExample
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        examples = [
            GazeExample(
                word="cat",
                left_context=["the", "big"],
                right_context=["sat", "on"],
                fixation_duration_ms=200.0,
                sentence_id=1,
            ),
            GazeExample(
                word="epistemological",
                left_context=["the"],
                right_context=["implications", "of"],
                fixation_duration_ms=450.0,
                sentence_id=2,
            ),
        ]

        dataset = GazeDataset(examples, tokenizer, max_length=32)
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["labels"].ndim == 0  # Scalar

    @pytest.mark.unit
    def test_normalization(self) -> None:
        """Normalized targets should have ~zero mean."""
        from src.data.geco import GazeDataset, GazeExample
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        examples = [
            GazeExample(
                word=f"word{i}",
                left_context=["the"],
                right_context=["is"],
                fixation_duration_ms=100.0 + i * 50,
                sentence_id=i,
            )
            for i in range(20)
        ]

        dataset = GazeDataset(examples, tokenizer, max_length=32, normalize=True)
        labels = [dataset[i]["labels"].item() for i in range(len(dataset))]

        import numpy as np

        mean_label = np.mean(labels)
        assert abs(mean_label) < 0.5  # Roughly centered


class TestScoreTokens:
    """Tests for GazePredictor.score_tokens — unit level, no training required."""

    @pytest.mark.slow
    def test_score_tokens_returns_tensor(self, gaze_predictor_config, bert_tokenizer) -> None:
        """score_tokens returns a torch.Tensor."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "The cat sat on the mat", return_tensors="pt"
        ).squeeze(0)
        scores = model.score_tokens(input_ids, bert_tokenizer)
        assert isinstance(scores, torch.Tensor)

    @pytest.mark.slow
    def test_score_tokens_length_matches_input(self, gaze_predictor_config, bert_tokenizer) -> None:
        """Output tensor length equals the number of input tokens."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "The cat sat on the mat", return_tensors="pt"
        ).squeeze(0)
        scores = model.score_tokens(input_ids, bert_tokenizer)
        assert scores.shape[0] == len(input_ids)

    @pytest.mark.slow
    def test_score_tokens_values_are_finite(self, gaze_predictor_config, bert_tokenizer) -> None:
        """No inf or nan values in the output."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "Scientists discovered a rare compound", return_tensors="pt"
        ).squeeze(0)
        scores = model.score_tokens(input_ids, bert_tokenizer)
        assert torch.isfinite(scores).all()

    @pytest.mark.slow
    def test_score_tokens_special_tokens_are_zero(self, gaze_predictor_config, bert_tokenizer) -> None:
        """CLS and SEP positions score zero because they are skipped."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "The quick brown fox", return_tensors="pt"
        ).squeeze(0)
        # BERT tokenizer wraps with [CLS] at 0 and [SEP] at -1
        scores = model.score_tokens(input_ids, bert_tokenizer)
        assert scores[0].item() == 0.0, "CLS token should score zero"
        assert scores[-1].item() == 0.0, "SEP token should score zero"

    @pytest.mark.slow
    def test_score_tokens_with_single_content_token(self, gaze_predictor_config, bert_tokenizer) -> None:
        """Edge case: sequence with only one non-special token still returns a tensor."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        # Manually build [CLS] word [SEP] — 3 tokens total
        cls_id = bert_tokenizer.cls_token_id
        sep_id = bert_tokenizer.sep_token_id
        word_id = bert_tokenizer.encode("cat", add_special_tokens=False)[0]
        input_ids = torch.tensor([cls_id, word_id, sep_id])
        scores = model.score_tokens(input_ids, bert_tokenizer)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()
        # The single content token must have received a score
        assert scores[1].item() != 0.0

    @pytest.mark.slow
    def test_score_tokens_all_special_returns_zeros(self, gaze_predictor_config, bert_tokenizer) -> None:
        """Sequence composed entirely of special tokens returns an all-zero tensor."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        cls_id = bert_tokenizer.cls_token_id
        sep_id = bert_tokenizer.sep_token_id
        pad_id = bert_tokenizer.pad_token_id
        input_ids = torch.tensor([cls_id, sep_id, pad_id])
        scores = model.score_tokens(input_ids, bert_tokenizer)
        assert scores.shape == (3,)
        assert (scores == 0.0).all()


class TestScoreVocabulary:
    """Tests for GazePredictor.score_vocabulary — unit level, no training required."""

    @pytest.mark.slow
    def test_score_vocabulary_returns_correct_length(self, gaze_predictor_config, bert_tokenizer) -> None:
        """Output length matches the number of vocabulary candidates."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "The cat sat on the mat", return_tensors="pt"
        ).squeeze(0)
        candidates = torch.tensor([100, 200, 300, 400, 500])
        scores = model.score_vocabulary(
            input_ids, position=3, vocab_candidates=candidates, tokenizer=bert_tokenizer
        )
        assert len(scores) == len(candidates)

    @pytest.mark.slow
    def test_score_vocabulary_values_are_finite(self, gaze_predictor_config, bert_tokenizer) -> None:
        """No inf or nan values in vocabulary scores."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "The cat sat on the mat", return_tensors="pt"
        ).squeeze(0)
        candidates = torch.tensor([100, 200, 300, 400, 500])
        scores = model.score_vocabulary(
            input_ids, position=3, vocab_candidates=candidates, tokenizer=bert_tokenizer
        )
        assert torch.isfinite(scores).all()

    @pytest.mark.slow
    def test_score_vocabulary_shape_is_1d(self, gaze_predictor_config, bert_tokenizer) -> None:
        """Output is a 1-D tensor of shape [n_candidates]."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "Dogs run very fast today", return_tensors="pt"
        ).squeeze(0)
        candidates = torch.tensor([1000, 2000, 3000])
        scores = model.score_vocabulary(
            input_ids, position=2, vocab_candidates=candidates, tokenizer=bert_tokenizer
        )
        assert scores.ndim == 1
        assert scores.shape[0] == 3

    @pytest.mark.slow
    def test_score_vocabulary_single_candidate(self, gaze_predictor_config, bert_tokenizer) -> None:
        """Edge case: a single candidate returns a tensor of length 1."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode(
            "The cat sat on the mat", return_tensors="pt"
        ).squeeze(0)
        candidates = torch.tensor([1996])  # "the"
        scores = model.score_vocabulary(
            input_ids, position=1, vocab_candidates=candidates, tokenizer=bert_tokenizer
        )
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()

    @pytest.mark.slow
    def test_score_vocabulary_at_boundary_position(self, gaze_predictor_config, bert_tokenizer) -> None:
        """Position at the last index produces finite scores (right context is empty)."""
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        input_ids = bert_tokenizer.encode("cat sat", return_tensors="pt").squeeze(0)
        candidates = torch.tensor([200, 300])
        last_pos = len(input_ids) - 1
        scores = model.score_vocabulary(
            input_ids, position=last_pos, vocab_candidates=candidates, tokenizer=bert_tokenizer
        )
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()


class TestLoadTrainedPredictor:
    """Tests for load_trained_predictor — checkpoint loading paths."""

    @pytest.mark.slow
    def test_load_checkpoint_with_dict_config(self, gaze_predictor_config, tmp_path) -> None:
        """Config stored as dict (asdict) is reconstructed into GazePredictorConfig."""
        model = GazePredictor(gaze_predictor_config)
        ckpt_path = tmp_path / "dict_config_ckpt.pt"
        torch.save(
            {
                "config": asdict(gaze_predictor_config),
                "model_state_dict": model.state_dict(),
                "normalization": (274.3, 150.0),
            },
            ckpt_path,
        )

        loaded_model, norm_stats = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        assert isinstance(loaded_model, GazePredictor)
        assert isinstance(loaded_model.config, GazePredictorConfig)
        assert loaded_model.config == gaze_predictor_config

    @pytest.mark.slow
    def test_load_checkpoint_normalization_stats_returned(self, gaze_predictor_config, tmp_path) -> None:
        """Normalization tuple stored in checkpoint is returned unchanged."""
        model = GazePredictor(gaze_predictor_config)
        expected_norm = (312.7, 89.4)
        ckpt_path = tmp_path / "norm_ckpt.pt"
        torch.save(
            {
                "config": asdict(gaze_predictor_config),
                "model_state_dict": model.state_dict(),
                "normalization": expected_norm,
            },
            ckpt_path,
        )

        _, norm_stats = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        assert norm_stats == expected_norm

    @pytest.mark.slow
    def test_load_checkpoint_missing_normalization_falls_back(
        self, gaze_predictor_config, tmp_path
    ) -> None:
        """When normalization key is absent, falls back to (0.0, 1.0)."""
        model = GazePredictor(gaze_predictor_config)
        ckpt_path = tmp_path / "no_norm_ckpt.pt"
        torch.save(
            {
                "config": asdict(gaze_predictor_config),
                "model_state_dict": model.state_dict(),
                # 'normalization' key intentionally omitted
            },
            ckpt_path,
        )

        _, norm_stats = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        assert norm_stats == (0.0, 1.0)

    @pytest.mark.slow
    def test_load_checkpoint_model_is_in_eval_mode(self, gaze_predictor_config, tmp_path) -> None:
        """Loaded model is automatically put into eval mode."""
        model = GazePredictor(gaze_predictor_config)
        ckpt_path = tmp_path / "eval_mode_ckpt.pt"
        torch.save(
            {
                "config": asdict(gaze_predictor_config),
                "model_state_dict": model.state_dict(),
                "normalization": (200.0, 80.0),
            },
            ckpt_path,
        )

        loaded_model, _ = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        assert not loaded_model.training

    @pytest.mark.slow
    def test_load_checkpoint_device_cpu(self, gaze_predictor_config, tmp_path) -> None:
        """Model parameters are on CPU when device=torch.device('cpu') is passed."""
        model = GazePredictor(gaze_predictor_config)
        ckpt_path = tmp_path / "cpu_device_ckpt.pt"
        torch.save(
            {
                "config": asdict(gaze_predictor_config),
                "model_state_dict": model.state_dict(),
                "normalization": (200.0, 80.0),
            },
            ckpt_path,
        )

        loaded_model, _ = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        for param in loaded_model.parameters():
            assert param.device.type == "cpu"
            break

    @pytest.mark.slow
    def test_load_checkpoint_weights_match_original(self, gaze_predictor_config, tmp_path) -> None:
        """Loaded model produces identical predictions to the original model."""
        torch.manual_seed(0)
        model = GazePredictor(gaze_predictor_config)
        model.eval()
        ckpt_path = tmp_path / "weights_match_ckpt.pt"
        torch.save(
            {
                "config": asdict(gaze_predictor_config),
                "model_state_dict": model.state_dict(),
                "normalization": (200.0, 80.0),
            },
            ckpt_path,
        )

        input_ids = torch.randint(0, 30522, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        with torch.no_grad():
            original_preds = model(input_ids, attention_mask)

        loaded_model, _ = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        with torch.no_grad():
            loaded_preds = loaded_model(input_ids, attention_mask)

        assert torch.allclose(original_preds, loaded_preds, atol=1e-6)

    @pytest.mark.slow
    def test_load_checkpoint_with_dataclass_config(self, gaze_predictor_config, tmp_path) -> None:
        """Config stored as a dataclass object (not dict) also loads correctly."""
        model = GazePredictor(gaze_predictor_config)
        ckpt_path = tmp_path / "dataclass_config_ckpt.pt"
        torch.save(
            {
                "config": gaze_predictor_config,  # stored as dataclass, not dict
                "model_state_dict": model.state_dict(),
                "normalization": (250.0, 100.0),
            },
            ckpt_path,
        )

        loaded_model, norm_stats = load_trained_predictor(ckpt_path, device=torch.device("cpu"))
        assert isinstance(loaded_model, GazePredictor)
        assert norm_stats == (250.0, 100.0)

    @pytest.mark.slow
    def test_load_checkpoint_auto_device_detection(self, gaze_predictor_config, tmp_path) -> None:
        """When device=None, load_trained_predictor auto-selects a device without error."""
        model = GazePredictor(gaze_predictor_config)
        ckpt_path = tmp_path / "auto_device_ckpt.pt"
        torch.save(
            {
                "config": asdict(gaze_predictor_config),
                "model_state_dict": model.state_dict(),
                "normalization": (200.0, 80.0),
            },
            ckpt_path,
        )

        # device=None triggers auto-detection (cuda → mps → cpu)
        loaded_model, norm_stats = load_trained_predictor(ckpt_path, device=None)
        assert isinstance(loaded_model, GazePredictor)
        assert not loaded_model.training
        # Model must be on some valid device
        device_type = next(loaded_model.parameters()).device.type
        assert device_type in {"cpu", "cuda", "mps"}
