"""Tests for gaze predictor model (src/gaze_predictor.py)."""

from __future__ import annotations

import pytest
import torch

from src.gaze_predictor import GazePredictor, GazePredictorConfig


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
