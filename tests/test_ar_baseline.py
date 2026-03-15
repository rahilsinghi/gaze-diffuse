"""Tests for AR gaze guidance baseline (src/ar_baseline.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.ar_baseline import ARGazeConfig, generate_ar_samples


class TestARGazeConfig:
    """Test AR baseline configuration."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        config = ARGazeConfig()
        assert config.model_name == "gpt2-medium"
        assert config.lam == -1.0
        assert config.top_k == 50
        assert config.max_new_tokens == 128

    @pytest.mark.unit
    def test_frozen(self) -> None:
        config = ARGazeConfig()
        with pytest.raises(AttributeError):
            config.lam = 0.5  # type: ignore[misc]

    @pytest.mark.unit
    def test_custom_values(self) -> None:
        config = ARGazeConfig(
            model_name="gpt2",
            lam=2.0,
            top_k=100,
            max_new_tokens=64,
        )
        assert config.model_name == "gpt2"
        assert config.lam == 2.0

    @pytest.mark.unit
    def test_negative_top_k_allowed(self) -> None:
        """ARGazeConfig is a frozen dataclass with no validation,
        so negative top_k is accepted but would fail at generation time."""
        config = ARGazeConfig(top_k=-5)
        assert config.top_k == -5

    @pytest.mark.unit
    def test_zero_max_new_tokens(self) -> None:
        """Zero max_new_tokens is accepted by config (generation produces nothing)."""
        config = ARGazeConfig(max_new_tokens=0)
        assert config.max_new_tokens == 0

    @pytest.mark.unit
    def test_lam_zero_means_unguided(self) -> None:
        """When lam=0, the config represents unguided generation."""
        config = ARGazeConfig(lam=0.0)
        assert config.lam == 0.0
        # Per the docstring: lam=0 means unguided
        assert config.lam == 0.0

    @pytest.mark.unit
    def test_temperature_defaults(self) -> None:
        """Temperature fields should default to 1.0."""
        config = ARGazeConfig()
        assert config.temperature == 1.0
        assert config.gaze_temperature == 1.0

    @pytest.mark.unit
    def test_gaze_temperature_custom(self) -> None:
        """Custom gaze_temperature values are stored correctly."""
        config = ARGazeConfig(gaze_temperature=0.5)
        assert config.gaze_temperature == 0.5

        config_high = ARGazeConfig(gaze_temperature=2.0)
        assert config_high.gaze_temperature == 2.0


class TestGenerateArSamples:
    """Test generate_ar_samples result format using mocked internals."""

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_result_format(self, mock_generate: MagicMock) -> None:
        """generate_ar_samples should return dicts with expected keys."""
        # Mock the low-level generate function to return a tensor
        prompt_ids = torch.tensor([1, 2, 3, 4, 5])
        generated_ids = torch.tensor([1, 2, 3, 4, 5, 100, 200, 300])
        mock_generate.return_value = generated_ids

        # Mock LM with parameters() returning a device
        mock_lm = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_lm.parameters.return_value = iter([mock_param])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = prompt_ids
        mock_tokenizer.decode.return_value = "some generated text"

        # Mock gaze predictor (not used since ar_gaze_guided_generate is mocked)
        mock_gaze = MagicMock()

        config = ARGazeConfig(lam=-1.0, max_new_tokens=3)
        results = generate_ar_samples(
            lm=mock_lm,
            gaze_predictor=mock_gaze,
            tokenizer=mock_tokenizer,
            prompts=["Hello world"],
            config=config,
            n_samples_per_prompt=2,
        )

        assert len(results) == 2
        expected_keys = {
            "prompt_idx",
            "sample_idx",
            "prompt",
            "generation",
            "full_text",
            "lam",
            "method",
            "model",
        }
        for result in results:
            assert set(result.keys()) == expected_keys
            assert result["method"] == "ar_gaze"
            assert result["lam"] == -1.0
            assert result["prompt"] == "Hello world"
            assert result["model"] == config.model_name

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_multiple_prompts(self, mock_generate: MagicMock) -> None:
        """generate_ar_samples should produce results for each prompt x sample."""
        mock_generate.return_value = torch.tensor([1, 2, 3, 4, 5, 100])

        mock_lm = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_lm.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = torch.tensor([1, 2, 3, 4, 5])
        mock_tokenizer.decode.return_value = "text"

        mock_gaze = MagicMock()
        config = ARGazeConfig()

        prompts = ["Prompt A", "Prompt B", "Prompt C"]
        results = generate_ar_samples(
            lm=mock_lm,
            gaze_predictor=mock_gaze,
            tokenizer=mock_tokenizer,
            prompts=prompts,
            config=config,
            n_samples_per_prompt=3,
        )

        assert len(results) == 9  # 3 prompts x 3 samples
        # Check prompt_idx increments correctly
        prompt_indices = [r["prompt_idx"] for r in results]
        assert prompt_indices == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # Check sample_idx cycles correctly
        sample_indices = [r["sample_idx"] for r in results]
        assert sample_indices == [0, 1, 2, 0, 1, 2, 0, 1, 2]
