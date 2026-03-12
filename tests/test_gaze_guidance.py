"""Tests for the GazeDiffuse sampler (src/gaze_guidance.py)."""

from __future__ import annotations

import pytest
import torch

from src.gaze_guidance import (
    GazeDiffuseConfig,
    confidence_schedule,
    gaze_guided_diffuse,
)


class TestConfidenceSchedule:
    """Test the confidence-based unmasking schedule."""

    @pytest.mark.unit
    def test_reveals_at_least_one(self) -> None:
        """Always reveal at least 1 token."""
        result = confidence_schedule(step=0, total_steps=10, n_masked=5)
        assert result >= 1

    @pytest.mark.unit
    def test_reveals_all_on_last_step(self) -> None:
        """On the last step, reveal all remaining masked tokens."""
        result = confidence_schedule(step=9, total_steps=10, n_masked=3)
        assert result == 3

    @pytest.mark.unit
    def test_does_not_exceed_masked(self) -> None:
        """Never try to reveal more tokens than are masked."""
        result = confidence_schedule(step=0, total_steps=2, n_masked=3)
        assert result <= 3

    @pytest.mark.unit
    def test_linear_schedule(self) -> None:
        """With equal steps, should reveal roughly equal tokens per step."""
        n_masked = 100
        total_steps = 10
        # First step: reveal ~10
        first = confidence_schedule(step=0, total_steps=total_steps, n_masked=n_masked)
        assert 5 <= first <= 15

    @pytest.mark.unit
    def test_zero_masked(self) -> None:
        """Edge case: no masked tokens left."""
        result = confidence_schedule(step=5, total_steps=10, n_masked=0)
        assert result == 0


class TestGazeDiffuseConfig:
    """Test GazeDiffuseConfig dataclass."""

    @pytest.mark.unit
    def test_default_values(self) -> None:
        config = GazeDiffuseConfig()
        assert config.lam == -1.0
        assert config.steps == 64
        assert config.top_k == 100
        assert config.gen_length == 128

    @pytest.mark.unit
    def test_custom_values(self) -> None:
        config = GazeDiffuseConfig(lam=2.0, steps=32, gen_length=64)
        assert config.lam == 2.0
        assert config.steps == 32
        assert config.gen_length == 64

    @pytest.mark.unit
    def test_frozen(self) -> None:
        config = GazeDiffuseConfig()
        with pytest.raises(AttributeError):
            config.lam = 0.5  # type: ignore[misc]


class TestGazeGuidedDiffuse:
    """Test the core GazeDiffuse sampling function."""

    @pytest.mark.unit
    def test_output_shape(self, mock_mdlm, sample_prompt_ids) -> None:
        """Output length = prompt + gen_length."""
        from tests.conftest import MockMDLM

        # Create a simple mock gaze predictor
        from unittest.mock import MagicMock

        gaze_pred = MagicMock()
        gaze_pred.score_vocabulary.return_value = torch.randn(100)
        gaze_pred.parameters.return_value = iter([torch.zeros(1)])

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        config = GazeDiffuseConfig(
            lam=0.0,  # Unguided for speed
            steps=2,
            gen_length=10,
            mask_token_id=tokenizer.mask_token_id,
        )

        output = gaze_guided_diffuse(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred,
            tokenizer=tokenizer,
            prompt_ids=sample_prompt_ids,
            config=config,
        )

        expected_len = len(sample_prompt_ids) + config.gen_length
        assert output.shape == (expected_len,)

    @pytest.mark.unit
    def test_prompt_preserved(self, mock_mdlm, sample_prompt_ids) -> None:
        """Prompt tokens should not be modified."""
        from unittest.mock import MagicMock

        gaze_pred = MagicMock()
        gaze_pred.score_vocabulary.return_value = torch.randn(100)
        gaze_pred.parameters.return_value = iter([torch.zeros(1)])

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        config = GazeDiffuseConfig(
            lam=0.0,
            steps=2,
            gen_length=10,
            mask_token_id=tokenizer.mask_token_id,
        )

        output = gaze_guided_diffuse(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred,
            tokenizer=tokenizer,
            prompt_ids=sample_prompt_ids,
            config=config,
        )

        prompt_len = len(sample_prompt_ids)
        assert torch.equal(output[:prompt_len], sample_prompt_ids)

    @pytest.mark.unit
    def test_no_masks_remaining(self, mock_mdlm, sample_prompt_ids) -> None:
        """After sampling, no mask tokens should remain."""
        from unittest.mock import MagicMock

        gaze_pred = MagicMock()
        gaze_pred.score_vocabulary.return_value = torch.randn(100)
        gaze_pred.parameters.return_value = iter([torch.zeros(1)])

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        mask_id = tokenizer.mask_token_id

        config = GazeDiffuseConfig(
            lam=0.0,
            steps=20,  # Enough steps to reveal all
            gen_length=10,
            mask_token_id=mask_id,
        )

        output = gaze_guided_diffuse(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred,
            tokenizer=tokenizer,
            prompt_ids=sample_prompt_ids,
            config=config,
        )

        assert (output == mask_id).sum().item() == 0
