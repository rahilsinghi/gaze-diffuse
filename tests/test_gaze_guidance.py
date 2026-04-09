"""Tests for the GazeDiffuse sampler (src/gaze_guidance.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from src.gaze_guidance import (
    GazeDiffuseConfig,
    confidence_schedule,
    gaze_guided_diffuse,
    gaze_guided_diffuse_mdlm,
    generate_samples,
    save_generations,
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


# ---------------------------------------------------------------------------
# MockMDLMWrapper — mimics MDLMWrapper interface without real model weights
# ---------------------------------------------------------------------------

class MockMDLMWrapper:
    """Minimal MDLMWrapper stand-in for testing gaze_guided_diffuse_mdlm."""

    mask_index: int = 103

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def tokenizer(self):
        return self._tokenizer

    def get_log_probs(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Return random log-probs with correct shape [batch, seq, vocab]."""
        batch, seq = x.shape
        vocab = self._tokenizer.vocab_size
        logits = torch.randn(batch, seq, vocab)
        return F.log_softmax(logits, dim=-1)

    def get_timesteps(self, n: int) -> torch.Tensor:
        return torch.linspace(1.0, 1e-5, n + 1)

    def create_masked_input(
        self, prompt_ids: torch.Tensor, gen_length: int
    ) -> torch.Tensor:
        masks = torch.full((gen_length,), self.mask_index, dtype=torch.long)
        return torch.cat([prompt_ids, masks])


# ---------------------------------------------------------------------------
# Fixtures used across the new test classes
# ---------------------------------------------------------------------------

@pytest.fixture
def gaze_pred_mock() -> MagicMock:
    """A gaze predictor that returns random scores for any vocabulary batch."""
    mock = MagicMock()
    mock.score_vocabulary.return_value = torch.randn(10)  # top_k=10
    mock.parameters.return_value = iter([torch.zeros(1)])
    return mock


@pytest.fixture
def mock_mdlm_wrapper(bert_tokenizer) -> MockMDLMWrapper:
    return MockMDLMWrapper(bert_tokenizer)


@pytest.fixture
def fast_config(bert_tokenizer) -> GazeDiffuseConfig:
    """Config tuned for fast tests: small gen_length, few steps."""
    return GazeDiffuseConfig(
        lam=-1.0,
        steps=2,
        gen_length=4,
        top_k=10,
        mask_token_id=bert_tokenizer.mask_token_id,
    )


# ---------------------------------------------------------------------------
# mask_token_id fallback (line 105)
# ---------------------------------------------------------------------------

class TestMaskIdFallback:
    """When mask_token_id is None in config, fall back to tokenizer value."""

    @pytest.mark.unit
    def test_mask_id_fallback(
        self,
        mock_mdlm,
        sample_prompt_ids,
        bert_tokenizer,
        gaze_pred_mock,
    ) -> None:
        # mask_token_id deliberately omitted (defaults to None)
        config = GazeDiffuseConfig(lam=0.0, steps=2, gen_length=4, top_k=10)
        assert config.mask_token_id is None  # confirm precondition

        output = gaze_guided_diffuse(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompt_ids=sample_prompt_ids,
            config=config,
        )

        expected_len = len(sample_prompt_ids) + config.gen_length
        assert output.shape == (expected_len,)
        # No mask tokens remain — the fallback mask_id was used correctly
        assert (output == bert_tokenizer.mask_token_id).sum().item() == 0


# ---------------------------------------------------------------------------
# gaze_guided_diffuse_mdlm (lines 195-261)
# ---------------------------------------------------------------------------

class TestGazeGuidedDiffuseMDLM:
    """Tests for the MDLMWrapper-based sampling path."""

    @pytest.mark.unit
    def test_output_shape(
        self,
        mock_mdlm_wrapper,
        sample_prompt_ids,
        gaze_pred_mock,
        fast_config,
    ) -> None:
        output = gaze_guided_diffuse_mdlm(
            mdlm_wrapper=mock_mdlm_wrapper,
            gaze_predictor=gaze_pred_mock,
            prompt_ids=sample_prompt_ids,
            config=fast_config,
        )
        expected = len(sample_prompt_ids) + fast_config.gen_length
        assert output.shape == (expected,)

    @pytest.mark.unit
    def test_prompt_preserved(
        self,
        mock_mdlm_wrapper,
        sample_prompt_ids,
        gaze_pred_mock,
        fast_config,
    ) -> None:
        output = gaze_guided_diffuse_mdlm(
            mdlm_wrapper=mock_mdlm_wrapper,
            gaze_predictor=gaze_pred_mock,
            prompt_ids=sample_prompt_ids,
            config=fast_config,
        )
        assert torch.equal(output[: len(sample_prompt_ids)], sample_prompt_ids)

    @pytest.mark.unit
    def test_no_masks_remaining(
        self,
        mock_mdlm_wrapper,
        sample_prompt_ids,
        gaze_pred_mock,
        bert_tokenizer,
    ) -> None:
        # Use enough steps to drain all masked positions
        config = GazeDiffuseConfig(
            lam=-1.0,
            steps=20,
            gen_length=4,
            top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )
        output = gaze_guided_diffuse_mdlm(
            mdlm_wrapper=mock_mdlm_wrapper,
            gaze_predictor=gaze_pred_mock,
            prompt_ids=sample_prompt_ids,
            config=config,
        )
        assert (output == mock_mdlm_wrapper.mask_index).sum().item() == 0

    @pytest.mark.unit
    def test_unguided_lam_zero(
        self,
        mock_mdlm_wrapper,
        sample_prompt_ids,
        gaze_pred_mock,
        bert_tokenizer,
    ) -> None:
        """lam=0 skips gaze scoring but must still produce valid output."""
        config = GazeDiffuseConfig(
            lam=0.0,
            steps=2,
            gen_length=4,
            top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )
        output = gaze_guided_diffuse_mdlm(
            mdlm_wrapper=mock_mdlm_wrapper,
            gaze_predictor=gaze_pred_mock,
            prompt_ids=sample_prompt_ids,
            config=config,
        )
        expected = len(sample_prompt_ids) + config.gen_length
        assert output.shape == (expected,)
        # gaze predictor must NOT have been called when lam == 0
        gaze_pred_mock.score_vocabulary.assert_not_called()

    @pytest.mark.unit
    def test_positive_lam(
        self,
        mock_mdlm_wrapper,
        sample_prompt_ids,
        gaze_pred_mock,
        bert_tokenizer,
    ) -> None:
        """lam>0 (harder text) path should also complete without error."""
        config = GazeDiffuseConfig(
            lam=1.0,
            steps=2,
            gen_length=4,
            top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )
        output = gaze_guided_diffuse_mdlm(
            mdlm_wrapper=mock_mdlm_wrapper,
            gaze_predictor=gaze_pred_mock,
            prompt_ids=sample_prompt_ids,
            config=config,
        )
        assert output.shape == (len(sample_prompt_ids) + config.gen_length,)


# ---------------------------------------------------------------------------
# generate_samples (lines 276-315)
# ---------------------------------------------------------------------------

class TestGenerateSamples:
    """Tests for the multi-prompt generation loop."""

    @pytest.fixture
    def gen_config(self, bert_tokenizer) -> GazeDiffuseConfig:
        return GazeDiffuseConfig(
            lam=-1.0,
            steps=2,
            gen_length=4,
            top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )

    @pytest.mark.unit
    def test_returns_correct_count(
        self,
        mock_mdlm,
        gaze_pred_mock,
        bert_tokenizer,
        gen_config,
    ) -> None:
        prompts = ["The cat sat", "Dogs run fast"]
        results = generate_samples(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompts=prompts,
            config=gen_config,
            n_samples_per_prompt=2,
        )
        # 2 prompts × 2 samples each = 4 results
        assert len(results) == 4

    @pytest.mark.unit
    def test_result_keys(
        self,
        mock_mdlm,
        gaze_pred_mock,
        bert_tokenizer,
        gen_config,
    ) -> None:
        prompts = ["The cat sat"]
        results = generate_samples(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompts=prompts,
            config=gen_config,
            n_samples_per_prompt=1,
        )
        required_keys = {"prompt_idx", "sample_idx", "prompt", "generation", "full_text", "lam", "steps"}
        assert required_keys.issubset(results[0].keys())

    @pytest.mark.unit
    def test_lam_stored_in_results(
        self,
        mock_mdlm,
        gaze_pred_mock,
        bert_tokenizer,
    ) -> None:
        config = GazeDiffuseConfig(
            lam=2.5,
            steps=2,
            gen_length=4,
            top_k=10,
            mask_token_id=bert_tokenizer.mask_token_id,
        )
        results = generate_samples(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompts=["Hello world"],
            config=config,
            n_samples_per_prompt=1,
        )
        assert results[0]["lam"] == 2.5

    @pytest.mark.unit
    def test_prompt_index_increments(
        self,
        mock_mdlm,
        gaze_pred_mock,
        bert_tokenizer,
        gen_config,
    ) -> None:
        prompts = ["Prompt one", "Prompt two", "Prompt three"]
        results = generate_samples(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompts=prompts,
            config=gen_config,
            n_samples_per_prompt=1,
        )
        prompt_indices = [r["prompt_idx"] for r in results]
        assert prompt_indices == [0, 1, 2]

    @pytest.mark.unit
    def test_sample_index_resets_per_prompt(
        self,
        mock_mdlm,
        gaze_pred_mock,
        bert_tokenizer,
        gen_config,
    ) -> None:
        prompts = ["First prompt", "Second prompt"]
        results = generate_samples(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompts=prompts,
            config=gen_config,
            n_samples_per_prompt=2,
        )
        # sample_idx should be 0,1 for prompt 0 and 0,1 for prompt 1
        assert [r["sample_idx"] for r in results] == [0, 1, 0, 1]

    @pytest.mark.unit
    def test_full_text_contains_prompt(
        self,
        mock_mdlm,
        gaze_pred_mock,
        bert_tokenizer,
        gen_config,
    ) -> None:
        prompt = "The ocean is vast"
        results = generate_samples(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompts=[prompt],
            config=gen_config,
            n_samples_per_prompt=1,
        )
        assert results[0]["full_text"].startswith(prompt)

    @pytest.mark.unit
    def test_empty_prompts_returns_empty(
        self,
        mock_mdlm,
        gaze_pred_mock,
        bert_tokenizer,
        gen_config,
    ) -> None:
        results = generate_samples(
            mdlm=mock_mdlm,
            gaze_predictor=gaze_pred_mock,
            tokenizer=bert_tokenizer,
            prompts=[],
            config=gen_config,
            n_samples_per_prompt=2,
        )
        assert results == []


# ---------------------------------------------------------------------------
# save_generations (lines 323-330)
# ---------------------------------------------------------------------------

class TestSaveGenerations:
    """Tests for the JSONL save utility."""

    @pytest.fixture
    def sample_results(self) -> list[dict]:
        return [
            {
                "prompt_idx": 0,
                "sample_idx": 0,
                "prompt": "The cat sat",
                "generation": "on the mat today",
                "full_text": "The cat sat on the mat today",
                "lam": -1.0,
                "steps": 2,
            },
            {
                "prompt_idx": 0,
                "sample_idx": 1,
                "prompt": "The cat sat",
                "generation": "quietly by the door",
                "full_text": "The cat sat quietly by the door",
                "lam": -1.0,
                "steps": 2,
            },
        ]

    @pytest.mark.unit
    def test_creates_file(self, sample_results, tmp_path) -> None:
        out = tmp_path / "output.jsonl"
        save_generations(sample_results, out)
        assert out.exists()

    @pytest.mark.unit
    def test_correct_line_count(self, sample_results, tmp_path) -> None:
        out = tmp_path / "output.jsonl"
        save_generations(sample_results, out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == len(sample_results)

    @pytest.mark.unit
    def test_each_line_valid_json(self, sample_results, tmp_path) -> None:
        out = tmp_path / "output.jsonl"
        save_generations(sample_results, out)
        for line in out.read_text().strip().splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    @pytest.mark.unit
    def test_fields_round_trip(self, sample_results, tmp_path) -> None:
        out = tmp_path / "output.jsonl"
        save_generations(sample_results, out)
        loaded = [json.loads(l) for l in out.read_text().strip().splitlines()]
        assert loaded[0]["prompt"] == sample_results[0]["prompt"]
        assert loaded[1]["lam"] == sample_results[1]["lam"]

    @pytest.mark.unit
    def test_creates_parent_directories(self, sample_results, tmp_path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "output.jsonl"
        save_generations(sample_results, nested)
        assert nested.exists()

    @pytest.mark.unit
    def test_empty_results_creates_empty_file(self, tmp_path) -> None:
        out = tmp_path / "empty.jsonl"
        save_generations([], out)
        assert out.exists()
        assert out.read_text() == ""
