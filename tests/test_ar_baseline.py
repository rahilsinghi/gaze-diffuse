"""Tests for AR gaze guidance baseline (src/ar_baseline.py)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.ar_baseline import ARGazeConfig, ar_gaze_guided_generate, generate_ar_samples


# ---------------------------------------------------------------------------
# EOS-forcing mock LM — defined at module level so it is reusable
# ---------------------------------------------------------------------------


class MockOutput:
    """Mock LM output with .logits attribute (mirrors conftest version)."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class LocalMockCausalLM(nn.Module):
    """Mock autoregressive LM returning random logits — mirrors conftest.MockCausalLM."""

    def __init__(self, vocab_size: int = 30522) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Linear(1, 1)

    def forward(self, input_ids: torch.Tensor) -> MockOutput:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return MockOutput(logits=logits)


class EOSMockLM(nn.Module):
    """LM that always assigns the highest logit to the EOS token."""

    def __init__(self, eos_token_id: int, vocab_size: int = 30522) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.dummy = nn.Linear(1, 1)  # keeps parameters() non-empty

    def forward(self, input_ids: torch.Tensor) -> MockOutput:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        logits[:, :, self.eos_token_id] = 100.0
        return MockOutput(logits=logits)


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


# ---------------------------------------------------------------------------
# TestARGazeGuidedGenerate — tests for the inner generation loop
# ---------------------------------------------------------------------------


class TestARGazeGuidedGenerate:
    """Tests for ar_gaze_guided_generate (lines 45-118)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_mock_gaze_predictor(self, top_k: int, device: torch.device) -> MagicMock:
        """Return a mock GazePredictor whose score_vocabulary returns zeros."""
        mock_gaze = MagicMock()
        mock_gaze.score_vocabulary.return_value = torch.zeros(top_k, device=device)
        return mock_gaze

    def _make_mock_tokenizer(self, eos_token_id: int = 102) -> MagicMock:
        """Return a mock tokenizer with a fixed eos_token_id."""
        tok = MagicMock()
        tok.eos_token_id = eos_token_id
        tok.decode.side_effect = lambda token_ids, **_kwargs: f"tok{token_ids[0]}"
        return tok

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_output_starts_with_prompt(self, bert_tokenizer) -> None:
        """Generated sequence must begin with the exact prompt token IDs."""
        config = ARGazeConfig(top_k=10, max_new_tokens=8, lam=0.0)
        lm = LocalMockCausalLM(vocab_size=30522)
        prompt_ids = torch.tensor([1996, 5765, 1997, 1037, 2047], dtype=torch.long)
        mock_gaze = self._make_mock_gaze_predictor(config.top_k, prompt_ids.device)
        mock_tokenizer = self._make_mock_tokenizer()

        output = ar_gaze_guided_generate(lm, mock_gaze, mock_tokenizer, prompt_ids, config)

        assert output.shape[0] >= len(prompt_ids)
        assert torch.equal(output[: len(prompt_ids)], prompt_ids)

    @pytest.mark.unit
    def test_output_length_bounded(self, bert_tokenizer) -> None:
        """Output length must not exceed prompt_len + max_new_tokens."""
        max_new = 8
        config = ARGazeConfig(top_k=10, max_new_tokens=max_new, lam=0.0)
        lm = LocalMockCausalLM(vocab_size=30522)
        prompt_ids = torch.tensor([101, 2054, 2003, 102], dtype=torch.long)
        mock_tokenizer = self._make_mock_tokenizer(eos_token_id=9999)  # never hits EOS
        mock_gaze = self._make_mock_gaze_predictor(config.top_k, prompt_ids.device)

        output = ar_gaze_guided_generate(lm, mock_gaze, mock_tokenizer, prompt_ids, config)

        assert output.shape[0] <= len(prompt_ids) + max_new

    @pytest.mark.unit
    def test_unguided_lam_zero_runs(self) -> None:
        """With lam=0, score_vocabulary is never called and generation completes."""
        config = ARGazeConfig(top_k=10, max_new_tokens=5, lam=0.0)
        lm = LocalMockCausalLM(vocab_size=30522)
        prompt_ids = torch.tensor([101, 2054], dtype=torch.long)
        mock_gaze = MagicMock()
        mock_tokenizer = self._make_mock_tokenizer(eos_token_id=9999)

        output = ar_gaze_guided_generate(lm, mock_gaze, mock_tokenizer, prompt_ids, config)

        # score_vocabulary must NOT be called when lam == 0
        mock_gaze.score_vocabulary.assert_not_called()
        assert output.shape[0] > len(prompt_ids)

    @pytest.mark.unit
    def test_guided_positive_lam_calls_gaze_predictor(self) -> None:
        """With lam != 0, score_vocabulary is called once per generation step."""
        max_new = 4
        config = ARGazeConfig(top_k=10, max_new_tokens=max_new, lam=1.0)
        lm = LocalMockCausalLM(vocab_size=30522)
        prompt_ids = torch.tensor([101, 2054], dtype=torch.long)
        mock_gaze = self._make_mock_gaze_predictor(config.top_k, prompt_ids.device)
        mock_tokenizer = self._make_mock_tokenizer(eos_token_id=9999)

        output = ar_gaze_guided_generate(lm, mock_gaze, mock_tokenizer, prompt_ids, config)

        steps_taken = output.shape[0] - len(prompt_ids)
        assert mock_gaze.score_vocabulary.call_count == steps_taken
        assert steps_taken == max_new

    @pytest.mark.unit
    def test_guided_negative_lam_calls_gaze_predictor(self) -> None:
        """With negative lam (readability mode), score_vocabulary is still called."""
        max_new = 3
        config = ARGazeConfig(top_k=10, max_new_tokens=max_new, lam=-1.0)
        lm = LocalMockCausalLM(vocab_size=30522)
        prompt_ids = torch.tensor([101, 2054], dtype=torch.long)
        mock_gaze = self._make_mock_gaze_predictor(config.top_k, prompt_ids.device)
        mock_tokenizer = self._make_mock_tokenizer(eos_token_id=9999)

        output = ar_gaze_guided_generate(lm, mock_gaze, mock_tokenizer, prompt_ids, config)

        assert mock_gaze.score_vocabulary.call_count == max_new

    @pytest.mark.unit
    def test_eos_stops_generation(self) -> None:
        """When the LM always predicts EOS, generation stops after the first new token."""
        eos_id = 102  # typical BERT [SEP] / GPT-2 eos_token_id equivalent
        lm = EOSMockLM(eos_token_id=eos_id, vocab_size=30522)
        config = ARGazeConfig(top_k=10, max_new_tokens=50, lam=0.0)
        prompt_ids = torch.tensor([101, 2054, 2003], dtype=torch.long)
        mock_gaze = MagicMock()
        mock_tokenizer = self._make_mock_tokenizer(eos_token_id=eos_id)

        output = ar_gaze_guided_generate(lm, mock_gaze, mock_tokenizer, prompt_ids, config)

        # EOS must be in top-k (it has logit 100.0), so generation stops at step 1
        assert output.shape[0] == len(prompt_ids) + 1
        assert output[-1].item() == eos_id

    @pytest.mark.unit
    def test_zero_max_new_tokens_returns_prompt(self) -> None:
        """With max_new_tokens=0, the output is identical to the prompt."""
        config = ARGazeConfig(top_k=10, max_new_tokens=0, lam=0.0)
        lm = LocalMockCausalLM(vocab_size=30522)
        prompt_ids = torch.tensor([101, 2054, 2003], dtype=torch.long)
        mock_gaze = MagicMock()
        mock_tokenizer = self._make_mock_tokenizer()

        output = ar_gaze_guided_generate(lm, mock_gaze, mock_tokenizer, prompt_ids, config)

        assert torch.equal(output, prompt_ids)


# ---------------------------------------------------------------------------
# TestGenerateArSamplesAdditional — covers remaining generate_ar_samples paths
# ---------------------------------------------------------------------------


class TestGenerateArSamplesAdditional:
    """Additional tests for generate_ar_samples (lines 121-165)."""

    def _make_mock_lm(self) -> MagicMock:
        mock_lm = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_lm.parameters.return_value = iter([mock_param])
        return mock_lm

    def _make_mock_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.encode.return_value = torch.tensor([101, 2054, 2003])
        tok.decode.return_value = "generated text"
        return tok

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_result_contains_method_field(self, mock_generate: MagicMock) -> None:
        """Each result dict must have method == 'ar_gaze'."""
        mock_generate.return_value = torch.tensor([101, 2054, 2003, 100])
        config = ARGazeConfig(top_k=10, max_new_tokens=8)
        results = generate_ar_samples(
            lm=self._make_mock_lm(),
            gaze_predictor=MagicMock(),
            tokenizer=self._make_mock_tokenizer(),
            prompts=["Hello world"],
            config=config,
            n_samples_per_prompt=1,
        )
        assert len(results) == 1
        assert results[0]["method"] == "ar_gaze"

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_result_contains_model_field(self, mock_generate: MagicMock) -> None:
        """Each result dict must have a model key matching config.model_name."""
        mock_generate.return_value = torch.tensor([101, 2054, 2003, 100])
        config = ARGazeConfig(model_name="gpt2", top_k=10, max_new_tokens=8)
        results = generate_ar_samples(
            lm=self._make_mock_lm(),
            gaze_predictor=MagicMock(),
            tokenizer=self._make_mock_tokenizer(),
            prompts=["Hello world"],
            config=config,
            n_samples_per_prompt=1,
        )
        assert results[0]["model"] == "gpt2"

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_single_sample_per_prompt(self, mock_generate: MagicMock) -> None:
        """n_samples_per_prompt=1 produces exactly len(prompts) results."""
        mock_generate.return_value = torch.tensor([101, 2054, 2003, 100])
        prompts = ["First", "Second", "Third", "Fourth", "Fifth"]
        config = ARGazeConfig(top_k=10, max_new_tokens=8)
        results = generate_ar_samples(
            lm=self._make_mock_lm(),
            gaze_predictor=MagicMock(),
            tokenizer=self._make_mock_tokenizer(),
            prompts=prompts,
            config=config,
            n_samples_per_prompt=1,
        )
        assert len(results) == len(prompts)
        for i, result in enumerate(results):
            assert result["prompt_idx"] == i
            assert result["sample_idx"] == 0

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_empty_prompts_returns_empty_list(self, mock_generate: MagicMock) -> None:
        """Passing an empty prompts list returns an empty results list."""
        config = ARGazeConfig(top_k=10, max_new_tokens=8)
        results = generate_ar_samples(
            lm=self._make_mock_lm(),
            gaze_predictor=MagicMock(),
            tokenizer=self._make_mock_tokenizer(),
            prompts=[],
            config=config,
            n_samples_per_prompt=3,
        )
        assert results == []
        mock_generate.assert_not_called()

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_logging_triggered_at_10_prompt_intervals(
        self, mock_generate: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """logger.info is called after every 10th prompt (line 163)."""
        mock_generate.return_value = torch.tensor([101, 2054, 2003, 100])
        # 10 prompts → log fires once (after prompt_idx 9)
        prompts = [f"Prompt {i}" for i in range(10)]
        config = ARGazeConfig(top_k=10, max_new_tokens=8)
        with caplog.at_level(logging.INFO, logger="src.ar_baseline"):
            generate_ar_samples(
                lm=self._make_mock_lm(),
                gaze_predictor=MagicMock(),
                tokenizer=self._make_mock_tokenizer(),
                prompts=prompts,
                config=config,
                n_samples_per_prompt=1,
            )
        log_messages = [r.message for r in caplog.records if "AR baseline" in r.message]
        assert len(log_messages) == 1
        assert "10/10" in log_messages[0]

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_logging_not_triggered_for_fewer_than_10_prompts(
        self, mock_generate: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """logger.info is NOT called when fewer than 10 prompts are processed."""
        mock_generate.return_value = torch.tensor([101, 2054, 2003, 100])
        prompts = [f"Prompt {i}" for i in range(5)]
        config = ARGazeConfig(top_k=10, max_new_tokens=8)
        with caplog.at_level(logging.INFO, logger="src.ar_baseline"):
            generate_ar_samples(
                lm=self._make_mock_lm(),
                gaze_predictor=MagicMock(),
                tokenizer=self._make_mock_tokenizer(),
                prompts=prompts,
                config=config,
                n_samples_per_prompt=1,
            )
        log_messages = [r.message for r in caplog.records if "AR baseline" in r.message]
        assert len(log_messages) == 0

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_full_text_is_prompt_plus_generation(self, mock_generate: MagicMock) -> None:
        """full_text field must be the concatenation of prompt + ' ' + generation."""
        mock_generate.return_value = torch.tensor([101, 2054, 2003, 100])
        tok = MagicMock()
        tok.encode.return_value = torch.tensor([101, 2054])
        tok.decode.return_value = "hello"
        config = ARGazeConfig(top_k=10, max_new_tokens=8)
        results = generate_ar_samples(
            lm=self._make_mock_lm(),
            gaze_predictor=MagicMock(),
            tokenizer=tok,
            prompts=["My prompt"],
            config=config,
            n_samples_per_prompt=1,
        )
        assert results[0]["full_text"] == "My prompt hello"

    @pytest.mark.unit
    @patch("src.ar_baseline.ar_gaze_guided_generate")
    def test_lam_value_stored_in_result(self, mock_generate: MagicMock) -> None:
        """lam value from config must be stored verbatim in each result."""
        mock_generate.return_value = torch.tensor([101, 100])
        config = ARGazeConfig(top_k=10, max_new_tokens=8, lam=2.5)
        results = generate_ar_samples(
            lm=self._make_mock_lm(),
            gaze_predictor=MagicMock(),
            tokenizer=self._make_mock_tokenizer(),
            prompts=["Test"],
            config=config,
            n_samples_per_prompt=1,
        )
        assert results[0]["lam"] == 2.5
