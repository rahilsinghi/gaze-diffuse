"""Tests for src/models/mdlm_wrapper.py.

Covers MDLMConfig and LLaDAWrapper configuration, direct construction
with mock models, and all wrapper methods. No real checkpoints or GPUs
needed — everything runs on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.models.mdlm_wrapper import MDLMConfig, MDLMWrapper, LLaDAWrapper


# ---------------------------------------------------------------------------
# Local mock helpers
# ---------------------------------------------------------------------------


class MockDiffusionModel(nn.Module):
    """Mimics MDLM's Diffusion.forward(x, sigma) → log_probs."""

    def __init__(self, vocab_size: int = 30522) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        batch, seq = x.shape
        # Return random log-probs (MDLM contract: log-prob tensor, not logits)
        return torch.randn(batch, seq, self.vocab_size)


class MockLLaDAModel(nn.Module):
    """Mimics LLaDA forward(x, attention_mask=...) → object with .logits."""

    def __init__(self, vocab_size: int = 32000) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Linear(1, 1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> "MockOutput":
        batch, seq = x.shape
        logits = torch.randn(batch, seq, self.vocab_size)
        return MockOutput(logits=logits)


class MockOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


def mock_noise_fn(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple noise schedule: sigma = t, sigma_rate = ones."""
    return t, torch.ones_like(t)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mdlm_wrapper(bert_tokenizer) -> MDLMWrapper:
    """MDLMWrapper built with a mock diffusion model."""
    return MDLMWrapper(
        model=MockDiffusionModel(),
        tokenizer=bert_tokenizer,
        mask_index=103,  # BERT [MASK]
        noise_fn=mock_noise_fn,
        device=torch.device("cpu"),
    )


@pytest.fixture
def llada_wrapper(bert_tokenizer) -> LLaDAWrapper:
    """LLaDAWrapper built with a mock LLaDA model."""
    return LLaDAWrapper(
        model=MockLLaDAModel(),
        tokenizer=bert_tokenizer,
        device=torch.device("cpu"),
    )


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    """Short prompt token sequence."""
    return torch.tensor([101, 1996, 5765, 102], dtype=torch.long)  # [CLS] the discovery [SEP]


# ---------------------------------------------------------------------------
# MDLMConfig
# ---------------------------------------------------------------------------


class TestMDLMConfig:
    @pytest.mark.unit
    def test_default_values(self) -> None:
        config = MDLMConfig()
        assert config.checkpoint_path == "checkpoints/mdlm-owt"
        assert config.mdlm_repo_path == "submodules/mdlm"
        assert config.parameterization == "subs"
        assert config.backbone == "dit"
        assert config.model_length == 1024
        assert config.sampling_steps == 128
        assert config.sampling_predictor == "ddpm_cache"
        assert config.noise_removal is True

    @pytest.mark.unit
    def test_frozen(self) -> None:
        config = MDLMConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.checkpoint_path = "other/path"  # type: ignore[misc]

    @pytest.mark.unit
    def test_custom_values(self) -> None:
        config = MDLMConfig(
            checkpoint_path="my/ckpt",
            sampling_steps=64,
            model_length=512,
            noise_removal=False,
        )
        assert config.checkpoint_path == "my/ckpt"
        assert config.sampling_steps == 64
        assert config.model_length == 512
        assert config.noise_removal is False

    @pytest.mark.unit
    def test_partial_override_preserves_defaults(self) -> None:
        """Only specified fields change; all others remain default."""
        config = MDLMConfig(backbone="transformer")
        assert config.backbone == "transformer"
        assert config.parameterization == "subs"  # default untouched
        assert config.sampling_steps == 128


# ---------------------------------------------------------------------------
# MDLMWrapper — get_log_probs
# ---------------------------------------------------------------------------


class TestMDLMWrapperGetLogProbs:
    @pytest.mark.unit
    def test_output_shape_single_item(self, mdlm_wrapper: MDLMWrapper) -> None:
        """[1, seq, vocab] returned for a single batch item."""
        seq_len = 20
        x = torch.randint(0, 1000, (1, seq_len))
        log_probs = mdlm_wrapper.get_log_probs(x, t=0.5)
        assert log_probs.shape == (1, seq_len, MockDiffusionModel().vocab_size)

    @pytest.mark.unit
    def test_output_shape_batch(self, mdlm_wrapper: MDLMWrapper) -> None:
        """[batch, seq, vocab] shape for batch > 1."""
        batch, seq_len = 3, 15
        x = torch.randint(0, 1000, (batch, seq_len))
        log_probs = mdlm_wrapper.get_log_probs(x, t=0.3)
        assert log_probs.shape == (batch, seq_len, MockDiffusionModel().vocab_size)

    @pytest.mark.unit
    def test_noise_fn_called_with_correct_shape(self) -> None:
        """noise_fn receives a (batch, 1) tensor at the given t value."""
        captured: list[torch.Tensor] = []

        def recording_noise_fn(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            captured.append(t.clone())
            return mock_noise_fn(t)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        wrapper = MDLMWrapper(
            model=MockDiffusionModel(),
            tokenizer=tokenizer,
            mask_index=103,
            noise_fn=recording_noise_fn,
            device=torch.device("cpu"),
        )

        batch = 4
        x = torch.randint(0, 100, (batch, 10))
        wrapper.get_log_probs(x, t=0.7)

        assert len(captured) == 1
        t_arg = captured[0]
        assert t_arg.shape == (batch, 1)
        assert torch.allclose(t_arg, torch.full((batch, 1), 0.7))

    @pytest.mark.unit
    def test_no_grad_applied(self, mdlm_wrapper: MDLMWrapper) -> None:
        """get_log_probs is decorated with @torch.no_grad; result has no grad."""
        x = torch.randint(0, 100, (1, 8))
        log_probs = mdlm_wrapper.get_log_probs(x, t=0.5)
        assert not log_probs.requires_grad

    @pytest.mark.unit
    def test_t_equals_one_fully_masked(self, mdlm_wrapper: MDLMWrapper) -> None:
        """t=1.0 (fully masked) still returns correct shape."""
        x = torch.full((1, 12), 103, dtype=torch.long)  # all [MASK]
        log_probs = mdlm_wrapper.get_log_probs(x, t=1.0)
        assert log_probs.shape[0] == 1
        assert log_probs.shape[1] == 12

    @pytest.mark.unit
    def test_t_near_zero_clean(self, mdlm_wrapper: MDLMWrapper) -> None:
        """t≈0 (nearly clean) still returns correct shape."""
        x = torch.randint(0, 1000, (1, 8))
        log_probs = mdlm_wrapper.get_log_probs(x, t=1e-5)
        assert log_probs.shape[0] == 1


# ---------------------------------------------------------------------------
# MDLMWrapper — get_timesteps
# ---------------------------------------------------------------------------


class TestMDLMWrapperGetTimesteps:
    @pytest.mark.unit
    def test_length(self, mdlm_wrapper: MDLMWrapper) -> None:
        """Returns num_steps + 1 timesteps."""
        ts = mdlm_wrapper.get_timesteps(num_steps=10)
        assert ts.shape == (11,)

    @pytest.mark.unit
    def test_starts_at_one(self, mdlm_wrapper: MDLMWrapper) -> None:
        """First timestep must be 1.0 (fully masked)."""
        ts = mdlm_wrapper.get_timesteps(num_steps=5)
        assert ts[0].item() == pytest.approx(1.0)

    @pytest.mark.unit
    def test_ends_at_eps(self, mdlm_wrapper: MDLMWrapper) -> None:
        """Last timestep must be eps (default 1e-5)."""
        ts = mdlm_wrapper.get_timesteps(num_steps=5)
        assert ts[-1].item() == pytest.approx(1e-5)

    @pytest.mark.unit
    def test_monotonically_decreasing(self, mdlm_wrapper: MDLMWrapper) -> None:
        """Timesteps should strictly decrease from 1.0 to eps."""
        ts = mdlm_wrapper.get_timesteps(num_steps=20)
        diffs = ts[:-1] - ts[1:]
        assert (diffs > 0).all()

    @pytest.mark.unit
    def test_custom_eps(self, mdlm_wrapper: MDLMWrapper) -> None:
        """Custom eps is respected as the final value."""
        ts = mdlm_wrapper.get_timesteps(num_steps=8, eps=1e-3)
        assert ts[-1].item() == pytest.approx(1e-3)

    @pytest.mark.unit
    def test_single_step(self, mdlm_wrapper: MDLMWrapper) -> None:
        """num_steps=1 returns exactly 2 values: [1.0, eps]."""
        ts = mdlm_wrapper.get_timesteps(num_steps=1)
        assert ts.shape == (2,)
        assert ts[0].item() == pytest.approx(1.0)
        assert ts[-1].item() == pytest.approx(1e-5)

    @pytest.mark.unit
    def test_on_correct_device(self, mdlm_wrapper: MDLMWrapper) -> None:
        """Timesteps tensor lives on the wrapper's device (cpu here)."""
        ts = mdlm_wrapper.get_timesteps(num_steps=4)
        assert ts.device.type == "cpu"


# ---------------------------------------------------------------------------
# MDLMWrapper — create_masked_input
# ---------------------------------------------------------------------------


class TestMDLMWrapperCreateMaskedInput:
    @pytest.mark.unit
    def test_total_length(self, mdlm_wrapper: MDLMWrapper, prompt_ids: torch.Tensor) -> None:
        """Output length = prompt_len + gen_length."""
        gen_length = 10
        result = mdlm_wrapper.create_masked_input(prompt_ids, gen_length)
        assert result.shape == (len(prompt_ids) + gen_length,)

    @pytest.mark.unit
    def test_prompt_preserved(self, mdlm_wrapper: MDLMWrapper, prompt_ids: torch.Tensor) -> None:
        """The leading prompt tokens are unchanged."""
        result = mdlm_wrapper.create_masked_input(prompt_ids, gen_length=5)
        assert torch.equal(result[: len(prompt_ids)], prompt_ids)

    @pytest.mark.unit
    def test_generation_slots_are_mask_id(self, mdlm_wrapper: MDLMWrapper, prompt_ids: torch.Tensor) -> None:
        """Every generation slot must equal mask_index (103)."""
        gen_length = 8
        result = mdlm_wrapper.create_masked_input(prompt_ids, gen_length)
        generation_part = result[len(prompt_ids):]
        assert (generation_part == mdlm_wrapper.mask_index).all()

    @pytest.mark.unit
    def test_uses_custom_mask_index(self, bert_tokenizer) -> None:
        """mask_index from constructor is used, not hard-coded 103."""
        custom_mask = 999
        wrapper = MDLMWrapper(
            model=MockDiffusionModel(),
            tokenizer=bert_tokenizer,
            mask_index=custom_mask,
            noise_fn=mock_noise_fn,
            device=torch.device("cpu"),
        )
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        result = wrapper.create_masked_input(prompt, gen_length=4)
        assert (result[3:] == custom_mask).all()

    @pytest.mark.unit
    def test_zero_gen_length(self, mdlm_wrapper: MDLMWrapper, prompt_ids: torch.Tensor) -> None:
        """gen_length=0 returns just the prompt, no masks appended."""
        result = mdlm_wrapper.create_masked_input(prompt_ids, gen_length=0)
        assert torch.equal(result, prompt_ids)

    @pytest.mark.unit
    def test_result_on_cpu(self, mdlm_wrapper: MDLMWrapper, prompt_ids: torch.Tensor) -> None:
        """Output tensor is on the wrapper's device (cpu)."""
        result = mdlm_wrapper.create_masked_input(prompt_ids, gen_length=5)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_dtype_is_long(self, mdlm_wrapper: MDLMWrapper, prompt_ids: torch.Tensor) -> None:
        result = mdlm_wrapper.create_masked_input(prompt_ids, gen_length=4)
        assert result.dtype == torch.long


# ---------------------------------------------------------------------------
# MDLMWrapper — decode
# ---------------------------------------------------------------------------


class TestMDLMWrapperDecode:
    @pytest.mark.unit
    def test_decode_calls_tokenizer(self, bert_tokenizer) -> None:
        """decode() delegates to tokenizer.decode with skip_special_tokens=True."""
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello world"

        wrapper = MDLMWrapper(
            model=MockDiffusionModel(),
            tokenizer=mock_tok,
            mask_index=103,
            noise_fn=mock_noise_fn,
            device=torch.device("cpu"),
        )

        ids = torch.tensor([7592, 2088], dtype=torch.long)
        result = wrapper.decode(ids)

        mock_tok.decode.assert_called_once_with(ids, skip_special_tokens=True)
        assert result == "hello world"

    @pytest.mark.unit
    def test_decode_returns_string(self, mdlm_wrapper: MDLMWrapper) -> None:
        """decode() always returns a str (real tokenizer round-trip)."""
        ids = torch.tensor([101, 7592, 2088, 102], dtype=torch.long)
        result = mdlm_wrapper.decode(ids)
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_decode_skips_special_tokens(self, mdlm_wrapper: MDLMWrapper, bert_tokenizer) -> None:
        """[CLS] and [SEP] (101, 102) should not appear in decoded text."""
        ids = torch.tensor([101, 7592, 102], dtype=torch.long)  # [CLS] hello [SEP]
        result = mdlm_wrapper.decode(ids)
        assert "[CLS]" not in result
        assert "[SEP]" not in result


# ---------------------------------------------------------------------------
# LLaDAWrapper — class constant
# ---------------------------------------------------------------------------


class TestLLaDAMaskId:
    @pytest.mark.unit
    def test_mask_id_constant(self) -> None:
        """MASK_ID must be exactly 126336 per the LLaDA specification."""
        assert LLaDAWrapper.MASK_ID == 126336

    @pytest.mark.unit
    def test_mask_index_attribute_matches_constant(self, llada_wrapper: LLaDAWrapper) -> None:
        """Instance attribute mask_index must equal the class constant."""
        assert llada_wrapper.mask_index == LLaDAWrapper.MASK_ID


# ---------------------------------------------------------------------------
# LLaDAWrapper — get_logits
# ---------------------------------------------------------------------------


class TestLLaDAWrapperGetLogits:
    @pytest.mark.unit
    def test_output_shape_single_item(self, llada_wrapper: LLaDAWrapper) -> None:
        """[1, seq, vocab] for batch=1."""
        vocab_size = MockLLaDAModel().vocab_size
        seq_len = 12
        x = torch.randint(0, 1000, (1, seq_len))
        logits = llada_wrapper.get_logits(x)
        assert logits.shape == (1, seq_len, vocab_size)

    @pytest.mark.unit
    def test_output_shape_batch(self, llada_wrapper: LLaDAWrapper) -> None:
        """[batch, seq, vocab] for batch > 1."""
        vocab_size = MockLLaDAModel().vocab_size
        batch, seq_len = 4, 20
        x = torch.randint(0, 1000, (batch, seq_len))
        logits = llada_wrapper.get_logits(x)
        assert logits.shape == (batch, seq_len, vocab_size)

    @pytest.mark.unit
    def test_attention_mask_passed_through(self) -> None:
        """attention_mask kwarg is forwarded to the underlying model."""
        captured: dict = {}

        class TracingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = 32000
                self.dummy = nn.Linear(1, 1)

            def forward(self, x: torch.Tensor, attention_mask=None) -> MockOutput:
                captured["mask"] = attention_mask
                return MockOutput(logits=torch.randn(x.shape[0], x.shape[1], self.vocab_size))

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        wrapper = LLaDAWrapper(
            model=TracingModel(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )

        x = torch.randint(0, 100, (2, 8))
        attn = torch.ones(2, 8, dtype=torch.long)
        wrapper.get_logits(x, attention_mask=attn)

        assert captured["mask"] is not None
        assert torch.equal(captured["mask"], attn)

    @pytest.mark.unit
    def test_no_attention_mask_passes_none(self) -> None:
        """Omitting attention_mask forwards None to the model."""
        captured: dict = {}

        class TracingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = 32000
                self.dummy = nn.Linear(1, 1)

            def forward(self, x: torch.Tensor, attention_mask=None) -> MockOutput:
                captured["mask"] = attention_mask
                return MockOutput(logits=torch.randn(x.shape[0], x.shape[1], self.vocab_size))

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        wrapper = LLaDAWrapper(
            model=TracingModel(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )

        x = torch.randint(0, 100, (1, 5))
        wrapper.get_logits(x)
        assert captured["mask"] is None

    @pytest.mark.unit
    def test_no_grad_applied(self, llada_wrapper: LLaDAWrapper) -> None:
        """get_logits is decorated with @torch.no_grad; output has no grad."""
        x = torch.randint(0, 100, (1, 6))
        logits = llada_wrapper.get_logits(x)
        assert not logits.requires_grad


# ---------------------------------------------------------------------------
# LLaDAWrapper — create_masked_input
# ---------------------------------------------------------------------------


class TestLLaDAWrapperCreateMaskedInput:
    @pytest.mark.unit
    def test_total_length(self, llada_wrapper: LLaDAWrapper, prompt_ids: torch.Tensor) -> None:
        """Output length = prompt_len + gen_length."""
        gen_length = 16
        result = llada_wrapper.create_masked_input(prompt_ids, gen_length)
        assert result.shape == (len(prompt_ids) + gen_length,)

    @pytest.mark.unit
    def test_prompt_preserved(self, llada_wrapper: LLaDAWrapper, prompt_ids: torch.Tensor) -> None:
        """Leading prompt tokens are unchanged."""
        result = llada_wrapper.create_masked_input(prompt_ids, gen_length=6)
        assert torch.equal(result[: len(prompt_ids)], prompt_ids)

    @pytest.mark.unit
    def test_generation_slots_are_mask_id(self, llada_wrapper: LLaDAWrapper, prompt_ids: torch.Tensor) -> None:
        """Every generation slot must equal LLaDAWrapper.MASK_ID (126336)."""
        gen_length = 10
        result = llada_wrapper.create_masked_input(prompt_ids, gen_length)
        generation_part = result[len(prompt_ids):]
        assert (generation_part == LLaDAWrapper.MASK_ID).all()

    @pytest.mark.unit
    def test_zero_gen_length(self, llada_wrapper: LLaDAWrapper, prompt_ids: torch.Tensor) -> None:
        """gen_length=0 returns just the prompt with no trailing masks."""
        result = llada_wrapper.create_masked_input(prompt_ids, gen_length=0)
        assert torch.equal(result, prompt_ids)

    @pytest.mark.unit
    def test_result_dtype_long(self, llada_wrapper: LLaDAWrapper, prompt_ids: torch.Tensor) -> None:
        result = llada_wrapper.create_masked_input(prompt_ids, gen_length=5)
        assert result.dtype == torch.long

    @pytest.mark.unit
    def test_result_on_cpu(self, llada_wrapper: LLaDAWrapper, prompt_ids: torch.Tensor) -> None:
        result = llada_wrapper.create_masked_input(prompt_ids, gen_length=5)
        assert result.device.type == "cpu"


# ---------------------------------------------------------------------------
# LLaDAWrapper — decode
# ---------------------------------------------------------------------------


class TestLLaDAWrapperDecode:
    @pytest.mark.unit
    def test_decode_calls_tokenizer(self) -> None:
        """decode() calls tokenizer.decode with skip_special_tokens=True."""
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "the cat sat"

        wrapper = LLaDAWrapper(
            model=MockLLaDAModel(),
            tokenizer=mock_tok,
            device=torch.device("cpu"),
        )

        ids = torch.tensor([101, 1996, 102], dtype=torch.long)
        result = wrapper.decode(ids)

        mock_tok.decode.assert_called_once_with(ids, skip_special_tokens=True)
        assert result == "the cat sat"

    @pytest.mark.unit
    def test_decode_returns_string(self, llada_wrapper: LLaDAWrapper) -> None:
        """decode() always returns str (real tokenizer round-trip)."""
        ids = torch.tensor([101, 7592, 2088, 102], dtype=torch.long)
        result = llada_wrapper.decode(ids)
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_decode_skips_special_tokens(self, llada_wrapper: LLaDAWrapper) -> None:
        """[CLS] and [SEP] are stripped from output."""
        ids = torch.tensor([101, 7592, 102], dtype=torch.long)
        result = llada_wrapper.decode(ids)
        assert "[CLS]" not in result
        assert "[SEP]" not in result
