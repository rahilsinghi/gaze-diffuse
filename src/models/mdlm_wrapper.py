"""Thin wrapper around MDLM for gaze-guided sampling.

Adapts the MDLM codebase API (diffusion.Diffusion) into our GazeDiffuse
sampling loop. Handles checkpoint loading, noise schedule access, and
the forward pass interface.

Key MDLM API notes:
- model.forward(x, sigma) returns LOG PROBS (not raw logits), shape (B, L, V)
- model.mask_index is the mask token ID
- model.noise(t) returns (sigma_t, sigma_rate_t) where t is (B, 1) in [0, 1]
- Time goes from 1.0 (fully masked) to eps (~1e-5, fully denoised)
- Parameterization: 'subs' (substitution-based, default)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MDLMConfig:
    """Configuration for loading MDLM."""

    checkpoint_path: str = "checkpoints/mdlm-owt"
    mdlm_repo_path: str = "submodules/mdlm"
    parameterization: str = "subs"
    backbone: str = "dit"
    model_length: int = 1024
    sampling_steps: int = 128
    sampling_predictor: str = "ddpm_cache"
    noise_removal: bool = True


class MDLMWrapper:
    """Wrapper around MDLM's Diffusion model for gaze-guided sampling.

    Provides a clean interface for:
    1. Loading pretrained MDLM from checkpoint
    2. Getting log probs at each denoising step
    3. Managing the noise schedule
    4. Accessing the mask token ID and tokenizer
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        mask_index: int,
        noise_fn,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mask_index = mask_index
        self.noise_fn = noise_fn
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        config: MDLMConfig,
        device: torch.device | None = None,
    ) -> "MDLMWrapper":
        """Load MDLM from a checkpoint.

        Adds the MDLM submodule to sys.path and loads using their API.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mdlm_path = Path(config.mdlm_repo_path).resolve()
        if str(mdlm_path) not in sys.path:
            sys.path.insert(0, str(mdlm_path))

        try:
            from diffusion import Diffusion
            from dataloader import get_tokenizer
        except ImportError as e:
            raise ImportError(
                f"Cannot import MDLM modules. Ensure {mdlm_path} exists "
                f"and contains diffusion.py. Error: {e}"
            ) from e

        # Build Hydra-style config for MDLM
        mdlm_cfg = OmegaConf.create({
            "backbone": config.backbone,
            "parameterization": config.parameterization,
            "time_conditioning": False,
            "model": {
                "hidden_size": 768,
                "cond_dim": 128,
                "length": config.model_length,
                "n_blocks": 12,
                "n_heads": 12,
                "dropout": 0.0,
            },
            "sampling": {
                "predictor": config.sampling_predictor,
                "steps": config.sampling_steps,
                "noise_removal": config.noise_removal,
                "semi_ar": False,
            },
            "noise": {
                "type": "loglinear",
                "sigma_min": 1e-3,
                "sigma_max": 1.0,
            },
            "data": {
                "tokenizer_name_or_path": "bert-base-uncased",
            },
            "eval": {
                "checkpoint_path": config.checkpoint_path,
            },
            "loader": {
                "eval_batch_size": 1,
            },
        })

        tokenizer = get_tokenizer(mdlm_cfg)
        model = Diffusion.load_from_checkpoint(
            config.checkpoint_path,
            tokenizer=tokenizer,
            config=mdlm_cfg,
        )
        model = model.to(device).eval()

        logger.info(
            "Loaded MDLM from %s (mask_index=%d, vocab_size=%d)",
            config.checkpoint_path,
            model.mask_index,
            model.vocab_size,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            mask_index=model.mask_index,
            noise_fn=model.noise,
            device=device,
        )

    @torch.no_grad()
    def get_log_probs(
        self,
        x: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """Get log probabilities for all positions at noise level t.

        Args:
            x: [batch, seq_len] token IDs (with mask tokens)
            t: Noise level in [0, 1]. 1.0 = fully masked, ~0 = clean.

        Returns:
            [batch, seq_len, vocab_size] log probabilities
        """
        batch_size = x.shape[0]
        t_tensor = torch.full(
            (batch_size, 1), t, device=self.device, dtype=torch.float32
        )
        sigma_t, _ = self.noise_fn(t_tensor)

        # MDLM forward returns log probs directly
        log_probs = self.model(x, sigma_t)
        return log_probs

    @torch.no_grad()
    def get_timesteps(
        self,
        num_steps: int,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Get the denoising timestep schedule.

        Returns timesteps from 1.0 (fully masked) to eps (clean).
        """
        return torch.linspace(1.0, eps, num_steps + 1, device=self.device)

    def create_masked_input(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int,
    ) -> torch.Tensor:
        """Create initial input: prompt + all [MASK] tokens.

        Args:
            prompt_ids: [prompt_len] prompt token IDs
            gen_length: Number of tokens to generate

        Returns:
            [prompt_len + gen_length] with masks at generation positions
        """
        masks = torch.full(
            (gen_length,),
            self.mask_index,
            dtype=torch.long,
            device=self.device,
        )
        return torch.cat([prompt_ids.to(self.device), masks])

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class LLaDAWrapper:
    """Wrapper around LLaDA for gaze-guided sampling.

    LLaDA uses mask_id=126336 and a different sampling loop than MDLM.
    """

    MASK_ID = 126336

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mask_index = self.MASK_ID

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: torch.device | None = None,
        load_in_4bit: bool = False,
    ) -> "LLaDAWrapper":
        """Load LLaDA from HuggingFace.

        Args:
            model_name: HuggingFace model ID
            device: Target device
            load_in_4bit: Use 4-bit quantization (for 2x RTX8000)
        """
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["device_map"] = device

        model = AutoModel.from_pretrained(model_name, **kwargs)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        tokenizer.padding_side = "left"

        logger.info("Loaded LLaDA from %s", model_name)

        return cls(model=model, tokenizer=tokenizer, device=device)

    @torch.no_grad()
    def get_logits(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get raw logits for all positions.

        Args:
            x: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] optional

        Returns:
            [batch, seq_len, vocab_size] logits
        """
        outputs = self.model(x, attention_mask=attention_mask)
        return outputs.logits

    def create_masked_input(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int,
    ) -> torch.Tensor:
        """Create initial input: prompt + all [MASK] tokens."""
        masks = torch.full(
            (gen_length,),
            self.MASK_ID,
            dtype=torch.long,
            device=self.device,
        )
        return torch.cat([prompt_ids.to(self.device), masks])

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
