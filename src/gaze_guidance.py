"""GazeDiffuse: Gaze-guided parallel denoising for MDLM (Experiment 4).

Core novel contribution. Applies gaze-based readability signal globally
across ALL unmasked positions simultaneously at each denoising step
of a masked diffusion language model.

Unlike AR gaze guidance (Sauberli et al.), which guides one token at a time,
GazeDiffuse leverages the parallel nature of diffusion to produce more
globally coherent readability changes.

Usage:
    python -m src.gaze_guidance \
        --mdlm_checkpoint checkpoints/mdlm-owt \
        --gaze_checkpoint checkpoints/gaze_predictor/gaze_predictor_best.pt \
        --lam -1.0 --steps 64 --n_samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.gaze_predictor import GazePredictor, load_trained_predictor
from src.models.mdlm_wrapper import MDLMConfig, MDLMWrapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GazeDiffuseConfig:
    """Configuration for GazeDiffuse sampling."""

    lam: float = -1.0  # <0 = easier, >0 = harder, 0 = unguided
    steps: int = 64  # Number of denoising (unmasking) steps
    top_k: int = 100  # Top-k candidates for scoring
    gen_length: int = 128  # Length of generated text (tokens)
    temperature: float = 1.0
    gaze_temperature: float = 1.0  # Temperature for gaze scores
    mask_token_id: int | None = None  # Set from tokenizer


def confidence_schedule(
    step: int,
    total_steps: int,
    n_masked: int,
) -> int:
    """Compute number of tokens to reveal at this step.

    Uses a linear schedule: reveals roughly equal tokens per step,
    with any remainder revealed in the last step.

    Args:
        step: Current denoising step (0-indexed)
        total_steps: Total number of denoising steps
        n_masked: Number of currently masked positions

    Returns:
        Number of tokens to reveal this step
    """
    remaining_steps = max(1, total_steps - step)
    n_reveal = max(1, n_masked // remaining_steps)
    return min(n_reveal, n_masked)


@torch.no_grad()
def gaze_guided_diffuse(
    mdlm: torch.nn.Module,
    gaze_predictor: GazePredictor,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    config: GazeDiffuseConfig,
) -> torch.Tensor:
    """Run gaze-guided denoising on MDLM.

    The core GazeDiffuse algorithm:
    1. Initialize: prompt tokens + [MASK] tokens for generation
    2. For each denoising step:
       a. Get MDLM logits for all positions (bidirectional)
       b. Get gaze predictor scores for all positions
       c. Combine: score = log P_LM(tok) + lambda * gaze(tok)
       d. For masked positions, select highest-confidence tokens
       e. Reveal top-confidence tokens this step
    3. Return fully unmasked sequence

    Args:
        mdlm: Pretrained MDLM model (frozen)
        gaze_predictor: Trained gaze predictor (frozen)
        tokenizer: Tokenizer (shared between MDLM and gaze predictor)
        prompt_ids: [prompt_len] token IDs for the prompt
        config: Sampling configuration

    Returns:
        [prompt_len + gen_length] fully denoised token IDs
    """
    device = prompt_ids.device
    mask_id = config.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.mask_token_id

    n_prompt = len(prompt_ids)
    gen_len = config.gen_length

    # Initialize: prompt + all [MASK]
    x = torch.cat([
        prompt_ids,
        torch.full((gen_len,), mask_id, dtype=torch.long, device=device),
    ])

    for step in range(config.steps):
        # Find masked positions
        masked_pos = (x == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            break  # All tokens revealed

        # --- MDLM forward: bidirectional logits over full sequence ---
        lm_logits = mdlm(x.unsqueeze(0)).logits[0]  # [L, V]
        lm_log_probs = F.log_softmax(
            lm_logits / config.temperature, dim=-1
        )  # [L, V]

        # --- Gaze scoring at masked positions ---
        if config.lam != 0.0:
            # For each masked position, score top-k candidates
            gaze_scores = torch.zeros_like(lm_log_probs)

            for pos in masked_pos:
                pos_idx = pos.item()
                top_k_ids = lm_log_probs[pos_idx].topk(config.top_k).indices

                # Score each candidate by substituting into sequence
                candidate_gaze = gaze_predictor.score_vocabulary(
                    sequence=x,
                    position=pos_idx,
                    vocab_candidates=top_k_ids,
                    tokenizer=tokenizer,
                )

                # Normalize gaze scores to [0, 1] range for stability
                if candidate_gaze.std() > 1e-6:
                    candidate_gaze = (candidate_gaze - candidate_gaze.mean()) / (
                        candidate_gaze.std() + 1e-8
                    )

                gaze_scores[pos_idx, top_k_ids] = (
                    candidate_gaze / config.gaze_temperature
                )

            # Combine LM + gaze signal
            guided_scores = lm_log_probs + config.lam * gaze_scores
        else:
            guided_scores = lm_log_probs

        # --- Select tokens at masked positions ---
        masked_scores = guided_scores[masked_pos]  # [n_masked, V]
        predicted_tokens = masked_scores.argmax(dim=-1)  # [n_masked]
        confidence = masked_scores.max(dim=-1).values  # [n_masked]

        # --- Reveal highest-confidence tokens this step ---
        n_reveal = confidence_schedule(step, config.steps, len(masked_pos))
        reveal_indices = confidence.topk(n_reveal).indices

        x[masked_pos[reveal_indices]] = predicted_tokens[reveal_indices]

    return x


@torch.no_grad()
def gaze_guided_diffuse_mdlm(
    mdlm_wrapper: MDLMWrapper,
    gaze_predictor: GazePredictor,
    prompt_ids: torch.Tensor,
    config: GazeDiffuseConfig,
) -> torch.Tensor:
    """Run gaze-guided denoising using the native MDLM API.

    This version uses MDLM's noise schedule and log-prob parameterization
    directly, matching the actual MDLM denoising process.

    Args:
        mdlm_wrapper: Loaded MDLMWrapper instance
        gaze_predictor: Trained gaze predictor (frozen)
        prompt_ids: [prompt_len] token IDs for the prompt
        config: Sampling configuration

    Returns:
        [prompt_len + gen_length] fully denoised token IDs
    """
    device = mdlm_wrapper.device
    mask_id = mdlm_wrapper.mask_index
    tokenizer = mdlm_wrapper.tokenizer

    # Initialize: prompt + all [MASK]
    x = mdlm_wrapper.create_masked_input(prompt_ids, config.gen_length)
    x = x.unsqueeze(0)  # [1, L]
    n_prompt = len(prompt_ids)

    # Get MDLM timesteps (1.0 → eps)
    timesteps = mdlm_wrapper.get_timesteps(config.steps)

    for step_idx in range(config.steps):
        t = timesteps[step_idx].item()

        # Find masked positions
        masked_pos = (x[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            break

        # MDLM forward: returns log probs directly
        log_probs = mdlm_wrapper.get_log_probs(x, t)  # [1, L, V]
        log_probs = log_probs[0]  # [L, V]

        # Apply temperature
        if config.temperature != 1.0:
            log_probs = log_probs / config.temperature

        # Gaze guidance at masked positions
        if config.lam != 0.0:
            gaze_scores = torch.zeros_like(log_probs)

            for pos in masked_pos:
                pos_idx = pos.item()
                top_k_ids = log_probs[pos_idx].topk(config.top_k).indices

                candidate_gaze = gaze_predictor.score_vocabulary(
                    sequence=x[0],
                    position=pos_idx,
                    vocab_candidates=top_k_ids,
                    tokenizer=tokenizer,
                )

                if candidate_gaze.std() > 1e-6:
                    candidate_gaze = (candidate_gaze - candidate_gaze.mean()) / (
                        candidate_gaze.std() + 1e-8
                    )

                gaze_scores[pos_idx, top_k_ids] = (
                    candidate_gaze / config.gaze_temperature
                )

            guided_scores = log_probs + config.lam * gaze_scores
        else:
            guided_scores = log_probs

        # Select and reveal tokens
        masked_scores = guided_scores[masked_pos]
        predicted_tokens = masked_scores.argmax(dim=-1)
        confidence = masked_scores.max(dim=-1).values

        n_reveal = confidence_schedule(step_idx, config.steps, len(masked_pos))
        reveal_indices = confidence.topk(n_reveal).indices

        x[0, masked_pos[reveal_indices]] = predicted_tokens[reveal_indices]

    return x[0]


def generate_samples(
    mdlm: torch.nn.Module,
    gaze_predictor: GazePredictor,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    config: GazeDiffuseConfig,
    n_samples_per_prompt: int = 4,
) -> list[dict[str, str | float]]:
    """Generate multiple samples for each prompt.

    Returns list of dicts with keys: prompt, generation, lam, steps.
    """
    device = next(mdlm.parameters()).device
    results: list[dict[str, str | float]] = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        prompt_ids = prompt_ids.to(device)

        for sample_idx in range(n_samples_per_prompt):
            output_ids = gaze_guided_diffuse(
                mdlm=mdlm,
                gaze_predictor=gaze_predictor,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                config=config,
            )

            generated_text = tokenizer.decode(
                output_ids[len(prompt_ids) :],
                skip_special_tokens=True,
            )

            results.append({
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "prompt": prompt,
                "generation": generated_text,
                "full_text": prompt + " " + generated_text,
                "lam": config.lam,
                "steps": config.steps,
            })

            if (sample_idx + 1) % n_samples_per_prompt == 0:
                logger.info(
                    "Prompt %d/%d complete (%d samples)",
                    prompt_idx + 1,
                    len(prompts),
                    n_samples_per_prompt,
                )

    return results


def save_generations(
    results: list[dict],
    output_path: str | Path,
) -> None:
    """Save generation results to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info("Saved %d generations to %s", len(results), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="GazeDiffuse: gaze-guided MDLM sampling")
    parser.add_argument("--mdlm_checkpoint", type=str, required=True)
    parser.add_argument("--gaze_checkpoint", type=str, required=True)
    parser.add_argument("--lam", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/gazediffuse_generations.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading gaze predictor from %s", args.gaze_checkpoint)
    gaze_pred, norm_stats = load_trained_predictor(args.gaze_checkpoint, device)

    logger.info("Loading MDLM from %s", args.mdlm_checkpoint)
    # MDLM loading depends on the mdlm submodule's API — adapt path on HPC
    # For now, use a HuggingFace-compatible masked LM as a stand-in
    from transformers import AutoModelForMaskedLM

    mdlm = AutoModelForMaskedLM.from_pretrained(args.mdlm_checkpoint).to(device)
    mdlm.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.mdlm_checkpoint)

    config = GazeDiffuseConfig(
        lam=args.lam,
        steps=args.steps,
        gen_length=args.gen_length,
        top_k=args.top_k,
        temperature=args.temperature,
        mask_token_id=tokenizer.mask_token_id,
    )

    from src.data.prompts import get_prompts

    prompts = get_prompts(args.n_prompts)
    samples_per_prompt = args.n_samples // len(prompts)

    results = generate_samples(
        mdlm=mdlm,
        gaze_predictor=gaze_pred,
        tokenizer=tokenizer,
        prompts=prompts,
        config=config,
        n_samples_per_prompt=samples_per_prompt,
    )

    save_generations(results, args.output)
    logger.info("Done. Generated %d samples with lam=%.1f, steps=%d", len(results), args.lam, args.steps)
