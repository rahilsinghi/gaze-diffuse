"""AR gaze guidance baseline — Sauberli et al. replication (Experiment 3).

Implements gaze-guided decoding on GPT-2 medium: at each autoregressive
step, score top-k candidates by predicted fixation duration and re-rank
with lambda weighting.

This is the KEY COMPARISON in the paper — GazeDiffuse must outperform
this sequential baseline.

Usage:
    python -m src.ar_baseline \
        --gaze_checkpoint checkpoints/gaze_predictor/gaze_predictor_best.pt \
        --lam -1.0 --n_samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gaze_predictor import GazePredictor, load_trained_predictor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ARGazeConfig:
    """Configuration for AR gaze-guided generation."""

    model_name: str = "gpt2-medium"
    lam: float = -1.0  # <0 = easier, >0 = harder, 0 = unguided
    top_k: int = 50  # Top-k candidates to score with gaze predictor
    max_new_tokens: int = 128
    temperature: float = 1.0
    gaze_temperature: float = 1.0


@torch.no_grad()
def ar_gaze_guided_generate(
    lm: torch.nn.Module,
    gaze_predictor: GazePredictor,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    config: ARGazeConfig,
) -> torch.Tensor:
    """Generate text with AR gaze guidance (Sauberli et al.).

    At each step:
    1. Get LM logits for next token
    2. Select top-k candidates
    3. Score each candidate with gaze predictor (fixation duration)
    4. Re-rank: score = log P_LM(tok) + lambda * gaze(tok)
    5. Pick highest-scoring token

    Args:
        lm: Pretrained autoregressive LM (frozen)
        gaze_predictor: Trained gaze predictor (frozen)
        tokenizer: Tokenizer
        prompt_ids: [prompt_len] token IDs
        config: Generation configuration

    Returns:
        [prompt_len + max_new_tokens] generated token IDs
    """
    device = prompt_ids.device
    generated = prompt_ids.clone()

    for _ in range(config.max_new_tokens):
        # LM forward pass
        outputs = lm(generated.unsqueeze(0))
        next_logits = outputs.logits[0, -1, :]  # [V]
        next_log_probs = F.log_softmax(
            next_logits / config.temperature, dim=-1
        )

        # Top-k candidates
        top_k_log_probs, top_k_ids = next_log_probs.topk(config.top_k)

        if config.lam != 0.0:
            # Score each candidate with gaze predictor
            gaze_scores = gaze_predictor.score_vocabulary(
                sequence=generated,
                position=len(generated),  # Next position
                vocab_candidates=top_k_ids,
                tokenizer=tokenizer,
            )

            # Normalize gaze scores
            if gaze_scores.std() > 1e-6:
                gaze_scores = (gaze_scores - gaze_scores.mean()) / (
                    gaze_scores.std() + 1e-8
                )

            # Combine: LM + lambda * gaze
            guided_scores = top_k_log_probs + config.lam * (
                gaze_scores / config.gaze_temperature
            )
        else:
            guided_scores = top_k_log_probs

        # Select best token
        best_idx = guided_scores.argmax()
        next_token = top_k_ids[best_idx].unsqueeze(0)

        generated = torch.cat([generated, next_token])

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated


def generate_ar_samples(
    lm: torch.nn.Module,
    gaze_predictor: GazePredictor,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    config: ARGazeConfig,
    n_samples_per_prompt: int = 4,
) -> list[dict[str, str | float]]:
    """Generate samples using AR gaze guidance for all prompts."""
    device = next(lm.parameters()).device
    results: list[dict[str, str | float]] = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        prompt_ids = prompt_ids.to(device)

        for sample_idx in range(n_samples_per_prompt):
            output_ids = ar_gaze_guided_generate(
                lm=lm,
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
                "method": "ar_gaze",
                "model": config.model_name,
            })

        if (prompt_idx + 1) % 10 == 0:
            logger.info("AR baseline: %d/%d prompts done", prompt_idx + 1, len(prompts))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="AR gaze guidance baseline (Sauberli replication)")
    parser.add_argument("--model_name", type=str, default="gpt2-medium")
    parser.add_argument("--gaze_checkpoint", type=str, required=True)
    parser.add_argument("--lam", type=float, default=-1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/ar_baseline_generations.jsonl")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Loading gaze predictor from %s", args.gaze_checkpoint)
    gaze_pred, _ = load_trained_predictor(args.gaze_checkpoint, device)

    logger.info("Loading LM: %s", args.model_name)
    lm = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    lm.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = ARGazeConfig(
        model_name=args.model_name,
        lam=args.lam,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    from src.data.prompts import get_prompts

    prompts = get_prompts(args.n_prompts)
    samples_per_prompt = max(1, args.n_samples // len(prompts))

    results = generate_ar_samples(
        lm=lm,
        gaze_predictor=gaze_pred,
        tokenizer=tokenizer,
        prompts=prompts,
        config=config,
        n_samples_per_prompt=samples_per_prompt,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("AR baseline done. %d samples saved to %s", len(results), args.output)
