"""Evaluation metrics for GazeDiffuse experiments (Experiment 1-5 evaluation).

Computes all metrics needed for the paper's results table:
- Flesch-Kincaid Grade Level (FKGL)
- Automated Readability Index (ARI)
- MAUVE score (distributional similarity)
- Self-perplexity (fluency via base model)
- Sentence-level FK variance (global coherence measure)

Usage:
    python -m src.metrics --input results/generations.jsonl --reference results/unguided.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import textstat
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsResult:
    """Evaluation metrics for a set of generated texts."""

    fkgl_mean: float
    fkgl_std: float
    ari_mean: float
    ari_std: float
    self_ppl: float
    fk_sentence_variance: float  # Key metric: global coherence
    mauve_score: float | None = None  # Computed separately (needs reference)
    n_samples: int = 0


def compute_fkgl(text: str) -> float:
    """Compute Flesch-Kincaid Grade Level for a text.

    Lower = easier to read. Negative values possible for very simple text.
    """
    try:
        return textstat.flesch_kincaid_grade(text)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def compute_ari(text: str) -> float:
    """Compute Automated Readability Index.

    Lower = easier. Corroborates FKGL.
    """
    try:
        return textstat.automated_readability_index(text)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def compute_sentence_fk_variance(text: str) -> float:
    """Compute variance of FKGL across sentences within a text.

    This is the KEY METRIC for the paper's "global coherence" claim:
    lower variance = more consistent readability across the generated text.
    Parallel guidance (GazeDiffuse) should produce lower variance than
    sequential AR guidance.
    """
    # Split into sentences using nltk (textstat >= 0.7 removed textstatistics)
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) < 2:
        return 0.0

    fk_per_sentence = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 3:
            continue
        fk = compute_fkgl(sent)
        if not math.isnan(fk):
            fk_per_sentence.append(fk)

    if len(fk_per_sentence) < 2:
        return 0.0

    return float(np.var(fk_per_sentence))


@torch.no_grad()
def compute_self_perplexity(
    texts: list[str],
    model_name: str = "gpt2-medium",
    max_length: int = 512,
    batch_size: int = 8,
) -> float:
    """Compute self-perplexity: run generated text through a base LM.

    Higher PPL = less fluent generation. We want guided text to
    maintain low perplexity (stay fluent despite guidance).

    Args:
        texts: Generated texts to evaluate
        model_name: LM for perplexity computation
        max_length: Max tokens per text
        batch_size: Batch size for evaluation

    Returns:
        Mean perplexity across all texts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        # Mask out padding tokens from loss
        n_tokens = attention_mask.sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return perplexity


def compute_mauve_score(
    generated_texts: list[str],
    reference_texts: list[str],
) -> float:
    """Compute MAUVE score between generated and reference texts.

    MAUVE measures distributional similarity. Score in [0, 1]:
    higher = generated distribution closer to reference.
    Drops significantly if guidance hurts fluency/diversity.

    Requires: pip install mauve-text
    """
    try:
        import mauve

        result = mauve.compute_mauve(
            p_text=reference_texts,
            q_text=generated_texts,
            device_id=0 if torch.cuda.is_available() else -1,
            max_text_length=512,
            verbose=False,
        )
        return float(result.mauve)
    except ImportError:
        logger.warning("mauve-text not installed, skipping MAUVE computation")
        return float("nan")
    except Exception as e:
        logger.warning("MAUVE computation failed: %s", e)
        return float("nan")


def evaluate_generations(
    texts: list[str],
    reference_texts: list[str] | None = None,
    compute_ppl: bool = True,
    ppl_model: str = "gpt2-medium",
) -> MetricsResult:
    """Run full evaluation pipeline on generated texts.

    Args:
        texts: Generated texts to evaluate
        reference_texts: Unguided texts for MAUVE comparison
        compute_ppl: Whether to compute self-perplexity (slow, needs GPU)
        ppl_model: Model name for perplexity computation

    Returns:
        MetricsResult with all metrics
    """
    # Readability metrics
    fkgl_scores = [compute_fkgl(t) for t in texts]
    ari_scores = [compute_ari(t) for t in texts]
    fk_variances = [compute_sentence_fk_variance(t) for t in texts]

    # Filter NaN
    fkgl_scores = [s for s in fkgl_scores if not math.isnan(s)]
    ari_scores = [s for s in ari_scores if not math.isnan(s)]
    fk_variances = [v for v in fk_variances if not math.isnan(v)]

    # Self-perplexity
    self_ppl = float("nan")
    if compute_ppl:
        self_ppl = compute_self_perplexity(texts, model_name=ppl_model)

    # MAUVE
    mauve_score = None
    if reference_texts is not None:
        mauve_score = compute_mauve_score(texts, reference_texts)

    return MetricsResult(
        fkgl_mean=float(np.mean(fkgl_scores)) if fkgl_scores else float("nan"),
        fkgl_std=float(np.std(fkgl_scores)) if fkgl_scores else float("nan"),
        ari_mean=float(np.mean(ari_scores)) if ari_scores else float("nan"),
        ari_std=float(np.std(ari_scores)) if ari_scores else float("nan"),
        self_ppl=self_ppl,
        fk_sentence_variance=float(np.mean(fk_variances)) if fk_variances else float("nan"),
        mauve_score=mauve_score,
        n_samples=len(texts),
    )


def load_generations(path: str | Path) -> list[dict]:
    """Load generations from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def print_results_table(
    results: dict[str, MetricsResult],
) -> str:
    """Format results as a markdown table matching the paper template."""
    header = (
        "| Method | FKGL (lam=-1) | FKGL (lam=0) | FKGL (lam=+1) "
        "| MAUVE | Self-PPL | FK Variance |"
    )
    separator = "|--------|---------------|--------------|---------------|-------|----------|-------------|"
    lines = [header, separator]

    for method, metrics in results.items():
        mauve_str = f"{metrics.mauve_score:.3f}" if metrics.mauve_score else "—"
        ppl_str = f"{metrics.self_ppl:.1f}" if not math.isnan(metrics.self_ppl) else "—"
        line = (
            f"| {method} "
            f"| {metrics.fkgl_mean:.2f} ± {metrics.fkgl_std:.2f} "
            f"| — "
            f"| — "
            f"| {mauve_str} "
            f"| {ppl_str} "
            f"| {metrics.fk_sentence_variance:.3f} |"
        )
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate GazeDiffuse generations")
    parser.add_argument("--input", type=str, required=True, help="JSONL file with generations")
    parser.add_argument("--reference", type=str, default=None, help="JSONL file with unguided reference")
    parser.add_argument("--no_ppl", action="store_true", help="Skip perplexity computation")
    parser.add_argument("--ppl_model", type=str, default="gpt2-medium")
    parser.add_argument("--output", type=str, default=None, help="Save metrics as JSON")
    args = parser.parse_args()

    # Load generations
    generations = load_generations(args.input)
    texts = [g["full_text"] for g in generations]
    logger.info("Loaded %d generations from %s", len(texts), args.input)

    # Load reference if provided
    ref_texts = None
    if args.reference:
        ref_gens = load_generations(args.reference)
        ref_texts = [g["full_text"] for g in ref_gens]

    # Evaluate
    metrics = evaluate_generations(
        texts=texts,
        reference_texts=ref_texts,
        compute_ppl=not args.no_ppl,
        ppl_model=args.ppl_model,
    )

    logger.info("Results:")
    logger.info("  FKGL:  %.2f ± %.2f", metrics.fkgl_mean, metrics.fkgl_std)
    logger.info("  ARI:   %.2f ± %.2f", metrics.ari_mean, metrics.ari_std)
    logger.info("  Self-PPL: %.1f", metrics.self_ppl)
    logger.info("  FK Sentence Variance: %.4f", metrics.fk_sentence_variance)
    if metrics.mauve_score is not None:
        logger.info("  MAUVE: %.3f", metrics.mauve_score)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "fkgl_mean": metrics.fkgl_mean,
                    "fkgl_std": metrics.fkgl_std,
                    "ari_mean": metrics.ari_mean,
                    "ari_std": metrics.ari_std,
                    "self_ppl": metrics.self_ppl,
                    "fk_sentence_variance": metrics.fk_sentence_variance,
                    "mauve_score": metrics.mauve_score,
                    "n_samples": metrics.n_samples,
                },
                f,
                indent=2,
            )
        logger.info("Metrics saved to %s", args.output)
