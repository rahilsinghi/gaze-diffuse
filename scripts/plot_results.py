"""Publication-quality result plots for the GazeDiffuse EMNLP 2026 paper.

Generates four figures:
  1. FKGL vs Lambda curve (primary result)
  2. FK Sentence Variance bar chart (global coherence claim)
  3. MAUVE preservation bar chart (fluency preservation)
  4. Readability radar/spider chart (multi-metric comparison)

Usage:
    # From real results
    python scripts/plot_results.py --results_dir results/ --output_dir results/figures/

    # Demo mode with synthetic data
    python scripts/plot_results.py --demo --output_dir results/figures/

    # Dark background
    python scripts/plot_results.py --demo --output_dir results/figures/ --style dark
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# Use non-interactive backend for server/HPC environments -- must be set
# before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAMBDA_VALUES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

METHOD_LABELS = {
    "gazediffuse_mdlm": "GazeDiffuse (MDLM)",
    "gazediffuse_llada": "GazeDiffuse (LLaDA)",
    "ar_baseline": "AR Baseline (GPT-2)",
}

# Colorblind-friendly palette (Wong 2011)
METHOD_COLORS = {
    "gazediffuse_mdlm": "#0072B2",
    "gazediffuse_llada": "#D55E00",
    "ar_baseline": "#999999",
}

METHOD_MARKERS = {
    "gazediffuse_mdlm": "o",
    "gazediffuse_llada": "s",
    "ar_baseline": "^",
}

DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 300

RADAR_METRICS = ["FKGL", "ARI", "Self-PPL", "MAUVE", "FK-Var"]


# ---------------------------------------------------------------------------
# MetricsResult (mirrors src/metrics.py, no import needed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsResult:
    """Evaluation metrics for a set of generated texts."""

    fkgl_mean: float
    fkgl_std: float
    ari_mean: float
    ari_std: float
    self_ppl: float
    fk_sentence_variance: float
    mauve_score: float | None = None
    n_samples: int = 0


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------


def apply_style(style: str) -> dict[str, Any]:
    """Configure matplotlib rcParams for publication-quality output.

    Returns a dict of colors used by plots so callers can adapt.
    """
    base_params: dict[str, Any] = {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    }

    if style == "dark":
        base_params.update({
            "figure.facecolor": "#1e1e1e",
            "axes.facecolor": "#1e1e1e",
            "savefig.facecolor": "#1e1e1e",
            "text.color": "#e0e0e0",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#e0e0e0",
            "ytick.color": "#e0e0e0",
            "axes.edgecolor": "#555555",
        })
        theme = {
            "text": "#e0e0e0",
            "grid": "#333333",
            "spine": "#555555",
        }
    else:
        base_params.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#333333",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.edgecolor": "#333333",
        })
        theme = {
            "text": "#333333",
            "grid": "#dddddd",
            "spine": "#333333",
        }

    matplotlib.rcParams.update(base_params)
    return theme


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metrics_json(path: Path) -> MetricsResult:
    """Load a single metrics JSON file into a MetricsResult."""
    with open(path) as f:
        data = json.load(f)
    return MetricsResult(
        fkgl_mean=data.get("fkgl_mean", float("nan")),
        fkgl_std=data.get("fkgl_std", float("nan")),
        ari_mean=data.get("ari_mean", float("nan")),
        ari_std=data.get("ari_std", float("nan")),
        self_ppl=data.get("self_ppl", float("nan")),
        fk_sentence_variance=data.get("fk_sentence_variance", float("nan")),
        mauve_score=data.get("mauve_score"),
        n_samples=data.get("n_samples", 0),
    )


def discover_results(results_dir: Path) -> dict[str, dict[float, MetricsResult]]:
    """Scan results directory for metric JSON files.

    Expected layout:
        results/<method>/lam_<value>/metrics.json
    or:
        results/<method>_lam<value>_metrics.json

    Returns:
        {method_key: {lambda_value: MetricsResult}}
    """
    results: dict[str, dict[float, MetricsResult]] = {}

    # Pattern 1: nested directories
    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method_key = method_dir.name
        for lam_dir in sorted(method_dir.iterdir()):
            if not lam_dir.is_dir():
                continue
            metrics_file = lam_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            # Parse lambda from directory name (e.g. "lam_-1.0" or "lam_neg1")
            lam_str = lam_dir.name.replace("lam_", "").replace("neg", "-")
            try:
                lam_val = float(lam_str)
            except ValueError:
                logger.warning("Could not parse lambda from %s", lam_dir.name)
                continue
            results.setdefault(method_key, {})[lam_val] = load_metrics_json(
                metrics_file
            )

    # Pattern 2: flat files
    for metrics_file in sorted(results_dir.glob("*_metrics.json")):
        stem = metrics_file.stem.replace("_metrics", "")
        # Try to parse method and lambda from filename
        parts = stem.rsplit("_lam", maxsplit=1)
        if len(parts) == 2:
            method_key = parts[0]
            try:
                lam_val = float(parts[1])
            except ValueError:
                continue
            results.setdefault(method_key, {})[lam_val] = load_metrics_json(
                metrics_file
            )

    if not results:
        logger.warning(
            "No metrics files found in %s. Use --demo for synthetic data.", results_dir
        )

    return results


# ---------------------------------------------------------------------------
# Synthetic / demo data
# ---------------------------------------------------------------------------


def generate_demo_data() -> dict[str, dict[float, MetricsResult]]:
    """Generate plausible synthetic data for all plots.

    Values are designed to tell the GazeDiffuse story:
    - FKGL decreases with negative lambda, increases with positive
    - GazeDiffuse shows stronger FKGL shift than AR baseline
    - GazeDiffuse has LOWER sentence-level FK variance (key claim)
    - MAUVE stays high for GazeDiffuse, drops slightly for AR
    """
    rng = np.random.default_rng(42)

    methods = {
        "gazediffuse_mdlm": {
            "fkgl_base": 8.5,
            "fkgl_slope": 2.8,
            "ari_base": 9.0,
            "ari_slope": 2.5,
            "ppl_base": 28.0,
            "fk_var_base": 3.2,
            "fk_var_slope": 0.4,
            "mauve_base": 0.92,
            "mauve_decay": 0.03,
        },
        "gazediffuse_llada": {
            "fkgl_base": 8.3,
            "fkgl_slope": 2.5,
            "ari_base": 8.8,
            "ari_slope": 2.3,
            "ppl_base": 30.0,
            "fk_var_base": 3.5,
            "fk_var_slope": 0.5,
            "mauve_base": 0.90,
            "mauve_decay": 0.04,
        },
        "ar_baseline": {
            "fkgl_base": 8.4,
            "fkgl_slope": 1.8,
            "ari_base": 8.9,
            "ari_slope": 1.7,
            "ppl_base": 25.0,
            "fk_var_base": 6.8,
            "fk_var_slope": 1.2,
            "mauve_base": 0.88,
            "mauve_decay": 0.06,
        },
    }

    results: dict[str, dict[float, MetricsResult]] = {}

    for method_key, params in methods.items():
        method_results: dict[float, MetricsResult] = {}
        for lam in LAMBDA_VALUES:
            fkgl_mean = params["fkgl_base"] + params["fkgl_slope"] * lam
            fkgl_std = 1.2 + 0.3 * abs(lam) + rng.normal(0, 0.1)

            ari_mean = params["ari_base"] + params["ari_slope"] * lam
            ari_std = 1.4 + 0.3 * abs(lam) + rng.normal(0, 0.1)

            self_ppl = params["ppl_base"] + 3.0 * abs(lam) + rng.normal(0, 1.0)

            fk_var = params["fk_var_base"] + params["fk_var_slope"] * abs(lam)
            fk_var = max(0.5, fk_var + rng.normal(0, 0.3))

            mauve = params["mauve_base"] - params["mauve_decay"] * abs(lam)
            mauve = float(np.clip(mauve + rng.normal(0, 0.01), 0.5, 1.0))

            method_results[lam] = MetricsResult(
                fkgl_mean=round(fkgl_mean, 2),
                fkgl_std=round(max(0.3, fkgl_std), 2),
                ari_mean=round(ari_mean, 2),
                ari_std=round(max(0.3, ari_std), 2),
                self_ppl=round(self_ppl, 1),
                fk_sentence_variance=round(fk_var, 3),
                mauve_score=round(mauve, 3),
                n_samples=200,
            )
        results[method_key] = method_results

    return results


# ---------------------------------------------------------------------------
# Plot 1: FKGL vs Lambda curve (primary result figure)
# ---------------------------------------------------------------------------


def plot_fkgl_vs_lambda(
    data: dict[str, dict[float, MetricsResult]],
    output_path: Path,
    theme: dict[str, str],
) -> None:
    """Line plot: FKGL (y) vs lambda (x) with error bars per method."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for method_key in ["gazediffuse_mdlm", "gazediffuse_llada", "ar_baseline"]:
        if method_key not in data:
            continue
        method_data = data[method_key]
        lambdas = sorted(method_data.keys())
        means = [method_data[l].fkgl_mean for l in lambdas]
        stds = [method_data[l].fkgl_std for l in lambdas]

        ax.errorbar(
            lambdas,
            means,
            yerr=stds,
            label=METHOD_LABELS.get(method_key, method_key),
            color=METHOD_COLORS.get(method_key, "#333333"),
            marker=METHOD_MARKERS.get(method_key, "o"),
            capsize=3,
            capthick=1.0,
            elinewidth=0.8,
        )

    ax.set_xlabel(r"Guidance strength $\lambda$")
    ax.set_ylabel("Flesch-Kincaid Grade Level (FKGL)")
    ax.set_title("Readability Control via Gaze Guidance")

    # Add a subtle reference line at lambda=0
    ax.axvline(x=0, color=theme["grid"], linestyle="--", linewidth=0.8, zorder=0)

    # Annotate simplification/complexification regions
    y_min, y_max = ax.get_ylim()
    ax.text(
        -1.5,
        y_max - 0.3 * (y_max - y_min),
        "Simplification",
        ha="center",
        va="center",
        fontsize=8,
        fontstyle="italic",
        color=theme["text"],
        alpha=0.5,
    )
    ax.text(
        1.5,
        y_max - 0.3 * (y_max - y_min),
        "Complexification",
        ha="center",
        va="center",
        fontsize=8,
        fontstyle="italic",
        color=theme["text"],
        alpha=0.5,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved FKGL vs lambda plot to %s", output_path)


# ---------------------------------------------------------------------------
# Plot 2: FK Sentence Variance bar chart
# ---------------------------------------------------------------------------


def plot_fk_variance(
    data: dict[str, dict[float, MetricsResult]],
    output_path: Path,
    theme: dict[str, str],
) -> None:
    """Grouped bar chart: FK sentence variance at lambda = -1, 0, +1."""
    target_lambdas = [-1.0, 0.0, 1.0]
    method_keys = [k for k in ["gazediffuse_mdlm", "gazediffuse_llada", "ar_baseline"] if k in data]

    n_groups = len(target_lambdas)
    n_methods = len(method_keys)
    bar_width = 0.22
    group_positions = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for method_idx, method_key in enumerate(method_keys):
        method_data = data[method_key]
        values = []
        for lam in target_lambdas:
            if lam in method_data:
                values.append(method_data[lam].fk_sentence_variance)
            else:
                # Find closest lambda
                closest = min(method_data.keys(), key=lambda x: abs(x - lam))
                values.append(method_data[closest].fk_sentence_variance)

        offset = (method_idx - (n_methods - 1) / 2) * bar_width
        ax.bar(
            group_positions + offset,
            values,
            bar_width,
            label=METHOD_LABELS.get(method_key, method_key),
            color=METHOD_COLORS.get(method_key, "#333333"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel(r"Guidance strength $\lambda$")
    ax.set_ylabel("FK Sentence Variance")
    ax.set_title("Sentence-Level Readability Consistency")
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"$\\lambda={l:+.0f}$" for l in target_lambdas])

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=n_methods,
        frameon=False,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved FK variance plot to %s", output_path)


# ---------------------------------------------------------------------------
# Plot 3: MAUVE preservation bar chart
# ---------------------------------------------------------------------------


def plot_mauve_preservation(
    data: dict[str, dict[float, MetricsResult]],
    output_path: Path,
    theme: dict[str, str],
) -> None:
    """Bar chart: MAUVE scores across methods and lambda values."""
    target_lambdas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    method_keys = [k for k in ["gazediffuse_mdlm", "gazediffuse_llada", "ar_baseline"] if k in data]

    n_groups = len(target_lambdas)
    n_methods = len(method_keys)
    bar_width = 0.22
    group_positions = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7, 4))

    for method_idx, method_key in enumerate(method_keys):
        method_data = data[method_key]
        values = []
        for lam in target_lambdas:
            if lam in method_data and method_data[lam].mauve_score is not None:
                values.append(method_data[lam].mauve_score)
            else:
                values.append(0.0)

        offset = (method_idx - (n_methods - 1) / 2) * bar_width
        ax.bar(
            group_positions + offset,
            values,
            bar_width,
            label=METHOD_LABELS.get(method_key, method_key),
            color=METHOD_COLORS.get(method_key, "#333333"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel(r"Guidance strength $\lambda$")
    ax.set_ylabel("MAUVE Score")
    ax.set_title("Fluency Preservation Under Guidance")
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"$\\lambda={l:+.0f}$" for l in target_lambdas])
    ax.set_ylim(0.0, 1.05)

    # Reference line at MAUVE=0.9 (typical good threshold)
    ax.axhline(y=0.9, color=theme["grid"], linestyle=":", linewidth=0.8, zorder=0)
    ax.text(
        n_groups - 0.5,
        0.91,
        "0.90",
        fontsize=7,
        color=theme["text"],
        alpha=0.5,
        ha="right",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=n_methods,
        frameon=False,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved MAUVE preservation plot to %s", output_path)


# ---------------------------------------------------------------------------
# Plot 4: Radar / spider chart
# ---------------------------------------------------------------------------


def _normalize_for_radar(
    values: dict[str, float],
    ranges: dict[str, tuple[float, float]],
) -> list[float]:
    """Normalize metric values to [0, 1] for radar chart.

    For metrics where lower is better (FKGL, ARI, Self-PPL, FK-Var),
    we invert so that outer = better.
    """
    # Metrics where lower raw value = better performance
    lower_is_better = {"FKGL", "ARI", "Self-PPL", "FK-Var"}
    normalized = []
    for metric in RADAR_METRICS:
        lo, hi = ranges[metric]
        raw = values.get(metric, 0.0)
        if hi - lo < 1e-9:
            normalized.append(0.5)
            continue
        frac = (raw - lo) / (hi - lo)
        frac = float(np.clip(frac, 0.0, 1.0))
        if metric in lower_is_better:
            frac = 1.0 - frac
        normalized.append(frac)
    return normalized


def plot_radar(
    data: dict[str, dict[float, MetricsResult]],
    output_path: Path,
    theme: dict[str, str],
    radar_lambda: float = -1.0,
) -> None:
    """Radar chart comparing methods on all metrics at a single lambda."""
    method_keys = [k for k in ["gazediffuse_mdlm", "gazediffuse_llada", "ar_baseline"] if k in data]

    # Collect raw values per method
    raw_values: dict[str, dict[str, float]] = {}
    for method_key in method_keys:
        method_data = data[method_key]
        lam = radar_lambda
        if lam not in method_data:
            closest = min(method_data.keys(), key=lambda x: abs(x - lam))
            lam = closest
        m = method_data[lam]
        raw_values[method_key] = {
            "FKGL": m.fkgl_mean,
            "ARI": m.ari_mean,
            "Self-PPL": m.self_ppl if not math.isnan(m.self_ppl) else 50.0,
            "MAUVE": m.mauve_score if m.mauve_score is not None else 0.0,
            "FK-Var": m.fk_sentence_variance,
        }

    # Compute ranges across all methods for normalization
    ranges: dict[str, tuple[float, float]] = {}
    for metric in RADAR_METRICS:
        all_vals = [rv[metric] for rv in raw_values.values()]
        lo = min(all_vals)
        hi = max(all_vals)
        margin = (hi - lo) * 0.2 if hi > lo else 1.0
        ranges[metric] = (lo - margin, hi + margin)

    # Angles for radar
    n_metrics = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles.append(angles[0])  # Close the polygon

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})

    for method_key in method_keys:
        normed = _normalize_for_radar(raw_values[method_key], ranges)
        normed.append(normed[0])  # Close polygon

        ax.plot(
            angles,
            normed,
            color=METHOD_COLORS.get(method_key, "#333333"),
            linewidth=1.8,
            label=METHOD_LABELS.get(method_key, method_key),
        )
        ax.fill(
            angles,
            normed,
            color=METHOD_COLORS.get(method_key, "#333333"),
            alpha=0.08,
        )

    # Configure radar axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_METRICS, fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Multi-Metric Comparison ($\\lambda={radar_lambda:+.0f}$)",
        y=1.08,
        fontsize=12,
    )

    # Subtle radial grid
    ax.set_rgrids(
        [0.25, 0.5, 0.75],
        labels=["", "", ""],
        angle=0,
    )
    ax.spines["polar"].set_color(theme["grid"])
    ax.grid(color=theme["grid"], linewidth=0.5, alpha=0.5)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(method_keys),
        frameon=False,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved radar chart to %s", output_path)


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


def plot_from_metrics_dict(
    metrics: dict[str, MetricsResult],
    output_dir: str | Path = "results/figures",
    style: str = "light",
    radar_lambda: float = -1.0,
) -> list[Path]:
    """Generate all plots from a flat {method_key: MetricsResult} dict.

    This is the programmatic entry point. The dict maps a composite key
    of the form "<method>__lam<value>" to a MetricsResult, e.g.:
        {
            "gazediffuse_mdlm__lam-1.0": MetricsResult(...),
            "gazediffuse_mdlm__lam0.0": MetricsResult(...),
            "ar_baseline__lam-1.0": MetricsResult(...),
            ...
        }

    Args:
        metrics: Composite-keyed dict mapping to MetricsResult objects.
        output_dir: Directory for saving figures.
        style: "light" or "dark".
        radar_lambda: Lambda value for the radar chart.

    Returns:
        List of paths to saved figure files.
    """
    theme = apply_style(style)
    output_dir = Path(output_dir)

    # Parse composite keys into nested structure
    data: dict[str, dict[float, MetricsResult]] = {}
    for composite_key, result in metrics.items():
        parts = composite_key.rsplit("__lam", maxsplit=1)
        if len(parts) == 2:
            method_key = parts[0]
            try:
                lam_val = float(parts[1])
            except ValueError:
                logger.warning("Cannot parse lambda from key: %s", composite_key)
                continue
            data.setdefault(method_key, {})[lam_val] = result
        else:
            logger.warning(
                "Key '%s' does not match expected format '<method>__lam<value>'",
                composite_key,
            )

    if not data:
        logger.error("No valid entries in metrics dict.")
        return []

    return _generate_all_plots(data, output_dir, theme, radar_lambda)


def _generate_all_plots(
    data: dict[str, dict[float, MetricsResult]],
    output_dir: Path,
    theme: dict[str, str],
    radar_lambda: float = -1.0,
    file_ext: str = ".pdf",
) -> list[Path]:
    """Run all four plot functions and return saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fkgl_path = output_dir / f"fkgl_vs_lambda{file_ext}"
    plot_fkgl_vs_lambda(data, fkgl_path, theme)
    saved.append(fkgl_path)

    fkvar_path = output_dir / f"fk_sentence_variance{file_ext}"
    plot_fk_variance(data, fkvar_path, theme)
    saved.append(fkvar_path)

    mauve_path = output_dir / f"mauve_preservation{file_ext}"
    plot_mauve_preservation(data, mauve_path, theme)
    saved.append(mauve_path)

    radar_path = output_dir / f"radar_comparison{file_ext}"
    plot_radar(data, radar_path, theme, radar_lambda=radar_lambda)
    saved.append(radar_path)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots for GazeDiffuse (EMNLP 2026).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing metrics JSON files (default: results/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures (default: results/figures/)",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["light", "dark"],
        default="light",
        help="Plot style: light (default) or dark background",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate plots with synthetic data (no real results needed)",
    )
    parser.add_argument(
        "--radar_lambda",
        type=float,
        default=-1.0,
        help="Lambda value for the radar chart (default: -1.0)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output file format (default: pdf)",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    theme = apply_style(args.style)
    output_dir = Path(args.output_dir)

    if args.demo:
        logger.info("Running in demo mode with synthetic data")
        data = generate_demo_data()
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            logger.error("Results directory does not exist: %s", results_dir)
            logger.info("Use --demo to generate plots with synthetic data.")
            sys.exit(1)
        data = discover_results(results_dir)
        if not data:
            logger.error("No results found. Use --demo for synthetic data.")
            sys.exit(1)

    saved = _generate_all_plots(
        data, output_dir, theme, args.radar_lambda, file_ext=f".{args.format}"
    )

    logger.info("All figures saved to %s/", output_dir)
    for path in saved:
        logger.info("  %s", path.name)


if __name__ == "__main__":
    main()
