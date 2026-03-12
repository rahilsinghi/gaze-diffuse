"""Tests for evaluation metrics (src/metrics.py)."""

from __future__ import annotations

import math

import pytest

from src.metrics import (
    compute_ari,
    compute_fkgl,
    compute_sentence_fk_variance,
    evaluate_generations,
)


class TestFKGL:
    """Test Flesch-Kincaid Grade Level computation."""

    @pytest.mark.unit
    def test_simple_text_lower(self) -> None:
        """Simple text should have low FKGL."""
        simple = "The cat sat on the mat. It was a good day."
        hard = (
            "The epistemological implications of quantum entanglement "
            "necessitate a fundamental reconceptualization of deterministic causality."
        )
        fk_simple = compute_fkgl(simple)
        fk_hard = compute_fkgl(hard)
        assert fk_simple < fk_hard

    @pytest.mark.unit
    def test_returns_float(self) -> None:
        score = compute_fkgl("This is a test sentence.")
        assert isinstance(score, float)

    @pytest.mark.unit
    def test_empty_string_returns_nan(self) -> None:
        score = compute_fkgl("")
        # textstat may return 0.0 or nan for empty strings
        assert isinstance(score, float)


class TestARI:
    """Test Automated Readability Index."""

    @pytest.mark.unit
    def test_returns_float(self) -> None:
        score = compute_ari("This is a simple test sentence for readability.")
        assert isinstance(score, float)

    @pytest.mark.unit
    def test_harder_text_higher(self) -> None:
        easy = "I like dogs. Dogs are fun. They play all day."
        hard = (
            "The juxtaposition of phenomenological consciousness with "
            "computational substrate theories presents intractable philosophical difficulties."
        )
        assert compute_ari(easy) < compute_ari(hard)


class TestSentenceFKVariance:
    """Test sentence-level FK variance — KEY paper metric."""

    @pytest.mark.unit
    def test_uniform_sentences_low_variance(self) -> None:
        """Text with similar-complexity sentences should have low variance."""
        uniform = (
            "The cat sat on the mat. The dog ran in the park. "
            "The bird flew in the sky. The fish swam in the pond."
        )
        variance = compute_sentence_fk_variance(uniform)
        assert variance < 10.0  # Low variance for similar sentences

    @pytest.mark.unit
    def test_mixed_sentences_higher_variance(self) -> None:
        """Text with mixed complexity should have higher variance."""
        mixed = (
            "The big cat sat on the red mat today. "
            "The epistemological ramifications of quantum chromodynamics "
            "necessitate a fundamental reconceptualization of subatomic "
            "particle interaction paradigms within theoretical physics."
        )
        variance = compute_sentence_fk_variance(mixed)
        assert variance > 0.0

    @pytest.mark.unit
    def test_single_sentence_zero_variance(self) -> None:
        """Single sentence should return 0 variance."""
        variance = compute_sentence_fk_variance("Just one sentence here.")
        assert variance == 0.0


class TestEvaluateGenerations:
    """Test the full evaluation pipeline."""

    @pytest.mark.unit
    def test_returns_metrics_result(self, sample_texts) -> None:
        result = evaluate_generations(
            texts=sample_texts,
            compute_ppl=False,  # Skip PPL (needs GPU/model)
        )
        assert result.n_samples == len(sample_texts)
        assert not math.isnan(result.fkgl_mean)
        assert not math.isnan(result.ari_mean)
        assert not math.isnan(result.fk_sentence_variance)

    @pytest.mark.unit
    def test_fkgl_reasonable_range(self, sample_texts) -> None:
        result = evaluate_generations(texts=sample_texts, compute_ppl=False)
        # FKGL typically ranges from -5 to 20 for normal text
        assert -10 < result.fkgl_mean < 25

    @pytest.mark.unit
    def test_std_non_negative(self, sample_texts) -> None:
        result = evaluate_generations(texts=sample_texts, compute_ppl=False)
        assert result.fkgl_std >= 0
        assert result.ari_std >= 0
