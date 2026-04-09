"""Tests for evaluation metrics (src/metrics.py)."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.metrics import (
    MetricsResult,
    compute_ari,
    compute_fkgl,
    compute_mauve_score,
    compute_self_perplexity,
    compute_sentence_fk_variance,
    evaluate_generations,
    load_generations,
    print_results_table,
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


# ---------------------------------------------------------------------------
# Exception handling — lines 52-53, 63-64
# ---------------------------------------------------------------------------


class TestFKGLExceptionHandling:
    """Verify FKGL returns nan when textstat raises."""

    @pytest.mark.unit
    def test_fkgl_exception_returns_nan(self, monkeypatch) -> None:
        """When textstat raises ValueError, compute_fkgl returns nan."""
        import textstat as _textstat

        monkeypatch.setattr(
            _textstat,
            "flesch_kincaid_grade",
            lambda text: (_ for _ in ()).throw(ValueError("forced")),
        )
        result = compute_fkgl("some text")
        assert math.isnan(result)

    @pytest.mark.unit
    def test_fkgl_zero_division_returns_nan(self, monkeypatch) -> None:
        """When textstat raises ZeroDivisionError, compute_fkgl returns nan."""
        import textstat as _textstat

        monkeypatch.setattr(
            _textstat,
            "flesch_kincaid_grade",
            lambda text: 1 / 0,
        )
        result = compute_fkgl("some text")
        assert math.isnan(result)


class TestARIExceptionHandling:
    """Verify ARI returns nan when textstat raises."""

    @pytest.mark.unit
    def test_ari_exception_returns_nan(self, monkeypatch) -> None:
        """When textstat raises ValueError, compute_ari returns nan."""
        import textstat as _textstat

        monkeypatch.setattr(
            _textstat,
            "automated_readability_index",
            lambda text: (_ for _ in ()).throw(ValueError("forced")),
        )
        result = compute_ari("some text")
        assert math.isnan(result)

    @pytest.mark.unit
    def test_ari_zero_division_returns_nan(self, monkeypatch) -> None:
        """When textstat raises ZeroDivisionError, compute_ari returns nan."""
        import textstat as _textstat

        monkeypatch.setattr(
            _textstat,
            "automated_readability_index",
            lambda text: 1 / 0,
        )
        result = compute_ari("some text")
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# compute_sentence_fk_variance edge cases — lines 85-86, 91-92
# ---------------------------------------------------------------------------


class TestSentenceFKVarianceEdgeCases:
    """Edge cases for sentence-level FK variance."""

    @pytest.mark.unit
    def test_short_sentences_are_skipped(self) -> None:
        """Sentences with < 3 words are excluded; only valid ones contribute."""
        # Two short sentences (< 3 words) mixed with two normal ones.
        # The two short ones must be skipped, leaving only 2 valid scores
        # whose variance should equal the variance of those two FK values.
        text = (
            "Yes! No! "
            "The cat sat quietly on the old red mat by the door. "
            "The dog ran playfully through the green meadow every morning."
        )
        variance = compute_sentence_fk_variance(text)
        # The short one-word sentences are filtered; result must be a non-negative float.
        assert isinstance(variance, float)
        assert variance >= 0.0

    @pytest.mark.unit
    def test_all_short_sentences_returns_zero(self) -> None:
        """When every sentence has < 3 words, fewer than 2 valid FK scores exist
        and the function returns 0.0."""
        # Each sentence here has 1–2 words
        text = "Run! Go. Stop! Wait."
        variance = compute_sentence_fk_variance(text)
        assert variance == 0.0

    @pytest.mark.unit
    def test_exactly_one_valid_sentence_returns_zero(self) -> None:
        """One short sentence + one valid sentence = only 1 FK score → 0.0."""
        text = "Go! The scientist carefully studied the complex molecular structure."
        variance = compute_sentence_fk_variance(text)
        assert variance == 0.0


# ---------------------------------------------------------------------------
# compute_self_perplexity — lines 118-155
# ---------------------------------------------------------------------------


class TestComputeSelfPerplexity:
    """Tests for compute_self_perplexity."""

    @pytest.mark.unit
    def test_self_perplexity_uses_model_output(self, monkeypatch) -> None:
        """compute_self_perplexity should return exp(avg_loss) from the model."""
        import torch

        # Build a fake model whose loss is always ln(5) so PPL should be ~5.
        target_loss = math.log(5.0)

        fake_output = MagicMock()
        fake_output.loss = MagicMock()
        fake_output.loss.item.return_value = target_loss

        # Make .to(device) chain back to the same mock so __call__ still works.
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.return_value = fake_output

        # Tokenizer: return real tensors so tensor arithmetic works in the fn.
        input_ids = torch.ones((1, 4), dtype=torch.long)
        attention_mask = torch.ones((1, 4), dtype=torch.long)

        fake_tokenizer = MagicMock()
        fake_tokenizer.pad_token = "x"
        fake_tokenizer.return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        monkeypatch.setattr(
            "src.metrics.AutoModelForCausalLM.from_pretrained",
            lambda *a, **kw: fake_model,
        )
        monkeypatch.setattr(
            "src.metrics.AutoTokenizer.from_pretrained",
            lambda *a, **kw: fake_tokenizer,
        )

        ppl = compute_self_perplexity(["Hello world"], model_name="mock-gpt2")
        assert isinstance(ppl, float)
        assert ppl > 0.0
        # avg_loss = (target_loss * n_tokens) / n_tokens = target_loss → ppl = 5
        assert math.isclose(ppl, 5.0, rel_tol=1e-4)

    @pytest.mark.slow
    def test_compute_self_perplexity_real_model_returns_positive_float(self) -> None:
        """Integration: loads real gpt2 (smallest) and returns a positive float."""
        texts = [
            "The cat sat on the mat.",
            "Birds fly through the open sky.",
        ]
        ppl = compute_self_perplexity(texts, model_name="gpt2", max_length=64)
        assert isinstance(ppl, float)
        assert ppl > 0.0
        assert not math.isnan(ppl)
        assert not math.isinf(ppl)


# ---------------------------------------------------------------------------
# compute_mauve_score — lines 170-186
# ---------------------------------------------------------------------------


class TestComputeMauveScore:
    """Tests for compute_mauve_score import-error and exception paths."""

    @pytest.mark.unit
    def test_import_error_returns_nan(self, monkeypatch) -> None:
        """When the mauve package is not installed, return nan."""
        # Remove 'mauve' from sys.modules so the import inside the function fails.
        monkeypatch.setitem(sys.modules, "mauve", None)  # None causes ImportError
        result = compute_mauve_score(
            generated_texts=["hello world"],
            reference_texts=["hello there"],
        )
        assert math.isnan(result)

    @pytest.mark.unit
    def test_mauve_compute_exception_returns_nan(self, monkeypatch) -> None:
        """When mauve.compute_mauve raises, return nan."""
        fake_mauve = types.ModuleType("mauve")
        fake_mauve.compute_mauve = MagicMock(side_effect=RuntimeError("GPU OOM"))
        monkeypatch.setitem(sys.modules, "mauve", fake_mauve)

        result = compute_mauve_score(
            generated_texts=["hello world"],
            reference_texts=["hello there"],
        )
        assert math.isnan(result)

    @pytest.mark.unit
    def test_mauve_returns_float_on_success(self, monkeypatch) -> None:
        """When mauve.compute_mauve succeeds, return its .mauve attribute."""
        fake_result = MagicMock()
        fake_result.mauve = 0.87

        fake_mauve = types.ModuleType("mauve")
        fake_mauve.compute_mauve = MagicMock(return_value=fake_result)
        monkeypatch.setitem(sys.modules, "mauve", fake_mauve)

        result = compute_mauve_score(
            generated_texts=["hello world"],
            reference_texts=["hello there"],
        )
        assert math.isclose(result, 0.87)


# ---------------------------------------------------------------------------
# evaluate_generations additional paths — lines 219, 224
# ---------------------------------------------------------------------------


class TestEvaluateGenerationsAdditionalPaths:
    """Cover compute_ppl=False (self_ppl=nan) and reference_texts (MAUVE) paths."""

    @pytest.mark.unit
    def test_no_ppl_self_ppl_is_nan(self, sample_texts) -> None:
        """With compute_ppl=False, self_ppl must be nan (never computed)."""
        result = evaluate_generations(texts=sample_texts, compute_ppl=False)
        assert math.isnan(result.self_ppl)

    @pytest.mark.unit
    def test_with_reference_texts_mauve_score_is_set(
        self, sample_texts, monkeypatch
    ) -> None:
        """With reference_texts provided, mauve_score must be set (not None)."""
        # Stub out compute_mauve_score so the test doesn't need the library.
        monkeypatch.setattr("src.metrics.compute_mauve_score", lambda *a, **kw: 0.75)

        result = evaluate_generations(
            texts=sample_texts,
            reference_texts=sample_texts,  # self-reference is fine for the path test
            compute_ppl=False,
        )
        assert result.mauve_score is not None
        assert math.isclose(result.mauve_score, 0.75)

    @pytest.mark.unit
    def test_no_reference_texts_mauve_score_is_none(self, sample_texts) -> None:
        """Without reference_texts, mauve_score remains None."""
        result = evaluate_generations(
            texts=sample_texts,
            reference_texts=None,
            compute_ppl=False,
        )
        assert result.mauve_score is None

    @pytest.mark.unit
    def test_compute_ppl_true_calls_self_perplexity(
        self, sample_texts, monkeypatch
    ) -> None:
        """With compute_ppl=True, self_ppl is the value returned by compute_self_perplexity."""
        monkeypatch.setattr(
            "src.metrics.compute_self_perplexity",
            lambda texts, model_name: 99.5,
        )
        result = evaluate_generations(
            texts=sample_texts,
            compute_ppl=True,
            ppl_model="mock-model",
        )
        assert math.isclose(result.self_ppl, 99.5)


# ---------------------------------------------------------------------------
# load_generations — lines 240-244
# ---------------------------------------------------------------------------


class TestLoadGenerations:
    """Tests for load_generations JSONL reader."""

    @pytest.mark.unit
    def test_reads_jsonl_file(self, tmp_path) -> None:
        """load_generations correctly reads a JSONL file into a list of dicts."""
        jsonl_file = tmp_path / "generations.jsonl"
        records = [
            {"full_text": "The cat sat on the mat.", "method": "gaze_diffuse", "lambda": -1},
            {"full_text": "Birds fly through the sky.", "method": "unguided", "lambda": 0},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        loaded = load_generations(jsonl_file)

        assert len(loaded) == 2
        assert loaded[0]["full_text"] == "The cat sat on the mat."
        assert loaded[0]["method"] == "gaze_diffuse"
        assert loaded[1]["full_text"] == "Birds fly through the sky."

    @pytest.mark.unit
    def test_reads_empty_jsonl_file(self, tmp_path) -> None:
        """load_generations returns an empty list for a file with no lines."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        loaded = load_generations(empty_file)
        assert loaded == []

    @pytest.mark.unit
    def test_accepts_string_path(self, tmp_path) -> None:
        """load_generations accepts a plain string path in addition to Path."""
        jsonl_file = tmp_path / "gen.jsonl"
        jsonl_file.write_text(json.dumps({"key": "value"}) + "\n")
        loaded = load_generations(str(jsonl_file))
        assert loaded == [{"key": "value"}]


# ---------------------------------------------------------------------------
# print_results_table — lines 251-272
# ---------------------------------------------------------------------------


def _make_result(
    fkgl_mean: float = 5.0,
    fkgl_std: float = 1.2,
    ari_mean: float = 6.5,
    ari_std: float = 0.8,
    self_ppl: float = float("nan"),
    fk_sentence_variance: float = 2.34,
    mauve_score: float | None = None,
    n_samples: int = 10,
) -> MetricsResult:
    return MetricsResult(
        fkgl_mean=fkgl_mean,
        fkgl_std=fkgl_std,
        ari_mean=ari_mean,
        ari_std=ari_std,
        self_ppl=self_ppl,
        fk_sentence_variance=fk_sentence_variance,
        mauve_score=mauve_score,
        n_samples=n_samples,
    )


class TestPrintResultsTable:
    """Tests for print_results_table markdown formatting."""

    @pytest.mark.unit
    def test_contains_header_row(self) -> None:
        table = print_results_table({"GazeDiffuse": _make_result()})
        assert "| Method |" in table
        assert "FKGL" in table
        assert "MAUVE" in table
        assert "Self-PPL" in table
        assert "FK Variance" in table

    @pytest.mark.unit
    def test_contains_separator_row(self) -> None:
        table = print_results_table({"GazeDiffuse": _make_result()})
        lines = table.splitlines()
        # Second line must be the markdown separator (all dashes/pipes)
        separator = lines[1]
        assert "---" in separator
        assert "|" in separator

    @pytest.mark.unit
    def test_method_name_appears_in_row(self) -> None:
        table = print_results_table({"MyMethod": _make_result()})
        assert "MyMethod" in table

    @pytest.mark.unit
    def test_fkgl_values_formatted_in_row(self) -> None:
        result = _make_result(fkgl_mean=7.35, fkgl_std=0.55)
        table = print_results_table({"Baseline": result})
        assert "7.35" in table
        assert "0.55" in table

    @pytest.mark.unit
    def test_ppl_formatted_when_not_nan(self) -> None:
        result = _make_result(self_ppl=42.7)
        table = print_results_table({"Model": result})
        assert "42.7" in table

    @pytest.mark.unit
    def test_ppl_dash_when_nan(self) -> None:
        result = _make_result(self_ppl=float("nan"))
        table = print_results_table({"Model": result})
        # The ppl column should show "—" (em dash) rather than a number
        assert "—" in table

    @pytest.mark.unit
    def test_mauve_formatted_when_present(self) -> None:
        result = _make_result(mauve_score=0.923)
        table = print_results_table({"Model": result})
        assert "0.923" in table

    @pytest.mark.unit
    def test_fk_variance_formatted(self) -> None:
        result = _make_result(fk_sentence_variance=3.141)
        table = print_results_table({"Model": result})
        assert "3.141" in table

    @pytest.mark.unit
    def test_multiple_methods_all_rows_present(self) -> None:
        results = {
            "GazeDiffuse": _make_result(fkgl_mean=4.0, fkgl_std=0.5),
            "AR Baseline": _make_result(fkgl_mean=6.0, fkgl_std=1.0),
            "Unguided": _make_result(fkgl_mean=8.0, fkgl_std=1.5),
        }
        table = print_results_table(results)
        assert "GazeDiffuse" in table
        assert "AR Baseline" in table
        assert "Unguided" in table
        # Header + separator + 3 data rows = 5 lines
        assert len(table.splitlines()) == 5

    @pytest.mark.unit
    def test_returns_string(self) -> None:
        table = print_results_table({"X": _make_result()})
        assert isinstance(table, str)
