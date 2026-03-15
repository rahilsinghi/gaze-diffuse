"""Validation tests for real GECO eye-tracking data.

All tests auto-skip if GECO data is not present locally.
Run after: bash scripts/download_data.sh
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.geco import (
    GazeDataConfig,
    GazeDataset,
    create_subject_cv_splits,
    extract_gaze_examples,
    load_geco_corpus,
)

GECO_DIR = Path("data/geco")
GECO_AVAILABLE = (
    (GECO_DIR / "MonolingualReadingData.xlsx").exists()
    or (GECO_DIR / "geco_processed.csv").exists()
)

pytestmark = pytest.mark.skipif(not GECO_AVAILABLE, reason="GECO data not available")


@pytest.fixture(scope="module")
def geco_df():
    """Load GECO corpus once per test module."""
    return load_geco_corpus(str(GECO_DIR))


@pytest.fixture(scope="module")
def geco_examples(geco_df):
    """Extract gaze examples from real GECO data."""
    config = GazeDataConfig(context_window=5, min_fixation_ms=0.0, max_fixation_ms=2000.0)
    return extract_gaze_examples(geco_df, config)


class TestGecoCsvLoad:

    def test_load_geco_corpus_columns(self, geco_df):
        """GECO loads with expected columns and substantial row count."""
        expected = {"word", "sentence_id", "word_position", "participant", "mean_fixation_ms"}
        assert expected == set(geco_df.columns)
        assert geco_df["mean_fixation_ms"].notna().all()
        assert len(geco_df) > 10000
        assert geco_df["sentence_id"].dtype in ("int64", "int32")

    def test_geco_fixation_ranges(self, geco_df):
        """Fixation durations are in physiologically plausible range."""
        fix = geco_df["mean_fixation_ms"]
        assert fix.min() >= 0, "Negative fixation duration"
        assert fix.max() < 10000, "Absurdly high fixation duration"
        median = fix.median()
        assert 50 < median < 1000, f"Unusual median fixation: {median}"


class TestGecoExamples:

    def test_extract_examples_from_real_geco(self, geco_examples):
        """extract_gaze_examples produces valid examples from real data."""
        assert len(geco_examples) > 5000
        for ex in geco_examples[:100]:
            assert len(ex.left_context) <= 5
            assert len(ex.right_context) <= 5
            assert 0.0 <= ex.fixation_duration_ms <= 2000.0

        sentence_ids = {ex.sentence_id for ex in geco_examples}
        assert len(sentence_ids) > 10
        participants = {ex.participant_id for ex in geco_examples}
        assert len(participants) > 1


class TestGecoCV:

    def test_cv_splits_real_data(self, geco_examples):
        """5-fold CV splits have no participant leakage on real data."""
        splits = list(create_subject_cv_splits(geco_examples, n_folds=5))

        assert len(splits) == 5
        all_test_parts = set()

        for train, test in splits:
            train_parts = {ex.participant_id for ex in train}
            test_parts = {ex.participant_id for ex in test}
            assert train_parts & test_parts == set()
            assert len(train) + len(test) == len(geco_examples)
            all_test_parts |= test_parts

        all_parts = {ex.participant_id for ex in geco_examples}
        assert all_test_parts == all_parts


class TestGecoDataset:

    def test_gaze_dataset_from_real_data(self, geco_examples, bert_tokenizer):
        """Tokenization works on real GECO words (punctuation, contractions, etc.)."""
        subset = geco_examples[:100]
        dataset = GazeDataset(subset, bert_tokenizer, max_length=64, normalize=True)

        for i in range(min(100, len(dataset))):
            item = dataset[i]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert not item["labels"].isnan()
            # At least CLS + 1 content + SEP
            n_tokens = item["attention_mask"].sum().item()
            assert n_tokens >= 3

        mean, std = dataset.get_normalization_stats()
        assert mean > 0
        assert std > 0
