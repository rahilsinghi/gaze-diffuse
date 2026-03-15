"""GECO eye-tracking corpus loader for gaze predictor training.

The GECO corpus contains fixation data from 14 participants reading
an English novel (5,031 sentences). Each word has mean fixation
duration annotations from eye-tracking.

Download from: https://expsy.ugent.be/geco/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Context window: 5 tokens left + target + 5 tokens right (Sauberli et al.)
CONTEXT_WINDOW = 5


@dataclass(frozen=True)
class GazeExample:
    """A single gaze prediction training example."""

    word: str
    left_context: list[str]
    right_context: list[str]
    fixation_duration_ms: float
    sentence_id: int
    participant_id: str | None = None


@dataclass(frozen=True)
class GazeDataConfig:
    """Configuration for GECO data loading."""

    data_dir: str = "data/geco"
    context_window: int = CONTEXT_WINDOW
    max_seq_length: int = 64
    min_fixation_ms: float = 0.0
    max_fixation_ms: float = 2000.0
    test_participants: list[str] = field(default_factory=list)
    normalize_durations: bool = True


def load_geco_corpus(data_dir: str | Path) -> pd.DataFrame:
    """Load the GECO eye-tracking corpus from the data directory.

    Expects either:
    - MonolingualReadingData.xlsx (original GECO format)
    - geco_processed.csv (pre-processed CSV)

    Returns DataFrame with columns: word, sentence_id, participant,
    mean_fixation_ms, word_position.
    """
    data_dir = Path(data_dir)

    csv_path = data_dir / "geco_processed.csv"
    if csv_path.exists():
        logger.info("Loading pre-processed GECO from %s", csv_path)
        return pd.read_csv(csv_path)

    xlsx_path = data_dir / "MonolingualReadingData.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"GECO data not found at {data_dir}. "
            "Download from https://expsy.ugent.be/geco/ and place "
            "MonolingualReadingData.xlsx in the data directory."
        )

    logger.info("Loading raw GECO from %s (this may take a minute)", xlsx_path)
    df = pd.read_excel(xlsx_path)

    # Standard GECO columns — adapt if format differs
    required_cols = {"WORD", "WORD_ID", "PP_NR"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"GECO file missing expected columns. Found: {list(df.columns)}"
        )

    # Extract sentence/trial ID and word position from WORD_ID.
    # GECO format is "PART-TRIAL-WORDPOS" (e.g., "1-5-3").
    # Use PART-TRIAL as the sentence identifier.
    parts = df["WORD_ID"].str.extract(r"(\d+)-(\d+)-(\d+)")
    # Drop rows where WORD_ID didn't match the expected format
    valid_mask = parts[0].notna()
    df = df[valid_mask].copy()
    parts = parts[valid_mask]
    df["sentence_id"] = parts[0].astype(int) * 10000 + parts[1].astype(int)
    df["word_position"] = parts[2].astype(int)

    # Use WORD_TOTAL_READING_TIME or WORD_FIXATION_DURATION if available
    fixation_col = None
    for candidate in [
        "WORD_TOTAL_READING_TIME",
        "WORD_FIXATION_DURATION",
        "WORD_FIRST_FIXATION_DURATION",
    ]:
        if candidate in df.columns:
            fixation_col = candidate
            break

    if fixation_col is None:
        raise ValueError("No fixation duration column found in GECO data")

    # GECO uses "." for missing fixation values — coerce to numeric NaN
    df[fixation_col] = pd.to_numeric(df[fixation_col], errors="coerce")

    processed = df.rename(
        columns={
            "WORD": "word",
            "PP_NR": "participant",
            fixation_col: "mean_fixation_ms",
        }
    )[["word", "sentence_id", "word_position", "participant", "mean_fixation_ms"]]

    # Drop NaN fixations (skipped words)
    processed = processed.dropna(subset=["mean_fixation_ms"])

    # Save processed version for faster loading next time
    processed.to_csv(csv_path, index=False)
    logger.info("Saved processed GECO to %s (%d rows)", csv_path, len(processed))

    return processed


def extract_gaze_examples(
    df: pd.DataFrame,
    config: GazeDataConfig,
) -> list[GazeExample]:
    """Convert GECO DataFrame into contextualized gaze examples.

    For each word, extracts a context window of surrounding words
    (Sauberli et al. format: 5 left + target + 5 right).
    """
    examples: list[GazeExample] = []

    for sent_id, sent_df in df.groupby("sentence_id"):
        sent_df = sent_df.sort_values("word_position")
        words = sent_df["word"].tolist()
        fixations = sent_df["mean_fixation_ms"].tolist()
        participants = (
            sent_df["participant"].tolist()
            if "participant" in sent_df.columns
            else [None] * len(words)
        )

        for i, (word, fix, part) in enumerate(zip(words, fixations, participants)):
            if not (config.min_fixation_ms <= fix <= config.max_fixation_ms):
                continue

            left_start = max(0, i - config.context_window)
            right_end = min(len(words), i + config.context_window + 1)

            left_ctx = words[left_start:i]
            right_ctx = words[i + 1 : right_end]

            examples.append(
                GazeExample(
                    word=word,
                    left_context=left_ctx,
                    right_context=right_ctx,
                    fixation_duration_ms=fix,
                    sentence_id=int(sent_id),
                    participant_id=part,
                )
            )

    return examples


class GazeDataset(Dataset):
    """PyTorch Dataset for gaze predictor training.

    Tokenizes context windows and returns (input_ids, attention_mask, target).
    Format: [CLS] left_context [SEP] target_word [SEP] right_context [SEP]
    """

    def __init__(
        self,
        examples: list[GazeExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 64,
        normalize: bool = True,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Compute normalization stats
        durations = np.array([ex.fixation_duration_ms for ex in examples])
        self.mean_duration = float(durations.mean())
        self.std_duration = float(durations.std())
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]

        left_text = " ".join(ex.left_context)
        right_text = " ".join(ex.right_context)

        # Sauberli format: [CLS] left [SEP] target [SEP] right [SEP]
        encoding = self.tokenizer(
            text=f"{left_text} [SEP] {ex.word}",
            text_pair=right_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = ex.fixation_duration_ms
        if self.normalize:
            target = (target - self.mean_duration) / (self.std_duration + 1e-8)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(target, dtype=torch.float32),
        }

    def get_normalization_stats(self) -> tuple[float, float]:
        """Return (mean, std) for denormalizing predictions."""
        return self.mean_duration, self.std_duration


def create_subject_cv_splits(
    examples: list[GazeExample],
    n_folds: int = 5,
) -> Iterator[tuple[list[GazeExample], list[GazeExample]]]:
    """Create subject-level cross-validation splits.

    Splits by participant ID so that the model is evaluated on
    unseen readers (not just unseen words).
    """
    participants = sorted(
        {ex.participant_id for ex in examples if ex.participant_id is not None}
    )

    if len(participants) < n_folds:
        raise ValueError(
            f"Only {len(participants)} participants, need >= {n_folds} for CV"
        )

    fold_size = len(participants) // n_folds
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < n_folds - 1 else len(participants)
        test_participants = set(participants[start:end])

        train = [ex for ex in examples if ex.participant_id not in test_participants]
        test = [ex for ex in examples if ex.participant_id in test_participants]

        yield train, test
