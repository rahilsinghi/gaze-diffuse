"""Shared test fixtures for GazeDiffuse test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data.geco import GazeExample
from src.gaze_predictor import GazePredictor, GazePredictorConfig


class MockMDLM(nn.Module):
    """Minimal mock of MDLM for testing without GPU or checkpoint.

    Returns random logits of the correct shape.
    """

    def __init__(self, vocab_size: int = 30522, mask_index: int = 103) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.dummy = nn.Linear(1, 1)  # So parameters() is non-empty

    def forward(self, x: torch.Tensor) -> "MockOutput":
        batch_size, seq_len = x.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return MockOutput(logits=logits)


class MockCausalLM(nn.Module):
    """Mock autoregressive LM matching GPT-2 interface for testing."""

    def __init__(self, vocab_size: int = 30522) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Linear(1, 1)  # So parameters() is non-empty

    def forward(self, input_ids: torch.Tensor) -> "MockOutput":
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return MockOutput(logits=logits)


class MockOutput:
    """Mock output with .logits attribute."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


@pytest.fixture
def mock_mdlm() -> MockMDLM:
    """Create a mock MDLM model."""
    return MockMDLM()


@pytest.fixture
def mock_causal_lm() -> MockCausalLM:
    """Create a mock causal LM for AR baseline testing."""
    return MockCausalLM()


@pytest.fixture
def gaze_predictor_config() -> GazePredictorConfig:
    """Minimal gaze predictor config for testing."""
    return GazePredictorConfig(
        bert_model="bert-base-uncased",
        hidden_size=768,
        dropout=0.1,
        learning_rate=2e-5,
        batch_size=4,
        epochs=1,
        max_seq_length=32,
    )


@pytest.fixture
def tiny_gaze_predictor_config() -> GazePredictorConfig:
    """High-LR config for fast convergence in smoke tests."""
    return GazePredictorConfig(
        bert_model="bert-base-uncased",
        hidden_size=768,
        dropout=0.1,
        learning_rate=5e-4,
        batch_size=4,
        epochs=2,
        max_seq_length=32,
    )


@pytest.fixture
def sample_prompt_ids() -> torch.Tensor:
    """Sample tokenized prompt for testing."""
    # "The discovery of a new" → approximate token IDs
    return torch.tensor([1996, 5765, 1997, 1037, 2047], dtype=torch.long)


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample generated texts for metrics testing."""
    return [
        "The cat sat on the mat. It was a sunny day. Birds were singing in the trees.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest. "
        "The species has unique wing patterns that help it camouflage.",
        "Education is important for personal growth. Students learn critical thinking "
        "skills that help them throughout life.",
        "The economy showed signs of recovery. Markets rallied on positive employment data. "
        "Investors remained cautiously optimistic about the future.",
    ]


def _make_synthetic_examples(n: int = 20, seed: int = 42) -> list[GazeExample]:
    """Create synthetic GazeExample instances for testing."""
    rng = np.random.RandomState(seed)
    participants = ["PP01", "PP02", "PP03", "PP04"]
    easy_words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "big", "red", "sun"]
    hard_words = ["epistemological", "photosynthesis", "juxtaposition", "phenomenological"]
    all_words = easy_words + hard_words

    examples = []
    for i in range(n):
        is_hard = i % 4 == 0
        word = rng.choice(hard_words if is_hard else easy_words)
        fix = rng.uniform(350, 500) if is_hard else rng.uniform(150, 250)
        n_left = rng.randint(1, 6)
        n_right = rng.randint(1, 6)
        left_ctx = [rng.choice(all_words) for _ in range(n_left)]
        right_ctx = [rng.choice(all_words) for _ in range(n_right)]
        examples.append(
            GazeExample(
                word=word,
                left_context=left_ctx,
                right_context=right_ctx,
                fixation_duration_ms=float(fix),
                sentence_id=i // 4,
                participant_id=participants[i % len(participants)],
            )
        )
    return examples


@pytest.fixture
def synthetic_gaze_examples() -> list[GazeExample]:
    """20 synthetic GazeExample instances with 4 participants."""
    return _make_synthetic_examples(20)


@pytest.fixture
def synthetic_gaze_dataframe() -> pd.DataFrame:
    """Synthetic DataFrame matching load_geco_corpus output format."""
    rng = np.random.RandomState(42)
    participants = ["PP01", "PP02", "PP03", "PP04"]
    sentences = [
        ["The", "cat", "sat", "on", "the", "mat"],
        ["Dogs", "run", "very", "fast", "today"],
        ["A", "big", "red", "sun", "rose", "slowly"],
        ["Students", "learn", "many", "things", "daily"],
        ["Birds", "sing", "beautiful", "songs", "outside"],
    ]
    rows = []
    for sent_id, words in enumerate(sentences):
        for part in participants:
            for pos, word in enumerate(words):
                rows.append({
                    "word": word,
                    "sentence_id": sent_id,
                    "word_position": pos,
                    "participant": part,
                    "mean_fixation_ms": float(rng.uniform(100, 500)),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def bert_tokenizer():
    """Session-scoped BERT tokenizer to avoid repeated downloads."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")
