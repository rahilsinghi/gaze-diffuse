"""Shared test fixtures for GazeDiffuse test suite."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

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


class MockOutput:
    """Mock output with .logits attribute."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


@pytest.fixture
def mock_mdlm() -> MockMDLM:
    """Create a mock MDLM model."""
    return MockMDLM()


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
