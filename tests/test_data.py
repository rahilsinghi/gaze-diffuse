"""Tests for data loading modules."""

from __future__ import annotations

import pytest

from src.data.prompts import PROMPT_SEEDS, get_prompts


class TestPrompts:
    """Test prompt seed data."""

    @pytest.mark.unit
    def test_has_50_prompts(self) -> None:
        assert len(PROMPT_SEEDS) == 50

    @pytest.mark.unit
    def test_all_non_empty(self) -> None:
        for prompt in PROMPT_SEEDS:
            assert len(prompt.strip()) > 0

    @pytest.mark.unit
    def test_get_prompts_all(self) -> None:
        prompts = get_prompts()
        assert len(prompts) == 50

    @pytest.mark.unit
    def test_get_prompts_subset(self) -> None:
        prompts = get_prompts(10)
        assert len(prompts) == 10

    @pytest.mark.unit
    def test_prompts_are_copies(self) -> None:
        """get_prompts should return a copy, not the original list."""
        prompts = get_prompts()
        prompts.append("extra")
        assert len(PROMPT_SEEDS) == 50

    @pytest.mark.unit
    def test_no_duplicates(self) -> None:
        assert len(set(PROMPT_SEEDS)) == len(PROMPT_SEEDS)


class TestGazeExample:
    """Test GazeExample dataclass."""

    @pytest.mark.unit
    def test_creation(self) -> None:
        from src.data.geco import GazeExample

        ex = GazeExample(
            word="hello",
            left_context=["say"],
            right_context=["world"],
            fixation_duration_ms=200.0,
            sentence_id=1,
        )
        assert ex.word == "hello"
        assert ex.fixation_duration_ms == 200.0
        assert ex.participant_id is None

    @pytest.mark.unit
    def test_frozen(self) -> None:
        from src.data.geco import GazeExample

        ex = GazeExample(
            word="test",
            left_context=[],
            right_context=[],
            fixation_duration_ms=100.0,
            sentence_id=0,
        )
        with pytest.raises(AttributeError):
            ex.word = "changed"  # type: ignore[misc]


class TestGazeDataConfig:
    """Test GazeDataConfig defaults."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        from src.data.geco import GazeDataConfig

        config = GazeDataConfig()
        assert config.context_window == 5
        assert config.max_seq_length == 64
        assert config.normalize_durations is True
