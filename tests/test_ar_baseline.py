"""Tests for AR gaze guidance baseline (src/ar_baseline.py)."""

from __future__ import annotations

import pytest

from src.ar_baseline import ARGazeConfig


class TestARGazeConfig:
    """Test AR baseline configuration."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        config = ARGazeConfig()
        assert config.model_name == "gpt2-medium"
        assert config.lam == -1.0
        assert config.top_k == 50
        assert config.max_new_tokens == 128

    @pytest.mark.unit
    def test_frozen(self) -> None:
        config = ARGazeConfig()
        with pytest.raises(AttributeError):
            config.lam = 0.5  # type: ignore[misc]

    @pytest.mark.unit
    def test_custom_values(self) -> None:
        config = ARGazeConfig(
            model_name="gpt2",
            lam=2.0,
            top_k=100,
            max_new_tokens=64,
        )
        assert config.model_name == "gpt2"
        assert config.lam == 2.0
