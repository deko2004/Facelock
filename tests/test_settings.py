from __future__ import annotations

from pathlib import Path

import pytest

from config import settings


def test_clamp_threshold() -> None:
    assert settings.clamp_threshold(-1) == 0.0
    assert settings.clamp_threshold(0.363) == pytest.approx(0.363)
    assert settings.clamp_threshold(2) == 1.0


def test_save_and_load_threshold(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings.save_threshold(0.45, path)
    assert settings.load_threshold(path) == pytest.approx(0.45)


def test_invalid_settings_fall_back(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert settings.load_threshold(path) == pytest.approx(settings.DEFAULT_THRESHOLD)


def test_missing_settings_fall_back(tmp_path: Path) -> None:
    assert settings.load_threshold(tmp_path / "missing.json") == pytest.approx(
        settings.DEFAULT_THRESHOLD
    )
