import json
import os

import pytest

from whispr import thresholds
from whispr.settings import SettingsError, load_settings, settings_path, update_settings


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return settings_path()


def test_update_then_load_round_trips(config):
    update_settings("transcribe", {"model": "small", "diarize": True})
    assert load_settings() == {"transcribe": {"model": "small", "diarize": True}}


def test_load_missing_returns_empty(config):
    assert load_settings() == {}


def test_load_corrupt_returns_empty(config):
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{not json", encoding="utf-8")
    assert load_settings() == {}


def test_load_non_dict_returns_empty(config):
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_settings() == {}


# -- Closing the application must not erase an administrator's thresholds --
#
# QA found that closing wrote {"transcribe": ...} over the whole file. The
# comparison thresholds the Thresholds dialog tells administrators to set in
# this same file vanished, and the next session silently used the defaults.


def test_saving_one_section_keeps_the_others(config):
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({"thresholds": {"comparison_high": 0.7}, "transcribe": {"a": 1}}),
        encoding="utf-8",
    )
    update_settings("transcribe", {"a": 2})
    saved = load_settings()
    assert saved["thresholds"] == {"comparison_high": 0.7}
    assert saved["transcribe"] == {"a": 2}


def test_configured_thresholds_survive_a_close(config, monkeypatch):
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({"thresholds": {"comparison_high": 0.7}}), encoding="utf-8"
    )
    monkeypatch.setattr(thresholds, "_active", None, raising=False)
    before = thresholds.from_settings(load_settings()).comparison_high
    update_settings("transcribe", {"model": "base"})  # what closing does
    after = thresholds.from_settings(load_settings()).comparison_high
    assert before == after == pytest.approx(0.7)


def test_an_unreadable_file_is_left_alone_rather_than_replaced(config):
    """It may hold an administrator's thresholds that one bad byte hid."""
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('{"thresholds": {"comparison_high": 0.7}', encoding="utf-8")
    original = config.read_bytes()
    with pytest.raises(SettingsError) as caught:
        update_settings("transcribe", {"a": 1})
    assert config.read_bytes() == original
    assert "left as it is" in str(caught.value)


def test_a_failed_save_is_reported_not_swallowed(config, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("whispr.settings.write_json_atomic", refuse)
    with pytest.raises(SettingsError) as caught:
        update_settings("transcribe", {"a": 1})
    assert "could not be saved" in str(caught.value)
    assert "Access is denied" in str(caught.value)


def test_saving_is_atomic(config, monkeypatch):
    """A crash mid-write must leave the previous file, not half of a new one."""
    update_settings("transcribe", {"a": 1})
    real_replace = os.replace

    def crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("whispr.atomicio.os.replace", crash)
    with pytest.raises(SettingsError):
        update_settings("transcribe", {"a": 2})
    monkeypatch.setattr("whispr.atomicio.os.replace", real_replace)
    assert load_settings() == {"transcribe": {"a": 1}}
    leftovers = [p for p in config.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], "no half-written temporary file left behind"
