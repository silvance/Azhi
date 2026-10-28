import json
import os

import pytest

from sotto import thresholds
from sotto.settings import (
    SettingsError,
    load_settings,
    settings_path,
    update_settings,
)


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


def test_an_unreadable_file_is_moved_aside_with_its_contents_kept(config):
    """It may hold an administrator's thresholds that one bad byte hid."""
    config.parent.mkdir(parents=True, exist_ok=True)
    broken = '{"thresholds": {"comparison_high": 0.7}'
    config.write_text(broken, encoding="utf-8")
    saved = update_settings("transcribe", {"a": 1})
    assert saved.set_aside is not None
    assert saved.set_aside.read_text(encoding="utf-8") == broken
    assert saved.set_aside.name.startswith("settings.json.unreadable-")
    assert load_settings() == {"transcribe": {"a": 1}}


def test_one_unreadable_file_does_not_mean_a_warning_on_every_close(config):
    """QA-33: it used to be left in the way, so every close failed again."""
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{broken", encoding="utf-8")
    assert update_settings("transcribe", {"a": 1}).set_aside is not None
    again = update_settings("transcribe", {"a": 2})
    assert again.set_aside is None
    assert load_settings() == {"transcribe": {"a": 2}}


def test_an_empty_file_is_simply_empty(config):
    """What the old in-place writer left after a crash: nothing to rescue."""
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("", encoding="utf-8")
    saved = update_settings("transcribe", {"a": 1})
    assert saved.set_aside is None
    assert load_settings() == {"transcribe": {"a": 1}}


def test_a_failed_save_is_reported_not_swallowed(config, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("sotto.settings.write_json_atomic", refuse)
    with pytest.raises(SettingsError) as caught:
        update_settings("transcribe", {"a": 1})
    assert "could not be saved" in str(caught.value)
    assert "Access is denied" in str(caught.value)
    assert "Expecting value" not in str(caught.value), "no parser jargon"


def test_saving_is_atomic(config, monkeypatch):
    """A crash mid-write must leave the previous file, not half of a new one."""
    update_settings("transcribe", {"a": 1})
    real_replace = os.replace

    def crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("sotto.atomicio.os.replace", crash)
    with pytest.raises(SettingsError):
        update_settings("transcribe", {"a": 2})
    monkeypatch.setattr("sotto.atomicio.os.replace", real_replace)
    assert load_settings() == {"transcribe": {"a": 1}}
    leftovers = [p for p in config.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], "no half-written temporary file left behind"


# -- what earlier versions saved, before the program was renamed ---------------


@pytest.mark.skipif(os.name == "nt", reason="the folder names here are Linux's")
def test_an_earlier_versions_settings_folder_is_moved_not_copied(tmp_path, monkeypatch):
    old = tmp_path / "whispr"
    (old / "speakers").mkdir(parents=True)
    (old / "settings.json").write_text('{"transcribe": {"model": "medium"}}')
    (old / "speakers" / "subject.whispr-speaker.json").write_text("{}")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = settings_path()
    assert path == tmp_path / "sotto" / "settings.json"
    assert load_settings() == {"transcribe": {"model": "medium"}}
    assert (tmp_path / "sotto" / "speakers" / "subject.whispr-speaker.json").exists()
    # Voice profiles are in one place, never two.
    assert not old.exists()


@pytest.mark.skipif(os.name == "nt", reason="the folder names here are Linux's")
def test_a_folder_that_cannot_be_moved_is_used_where_it_is(tmp_path, monkeypatch):
    old = tmp_path / "whispr"
    old.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        type(old), "rename", lambda self, target: (_ for _ in ()).throw(OSError)
    )
    assert settings_path() == old / "settings.json"


@pytest.mark.skipif(os.name == "nt", reason="the folder names here are Linux's")
def test_the_whispers_folder_is_taken_over_the_older_whispr_one(tmp_path, monkeypatch):
    for old in ("whispers", "whispr"):
        (tmp_path / old).mkdir()
        (tmp_path / old / "settings.json").write_text(f'{{"from": "{old}"}}')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert settings_path() == tmp_path / "sotto" / "settings.json"
    assert load_settings() == {"from": "whispers"}
    assert not (tmp_path / "whispers").exists()
    assert (tmp_path / "whispr").exists()  # left alone, not merged


def test_the_windows_whispers_folder_becomes_sotto(tmp_path):
    from sotto.settings import OLD_WINDOWS_NAMES, _settings_folder

    (tmp_path / "Whispers" / "speakers").mkdir(parents=True)
    (tmp_path / "Whispers" / "settings.json").write_text("{}")
    folder = _settings_folder(tmp_path, "Sotto", OLD_WINDOWS_NAMES)
    assert folder == tmp_path / "Sotto"
    assert (folder / "speakers").is_dir() and (folder / "settings.json").is_file()
    assert not (tmp_path / "Whispers").exists()
    # Once moved, it is simply used.
    assert _settings_folder(tmp_path, "Sotto", OLD_WINDOWS_NAMES) == folder


def test_both_earlier_file_endings_are_brought_across(tmp_path):
    from sotto.settings import adopt_old_files

    (tmp_path / "a.whispers-speaker.json").write_text("from Whispers")
    (tmp_path / "b.whispr-speaker.json").write_text("from Whispr")
    found = adopt_old_files(
        tmp_path,
        ".sotto-speaker.json",
        (".whispers-speaker.json", ".whispr-speaker.json"),
    )
    assert [p.name for p in found] == ["a.sotto-speaker.json", "b.sotto-speaker.json"]
    assert (tmp_path / "a.sotto-speaker.json").read_text() == "from Whispers"


def test_old_file_endings_are_renamed_once_and_listed_once(tmp_path):
    from sotto.settings import adopt_old_files

    (tmp_path / "a.whispr-speaker.json").write_text("old a")
    (tmp_path / "b.sotto-speaker.json").write_text("new b")
    # Both names for c: the new one wins, the old one is left untouched.
    (tmp_path / "c.whispr-speaker.json").write_text("old c")
    (tmp_path / "c.sotto-speaker.json").write_text("new c")
    found = adopt_old_files(tmp_path, ".sotto-speaker.json", ".whispr-speaker.json")
    assert [p.name for p in found] == [
        "a.sotto-speaker.json",
        "b.sotto-speaker.json",
        "c.sotto-speaker.json",
    ]
    assert (tmp_path / "a.sotto-speaker.json").read_text() == "old a"
    assert (tmp_path / "c.whispr-speaker.json").read_text() == "old c"
    assert adopt_old_files(tmp_path / "missing", ".x", ".y") == []
