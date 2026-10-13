import pytest

import whispr.profiles as profiles
from whispr.profiles import (
    Profile,
    ProfileSaveError,
    delete_profile,
    existing_profile_name,
    export_profile,
    list_profiles,
    load_profile,
    profile_path,
    read_profile_file,
    save_profile,
)
from whispr.voiceprints import Voiceprint


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "profiles_dir", lambda: tmp_path / "profiles")


def test_save_load_round_trip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    prof = Profile(name="Op Nightfall", settings={"model": "small.en", "diarize": True})
    prof.voiceprint_for("Alpha").add([1.0, 0.0])
    prof.voiceprint_for("Bravo").add([0.0, 1.0])
    save_profile(prof)

    loaded = load_profile("Op Nightfall")
    assert loaded is not None
    assert loaded.name == "Op Nightfall"
    assert loaded.settings["model"] == "small.en"
    assert set(loaded.voiceprints) == {"Alpha", "Bravo"}
    assert loaded.voiceprints["Alpha"].vectors == [[1.0, 0.0]]


def test_list_profiles_sorted(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_profile(Profile(name="Zulu"))
    save_profile(Profile(name="Alpha"))
    assert list_profiles() == ["Alpha", "Zulu"]


def test_delete_profile(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_profile(Profile(name="Temp"))
    assert list_profiles() == ["Temp"]
    delete_profile("Temp")
    assert list_profiles() == []


def test_load_missing_profile_is_none(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert load_profile("nope") is None


def test_voiceprint_for_creates_and_reuses(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    prof = Profile(name="Op")
    first = prof.voiceprint_for("Alpha")
    again = prof.voiceprint_for("Alpha")
    assert first is again
    assert isinstance(first, Voiceprint)


def test_names_with_odd_characters_persist(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_profile(Profile(name="Op: Night/Fall #2"))
    # The display name is preserved even though the filename is slugged.
    assert list_profiles() == ["Op: Night/Fall #2"]
    assert load_profile("Op: Night/Fall #2") is not None


def test_export_then_read_round_trip(tmp_path):
    prof = Profile(name="Op Falcon", settings={"model": "medium.en"})
    prof.voiceprint_for("Alpha").add([1.0, 0.0])
    out = tmp_path / "falcon.whispr-profile.json"
    export_profile(prof, out)

    loaded = read_profile_file(out)
    assert loaded.name == "Op Falcon"
    assert loaded.settings["model"] == "medium.en"
    assert loaded.voiceprints["Alpha"].vectors == [[1.0, 0.0]]


def test_import_lands_in_profiles_dir(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    src = tmp_path / "shared.whispr-profile.json"
    export_profile(Profile(name="Shared", settings={"engine": "sherpa"}), src)
    # Simulate the GUI import: read the file, then save into this machine's dir.
    imported = read_profile_file(src)
    save_profile(imported)
    assert list_profiles() == ["Shared"]
    assert load_profile("Shared").settings["engine"] == "sherpa"


def test_read_profile_file_rejects_non_profile(tmp_path):
    bad = tmp_path / "notaprofile.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="isn't a Whispers profile"):
        read_profile_file(bad)


def test_read_profile_file_rejects_garbage(tmp_path):
    bad = tmp_path / "garbage.json"
    bad.write_text("not json at all", encoding="utf-8")
    with pytest.raises(ValueError):
        read_profile_file(bad)


# -- Operation profiles never disappear without the operator knowing -------
#
# QA found that creating "op alpha" while "Op Alpha" existed replaced it - its
# learned voice "Viktor", five vectors, became {} - because both names are kept
# in the same op_alpha.json. Delete asked nothing, and a failed save was
# swallowed while the page said "Saved".


def test_a_name_that_shares_a_file_is_reported_with_the_stored_name(
    tmp_path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    alpha = Profile(name="Op Alpha")
    alpha.voiceprint_for("Viktor").add([1.0, 0.0])
    save_profile(alpha)
    for other in ("op alpha", "OP ALPHA", "op_alpha", "Op Alpha!"):
        assert existing_profile_name(other) == "Op Alpha", other
    assert existing_profile_name("Op Bravo") is None


def test_an_unreadable_profile_file_still_counts_as_taken(tmp_path, monkeypatch):
    """Replacing it would destroy it as surely as replacing a readable one."""
    _isolate(tmp_path, monkeypatch)
    path = profile_path("Op Alpha")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    assert existing_profile_name("op alpha") == path.name


def test_a_failed_save_raises_instead_of_reporting_success(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("whispr.profiles.write_json_atomic", refuse)
    with pytest.raises(ProfileSaveError) as caught:
        save_profile(Profile(name="Op Alpha"))
    assert "could not be saved" in str(caught.value)
    assert "Access is denied" in str(caught.value)


def test_a_save_that_dies_part_way_leaves_the_old_profile(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    alpha = Profile(name="Op Alpha")
    alpha.voiceprint_for("Viktor").add([1.0, 0.0])
    save_profile(alpha)

    def crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("whispr.atomicio.os.replace", crash)
    with pytest.raises(ProfileSaveError):
        save_profile(Profile(name="Op Alpha"))
    monkeypatch.undo()
    _isolate(tmp_path, monkeypatch)
    kept = load_profile("Op Alpha")
    assert kept is not None and "Viktor" in kept.voiceprints


def test_deleting_a_missing_profile_is_not_an_error(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    delete_profile("never existed")


def test_a_delete_that_fails_says_so(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_profile(Profile(name="Op Alpha"))

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("pathlib.Path.unlink", refuse)
    with pytest.raises(ProfileSaveError) as caught:
        delete_profile("Op Alpha")
    assert "could not be deleted" in str(caught.value)
