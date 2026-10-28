"""The analysis record names the speaker split that actually ran.

QA found the record storing the *setting* ("auto") as the engine, and filling
the same model fields whichever engine ran: the sherpa segmentation file, and as
the embedding model the Speaker Profiles model - which neither engine uses to
separate speakers. So a pyannote run was described by sherpa's files, and even
a sherpa run named the wrong embedding model. It also printed the shipped
thresholds while configured ones were in force. Each is a fabricated fact in a
document whose job is to say what produced a result.
"""

import json

import pytest

from sotto import diarization, hub_cache, resources, thresholds
from sotto.provenance import AnalysisProvenance, DiarizationProvenance


@pytest.fixture
def sherpa_files(tmp_path, monkeypatch):
    # As a real bundle lays them out: one diarization/embedding.onnx serves
    # both sherpa's speaker split and Speaker Profiles, with its name in
    # embedding_model.txt beside it.
    seg = tmp_path / "segmentation.onnx"
    emb = tmp_path / "embedding.onnx"
    seg.write_bytes(b"segmentation")
    emb.write_bytes(b"embedding")
    monkeypatch.setattr(resources, "bundled_diarization_models", lambda: (seg, emb))
    monkeypatch.setattr(resources, "bundled_embedding_model", lambda: emb)
    monkeypatch.setattr(
        resources, "bundled_embedding_model_name", lambda: "3D-Speaker ERes2Net"
    )
    return seg, emb


@pytest.fixture
def pyannote_cache(tmp_path, monkeypatch):
    cache = tmp_path / "pyannote"
    hub = cache / "hub"
    for index, repo in enumerate(hub_cache.PYANNOTE_REPOS):
        commit = f"{index}" * 40
        (hub_cache.repo_folder(hub, repo) / "snapshots" / commit).mkdir(parents=True)
        hub_cache.write_ref(hub, repo, commit)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: cache)
    return hub


def test_auto_is_resolved_to_the_engine_that_runs(monkeypatch):
    monkeypatch.setattr(diarization, "_pyannote_usable_offline", lambda: False)
    assert diarization.chosen_backend("auto") == "sherpa"
    monkeypatch.setattr(diarization, "_pyannote_usable_offline", lambda: True)
    assert diarization.chosen_backend("auto") == "pyannote"
    assert diarization.chosen_backend("SHERPA") == "sherpa"


def test_the_record_never_says_auto(monkeypatch, sherpa_files):
    monkeypatch.setattr(diarization, "_pyannote_usable_offline", lambda: False)
    record = DiarizationProvenance.from_bundle(engine="auto")
    assert record.engine == "sherpa"


def test_a_sherpa_run_names_its_models_by_what_they_are(monkeypatch, sherpa_files):
    """The readable name the build recorded, and the hash of the file used."""
    from sotto.hashing import sha256_file

    monkeypatch.setattr(diarization, "_pyannote_usable_offline", lambda: False)
    seg, emb = sherpa_files
    record = DiarizationProvenance.from_bundle(engine="sherpa")
    assert record.segmentation_model == seg.name
    assert record.segmentation_model_sha256 == sha256_file(seg)
    assert record.embedding_model == "3D-Speaker ERes2Net"
    assert record.embedding_model_sha256 == sha256_file(emb)
    assert record.pipeline == ""
    lines = "\n".join(AnalysisProvenance(diarization=record).describe())
    assert f"Segmentation SHA-256: {sha256_file(seg)}" in lines


def test_a_pyannote_run_is_named_by_its_own_repositories(
    monkeypatch, sherpa_files, pyannote_cache
):
    monkeypatch.setattr(diarization, "_pyannote_usable_offline", lambda: True)
    record = DiarizationProvenance.from_bundle(engine="auto")
    assert record.engine == "pyannote"
    assert record.pipeline == "pyannote/speaker-diarization-3.1@" + "0" * 40
    assert record.segmentation_model == "pyannote/segmentation-3.0@" + "1" * 40
    assert record.embedding_model.startswith("pyannote/wespeaker-voxceleb-resnet34-LM@")
    for value in (record.segmentation_model, record.embedding_model):
        assert "onnx" not in value, "never sherpa's files for a pyannote run"


def test_without_a_cache_the_revision_is_said_to_be_unknown(monkeypatch):
    monkeypatch.setattr(diarization, "_pyannote_usable_offline", lambda: True)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: None)
    record = DiarizationProvenance.from_bundle(engine="pyannote")
    assert record.segmentation_model.endswith("(revision not recorded)")


def test_names_from_voice_similarity_are_recorded_as_leads(monkeypatch, sherpa_files):
    monkeypatch.setattr(diarization, "_pyannote_usable_offline", lambda: False)
    record = DiarizationProvenance.from_bundle(
        engine="sherpa", voice_leads=["Viktor", "Anna", "Viktor", ""]
    )
    assert record.voice_leads == ["Anna", "Viktor"]
    again = DiarizationProvenance.from_dict(json.loads(json.dumps(record.to_dict())))
    assert again.voice_leads == ["Anna", "Viktor"]
    lines = "\n".join(AnalysisProvenance(diarization=record).describe())
    assert "leads, not identifications" in lines and "Viktor" in lines


def test_the_record_holds_the_thresholds_in_force(monkeypatch):
    configured = thresholds.Thresholds(recognition_acceptance=0.75)
    monkeypatch.setattr(thresholds, "_active", configured)
    record = AnalysisProvenance()
    assert record.thresholds.recognition_acceptance == pytest.approx(0.75)
    assert "Speaker acceptance threshold: 0.75" in record.describe()


def test_a_record_with_no_creation_time_is_not_given_one():
    """Reading a record must not stamp it with the time it was read."""
    record = AnalysisProvenance.from_dict({"source": {}})
    assert record.created_utc == ""
    assert "Analysed (UTC): not recorded" in record.describe()
