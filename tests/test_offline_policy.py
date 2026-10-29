"""A shipped copy never reaches for the network, by any route.

QA found three routes that did. With no model bundled (a damaged or partial
copy) the Quality line promised "will be downloaded on first use" and the run
handed the name to faster-whisper, which downloads it. The Live page passed any
typed model name the same way. And "auto" speaker separation used pyannote
whenever the library was installed, so a copy without the pyannote cache went
to the Hugging Face Hub. All three now go through one policy, and in a bundle
the Hugging Face libraries are told to stay offline before they are imported.
"""

import sys
import types

import pytest

from azhi import diagnostics, diarization, resources, transcription
from azhi.transcription import ModelUnavailable, resolve_model_choice

_OFFLINE = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
_ARGOS = ("ARGOS_MODEL_PROVIDER", "ARGOS_CHUNK_TYPE")


@pytest.fixture(autouse=True)
def _contained(monkeypatch):
    """enforce_offline changes process-wide state; keep it inside each test."""
    import urllib.request

    monkeypatch.setattr(urllib.request, "_opener", urllib.request._opener)
    for flag in _OFFLINE + _ARGOS:
        monkeypatch.delenv(flag, raising=False)


@pytest.fixture
def bundle(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)


@pytest.fixture
def source(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)


def _model_folder(tmp_path, name="model"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"\0")
    return folder


# -- Which model a run may use --------------------------------------------


def test_a_bundled_name_resolves_to_its_folder(bundle, tmp_path):
    folder = _model_folder(tmp_path)
    assert resolve_model_choice("medium", {"medium": folder}) == str(folder)


def test_a_folder_holding_a_model_is_used_as_it_is(bundle, tmp_path):
    folder = _model_folder(tmp_path)
    assert resolve_model_choice(str(folder), {}) == str(folder)


def test_a_folder_with_no_model_in_it_is_refused(source, tmp_path):
    with pytest.raises(ModelUnavailable) as caught:
        resolve_model_choice(str(tmp_path), {})
    assert "model.bin" in str(caught.value)


def test_a_shipped_copy_with_no_models_refuses_rather_than_downloads(bundle):
    """The case QA reproduced: nothing bundled, and the run went to fetch it."""
    with pytest.raises(ModelUnavailable) as caught:
        resolve_model_choice("medium", {})
    message = str(caught.value)
    assert "This copy has no" in message and "medium" in message
    assert "download" not in message.lower()
    assert "internet" not in message.lower()


def test_a_name_the_bundle_lacks_is_refused(source, tmp_path):
    with pytest.raises(ModelUnavailable):
        resolve_model_choice("small.en", {"base": _model_folder(tmp_path)})


def test_from_source_with_nothing_bundled_a_size_name_passes_through(source):
    """The developer convenience, which a shipped copy never reaches."""
    assert resolve_model_choice("base", {}) == "base"


def test_no_model_chosen_is_refused(source):
    with pytest.raises(ModelUnavailable):
        resolve_model_choice("  ", {})


def test_a_refusal_reaches_the_operator_as_written():
    friendly_error = pytest.importorskip("azhi.gui.errors").friendly_error
    message = "This copy has no ‘medium’ model. System status lists them."
    assert friendly_error(ModelUnavailable(message)) == message


# -- The backstop under the gate ------------------------------------------


@pytest.mark.parametrize("frozen,expected", [(True, True), (False, False)])
def test_the_model_loader_forbids_downloading_in_a_bundle(
    monkeypatch, frozen, expected
):
    calls = {}

    class FakeWhisperModel:
        def __init__(self, name, **kwargs):
            calls.update(kwargs, name=name)

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setattr(transcription, "_MODEL_CACHE", {})
    if frozen:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
    else:
        monkeypatch.delattr(sys, "frozen", raising=False)
    transcription._load_model("base", "cpu", "int8")
    assert calls["local_files_only"] is expected


# -- Hugging Face told to stay offline -------------------------------------


def test_a_bundle_switches_every_hugging_face_library_offline(bundle, monkeypatch):
    for flag in _OFFLINE:
        monkeypatch.delenv(flag, raising=False)
    assert resources.enforce_offline() is True
    for flag in _OFFLINE:
        assert __import__("os").environ[flag] == "1"


def test_running_from_source_leaves_them_alone(source, monkeypatch):
    for flag in _OFFLINE:
        monkeypatch.delenv(flag, raising=False)
    assert resources.enforce_offline() is False
    for flag in _OFFLINE:
        assert flag not in __import__("os").environ


def test_a_bundle_missing_the_pyannote_cache_is_still_offline(bundle, monkeypatch):
    """The early return used to skip setting HF_HUB_OFFLINE exactly here."""
    for flag in _OFFLINE:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: None)
    assert resources.configure_offline_hf_cache() is None
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"


# -- Speaker separation never goes to the Hub from a bundle ----------------


def test_auto_does_not_choose_pyannote_without_its_cache_in_a_bundle(
    bundle, monkeypatch
):
    monkeypatch.setattr(diarization, "_pyannote_available", lambda: True)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: None)
    assert diarization._pyannote_usable_offline() is False


def test_from_source_pyannote_may_still_use_the_hub(source, monkeypatch):
    monkeypatch.setattr(diarization, "_pyannote_available", lambda: True)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: None)
    assert diarization._pyannote_usable_offline() is True


def test_asking_for_pyannote_in_a_bundle_without_it_says_so(
    bundle, monkeypatch, tmp_path
):
    monkeypatch.setattr(diarization, "_pyannote_available", lambda: True)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: None)
    with pytest.raises(RuntimeError) as caught:
        diarization.diarize(tmp_path / "x.wav", backend="pyannote")
    message = str(caught.value)
    assert "does not include the pyannote" in message
    assert "internet" not in message.lower()


# -- System status says what is true for this copy -------------------------


def test_system_status_states_the_no_download_promise_only_where_it_is_kept(
    monkeypatch,
):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    shipped = diagnostics.format_report(checks=[])
    monkeypatch.delattr(sys, "frozen", raising=False)
    from_source = diagnostics.format_report(checks=[])
    assert "refuses any model it does not already hold" in shipped
    assert "downloaded the first time it is used" in from_source
    assert "no model or software downloads" not in from_source


# -- Argos Translate cannot reach the network from a bundle either ----------
#
# QA traced it, and the pinned source confirms it: MiniSBD downloads its
# sentence-splitter model from GitHub with urllib.request.urlretrieve whenever
# the one it wants is not cached, and none of the Hugging Face switches touch
# that. Argos also reads its model provider from a settings.json under the
# user's config folder, where LIBRETRANSLATE or OPENAI would send the text out.


def test_a_bundle_pins_argos_to_its_local_engine(bundle):
    resources.enforce_offline()
    import os

    assert os.environ["ARGOS_MODEL_PROVIDER"] == "OPENNMT"
    assert os.environ["ARGOS_CHUNK_TYPE"] == "MINISBD"


def test_a_bundle_refuses_every_download_plainly(bundle, tmp_path):
    import urllib.request

    resources.enforce_offline()
    url = "https://github.com/LibreTranslate/MiniSBD/releases/download/v0.0.1/en.onnx"
    with pytest.raises(resources.NetworkRefused) as caught:
        urllib.request.urlretrieve(url, str(tmp_path / "en.onnx"))
    assert "never uses the network" in str(caught.value)
    assert not (tmp_path / "en.onnx").exists()


def test_a_bundle_can_still_read_local_files(bundle, tmp_path):
    import urllib.request

    resources.enforce_offline()
    local = tmp_path / "note.txt"
    local.write_text("local", encoding="utf-8")
    assert urllib.request.urlopen(local.as_uri()).read() == b"local"


def test_running_from_source_does_not_refuse_downloads(source):
    import urllib.request

    before = urllib.request._opener
    resources.enforce_offline()
    assert urllib.request._opener is before


def test_the_refusal_reaches_the_operator_as_written():
    friendly_error = pytest.importorskip("azhi.gui.errors").friendly_error
    message = "This copy tried to download https://x and was stopped."
    assert friendly_error(resources.NetworkRefused(message)) == message


def test_system_status_lists_the_offline_sentence_splitters(tmp_path, monkeypatch):
    data = tmp_path / "argos" / "argos-translate"
    (data / "minisbd").mkdir(parents=True)
    (data / "minisbd" / "de.onnx").write_bytes(b"x")
    pack = data / "packages" / "translate-ru_en" / "minisbd"
    pack.mkdir(parents=True)
    (pack / "ru.onnx").write_bytes(b"x")
    monkeypatch.setattr(resources, "bundled_argos_data_dir", lambda: data)
    assert resources.bundled_sentence_splitters() == ["de", "ru"]


def test_packs_without_any_splitter_are_flagged(tmp_path, monkeypatch):
    """What a build whose warm-up failed used to ship, saying "bundled"."""
    data = tmp_path / "argos" / "argos-translate"
    (data / "packages" / "translate-de_en").mkdir(parents=True)
    monkeypatch.setattr(resources, "bundled_argos_data_dir", lambda: data)
    checks = {c.label: c for c in diagnostics.gather()}
    splitter = checks["Translation: offline sentence splitters"]
    assert splitter.ok is False
    assert "refuses" in splitter.detail
