"""Nemotron speaker separation: running NVIDIA's program, offline, and reading it.

The real program and model are not in the test environment. A stand-in script
takes their place: it is invoked exactly as the real one would be, records how
it was invoked, and answers the way the real one does (its JSON output format
is copied from NeMo-Speech.cpp 0.2.0, app/diarize.cpp).
"""

import json
import os
import sys
import time

import pytest

from whispr import diarization, nemotron, resources
from whispr.provenance import AnalysisProvenance, DiarizationProvenance
from whispr.transcription import CancelledError

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the stand-in program is a script with a #! line"
)

FAKE_PROGRAM = """#!{python}
import json, os, sys, time
args = sys.argv[1:]
log = os.environ["FAKE_LOG"]
mode = os.environ.get("FAKE_MODE", "ok")
value = lambda flag: args[args.index(flag) + 1]
wav, model = args[args.index("diarize") + 1], value("--model")
with open(log, "w") as handle:
    json.dump({{
        "args": args,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "wav_readable": os.path.isfile(wav),
        "model_readable": os.path.isfile(model),
        "hf_base_url": os.environ.get("NEMO_SPEECH_HF_BASE_URL"),
        "model_dir": os.environ.get("NEMO_SPEECH_MODEL_DIR"),
    }}, handle)
if mode == "slow":
    time.sleep(60)
if mode == "fail":
    sys.stderr.write("[model] loading\\nerror: diarize: cannot open audio file\\n")
    sys.exit(1)
with open(value("--output"), "w") as out:
    if mode == "garbage":
        out.write("not json")
    else:
        out.write(json.dumps({{"file": wav, "segments": json.loads(os.environ["FAKE_SEGMENTS"])}}))
"""

TWO_SPEAKERS = [
    {"start": 2.0, "end": 2.6, "speaker": 2},
    {"start": 0.0, "end": 2.1, "speaker": 1},
]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A Nemotron asset folder holding the stand-in program and a fake model."""
    assets = tmp_path / "whispr_assets"
    folder = assets / "nemotron"
    (folder / "bin").mkdir(parents=True)
    program = folder / "bin" / "nemo-speech"
    program.write_text(FAKE_PROGRAM.format(python=sys.executable), encoding="utf-8")
    program.chmod(0o755)
    model = folder / resources.NEMOTRON_MODEL_FILENAME
    model.write_bytes(b"gguf-weights")
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    log = tmp_path / "invocation.json"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_SEGMENTS", json.dumps(TWO_SPEAKERS))
    wav = tmp_path / "call.wav"
    wav.write_bytes(b"RIFF")
    return {
        "folder": folder,
        "program": program,
        "model": model,
        "log": log,
        "wav": wav,
    }


def _invocation(bundle):
    return json.loads(bundle["log"].read_text())


def test_turns_come_back_numbered_from_zero_in_time_order(bundle):
    turns = diarization.diarize(bundle["wav"], backend="nemotron")
    assert [(t.start, t.end, t.speaker) for t in turns] == [
        (0.0, 2.1, "SPEAKER_00"),
        (2.0, 2.6, "SPEAKER_01"),
    ]


def test_program_gets_the_bundled_model_and_no_way_to_download(bundle):
    diarization.diarize(bundle["wav"], backend="nemotron")
    seen = _invocation(bundle)
    args = seen["args"]
    # A path to the file, never a repository name the program would fetch.
    assert args[args.index("--model") + 1] == str(bundle["model"])
    assert seen["model_readable"] and seen["wav_readable"]
    assert seen["hf_base_url"] == "http://127.0.0.1:9"
    assert seen["model_dir"].startswith(seen["cwd"])
    assert args[args.index("--device") + 1] == "cpu"
    for flag, number in nemotron.GEOMETRY.items():
        assert args[args.index(flag) + 1] == str(number)
    # The work folder, and the model cache in it, are gone afterwards.
    assert not os.path.exists(seen["cwd"])


def test_paths_that_are_not_plain_ascii_are_passed_by_a_plain_name(bundle, tmp_path):
    folder = tmp_path / "José"
    folder.mkdir()
    wav = folder / "appel.wav"
    wav.write_bytes(b"RIFF")
    diarization.diarize(wav, backend="nemotron")
    seen = _invocation(bundle)
    args = seen["args"]
    assert args[args.index("diarize") + 1] == "input.wav"
    assert seen["wav_readable"]


def test_failure_says_what_the_program_said(bundle, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    with pytest.raises(RuntimeError, match=r"exit code 1.*cannot open audio file"):
        diarization.diarize(bundle["wav"], backend="nemotron")


def test_unreadable_output_is_a_plain_error(bundle, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "garbage")
    with pytest.raises(RuntimeError, match="could not be read"):
        diarization.diarize(bundle["wav"], backend="nemotron")


def test_cancel_stops_the_program(bundle, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    began = time.monotonic()

    def cancelled():
        # Once the program has started (it writes its log first).
        return bundle["log"].exists()

    with pytest.raises(CancelledError):
        diarization.diarize(bundle["wav"], backend="nemotron", cancelled=cancelled)
    assert time.monotonic() - began < 20
    with pytest.raises(ProcessLookupError):
        os.kill(_invocation(bundle)["pid"], 0)


def test_a_different_count_is_reported_not_forced(bundle, monkeypatch):
    three = TWO_SPEAKERS + [{"start": 3.0, "end": 4.0, "speaker": 3}]
    monkeypatch.setenv("FAKE_SEGMENTS", json.dumps(three))
    said = []
    turns = diarization.diarize(
        bundle["wav"], backend="nemotron", num_speakers=2, progress=said.append
    )
    assert len({t.speaker for t in turns}) == 3
    assert any("found 3 speaker(s); 2 were expected" in line for line in said)


def test_without_the_bundle_it_says_to_choose_another_method(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    monkeypatch.setattr(resources, "asset_dirs", lambda: [])
    wav = tmp_path / "call.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(RuntimeError, match="Choose another Method"):
        diarization.diarize(wav, backend="nemotron")


def test_offered_only_when_bundled_and_never_chosen_by_auto(bundle, monkeypatch):
    assert "nemotron" in diarization.available_backends()
    assert diarization.chosen_backend("auto") != "nemotron"
    bundle["model"].unlink()
    assert resources.bundled_nemotron() is None
    assert "nemotron" not in diarization.available_backends()


def test_output_turns_without_length_are_dropped():
    text = json.dumps(
        {
            "segments": [
                {"start": 1.0, "end": 1.0, "speaker": 1},
                {"start": 0, "end": 1, "speaker": 2},
            ]
        }
    )
    assert nemotron.parse_output(text) == [(0.0, 1.0, 1)]


# -- what the record says ----------------------------------------------------------


def test_record_names_runtime_model_revision_configuration_and_hash(bundle):
    from whispr.hashing import sha256_file

    (bundle["folder"] / "nemotron.json").write_text(
        json.dumps(
            {
                "runtime": {"name": "NeMo-Speech.cpp", "version": "0.2.0"},
                "model": {
                    "repo": "nvidia/Nemotron-3-Diarization",
                    "revision": "f667ed7",
                },
            }
        ),
        encoding="utf-8",
    )
    record = DiarizationProvenance.from_bundle(
        engine="nemotron", expected_speaker_count=2
    )
    assert record.engine == "nemotron"
    assert record.engine_version == "NeMo-Speech.cpp 0.2.0"
    assert record.pipeline.startswith(
        "nvidia/Nemotron-3-Diarization@f667ed7; chunk 340"
    )
    assert record.segmentation_model == resources.NEMOTRON_MODEL_FILENAME
    assert record.segmentation_model_sha256 == sha256_file(bundle["model"])
    assert record.embedding_model == "" and record.embedding_model_sha256 is None

    lines = AnalysisProvenance(diarization=record).describe()
    assert f"Diarization model: {resources.NEMOTRON_MODEL_FILENAME}" in lines
    assert not any(
        line.startswith(("Speaker embedding", "Segmentation")) for line in lines
    )


def test_record_invents_no_revision_the_build_did_not_write(bundle):
    record = DiarizationProvenance.from_bundle(engine="nemotron")
    assert record.engine_version == ""
    assert "revision not recorded" in record.pipeline


def test_method_list_offers_nemotron_only_when_bundled(bundle):
    pytest.importorskip("tkinter")
    from whispr.gui import transcribe_tab

    offered = transcribe_tab.engine_labels()
    assert [transcribe_tab.ENGINE_CHOICES[label] for label in offered][-1] == "nemotron"
    bundle["model"].unlink()
    offered = transcribe_tab.engine_labels()
    assert "nemotron" not in {transcribe_tab.ENGINE_CHOICES[label] for label in offered}
    assert offered[0] == transcribe_tab.ENGINE_LABELS[0]
