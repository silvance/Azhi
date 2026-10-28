"""Nemotron speaker separation: probabilities in, turns out, offline.

Two halves. Turning the model's per-frame probabilities into turns is plain
arithmetic and is tested on probability arrays written out by hand - these
check the rule, not the model. Running the model is tested against a stand-in
for NVIDIA's C library, compiled here from the declarations in its header
(include/nemo_speech/diar.h, NeMo-Speech.cpp 0.2.0): the real worker process
loads it through the real ctypes binding and gets a known pattern back.
"""

import json
import os
import shutil
import subprocess
import time
import wave

import numpy as np
import pytest

from whispers import diarization, nemotron, resources
from whispers.provenance import AnalysisProvenance, DiarizationProvenance
from whispers.transcription import CancelledError

STEP = 0.01


def _probs(seconds, active):
    """Frames x 8 probabilities: ``active`` maps channel -> [(start, end, p)]."""
    probs = np.full((int(round(seconds / STEP)), 8), 0.02, dtype=np.float32)
    for channel, spans in active.items():
        for start, end, p in spans:
            probs[int(round(start / STEP)) : int(round(end / STEP)), channel] = p
    return probs


def _turns(activity):
    return [(round(a, 2), round(b, 2), s) for a, b, s in activity.turns]


# -- the rule ------------------------------------------------------------------


def test_a_one_word_reply_inside_the_other_persons_speech_is_its_own_turn():
    # A speaks 0-2.0 s and B says "Yes" at 1.0-1.3 s. NVIDIA's cut would have
    # bridged A over it (gap < 0.3 s after 0.3 s of widening) and dropped B's
    # turn as under half a second.
    probs = _probs(2.0, {0: [(0.0, 1.0, 0.9), (1.3, 2.0, 0.9)], 1: [(1.0, 1.3, 0.8)]})
    assert _turns(nemotron.turns_from_activity(probs, STEP)) == [
        (0.0, 1.0, 0),
        (1.0, 1.3, 1),
        (1.3, 2.0, 0),
    ]


def test_overlapping_speech_goes_to_whoever_is_more_likely():
    probs = _probs(2.0, {0: [(0.0, 2.0, 0.7)], 1: [(1.2, 1.5, 0.9)]})
    assert _turns(nemotron.turns_from_activity(probs, STEP)) == [
        (0.0, 1.2, 0),
        (1.2, 1.5, 1),
        (1.5, 2.0, 0),
    ]


def test_nothing_more_likely_than_not_is_silence():
    probs = _probs(3.0, {0: [(0.0, 1.0, 0.9), (2.0, 3.0, 0.9)], 1: [(1.2, 1.6, 0.45)]})
    activity = nemotron.turns_from_activity(probs, STEP)
    assert _turns(activity) == [(0.0, 1.0, 0), (2.0, 3.0, 0)]
    assert activity.voices_heard == 1


def test_speakers_are_numbered_in_the_order_they_first_speak():
    probs = _probs(2.0, {5: [(0.0, 1.0, 0.9)], 2: [(1.0, 2.0, 0.9)]})
    assert [s for _a, _b, s in nemotron.turns_from_activity(probs, STEP).turns] == [
        0,
        1,
    ]


def test_a_given_count_keeps_the_voices_heard_longest():
    # Three voices; told two. The third's moments go to whichever of the two
    # kept is more likely at each of them.
    probs = _probs(
        5.0,
        {
            0: [(0.0, 2.0, 0.9), (4.0, 4.5, 0.4)],
            1: [(2.0, 4.0, 0.9), (4.5, 5.0, 0.3)],
            2: [(4.0, 5.0, 0.8)],
        },
    )
    activity = nemotron.turns_from_activity(probs, STEP, num_speakers=2)
    assert (activity.voices_heard, activity.voices_kept) == (3, 2)
    assert _turns(activity) == [
        (0.0, 2.0, 0),
        (2.0, 4.0, 1),
        (4.0, 4.5, 0),
        (4.5, 5.0, 1),
    ]


def test_a_count_larger_than_heard_changes_nothing_and_is_reported():
    probs = _probs(2.0, {0: [(0.0, 2.0, 0.9)]})
    activity = nemotron.turns_from_activity(probs, STEP, num_speakers=2)
    assert activity.voices_kept == 1
    notes = nemotron._count_notes(activity, 2)
    assert "cannot split a voice" in notes[0]


def test_empty_output_is_no_turns():
    activity = nemotron.turns_from_activity(np.zeros((0, 8)), STEP)
    assert activity.turns == [] and activity.voices_heard == 0


# -- the run, against a stand-in library ---------------------------------------

FAKE_LIBRARY = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    size_t size;
    const char* model_path;
    int32_t gpu;
    const char* preset;
    int32_t chunk_frames, right_context_frames, left_context_frames,
            fifo_frames, spkcache_frames, update_period_frames;
} config_t;
typedef struct { int64_t samples; } stream_t;

static char error[512] = "";

const char* nemo_speech_asr_last_error(void) { return error; }

int nemo_speech_diar_create(const config_t* cfg, void** out) {
    FILE* model = fopen(cfg->model_path, "rb");
    if (!model || getenv("FAKE_FAIL")) {
        snprintf(error, sizeof error, "cannot load %s", cfg->model_path);
        if (model) fclose(model);
        return 1;
    }
    fclose(model);
    FILE* log = fopen(getenv("FAKE_LOG"), "w");
    fprintf(log, "{\"size\": %zu, \"model\": \"%s\", \"gpu\": %d, \"preset\": %s,"
            " \"chunk\": %d, \"rc\": %d, \"lc\": %d, \"fifo\": %d, \"cache\": %d,"
            " \"update\": %d, \"pid\": %d}",
            cfg->size, cfg->model_path, cfg->gpu, cfg->preset ? "\"set\"" : "null",
            cfg->chunk_frames, cfg->right_context_frames, cfg->left_context_frames,
            cfg->fifo_frames, cfg->spkcache_frames, cfg->update_period_frames,
            (int)getpid());
    fclose(log);
    *out = malloc(1);
    return 0;
}
void nemo_speech_diar_destroy(void* model) { free(model); }
int32_t nemo_speech_diar_num_speakers(const void* model) { return 8; }
double nemo_speech_diar_seconds_per_frame(const void* model) { return 0.01; }
int nemo_speech_diar_stream_open(void* model, void** out) {
    *out = calloc(1, sizeof(stream_t));
    return 0;
}
int nemo_speech_diar_stream_push_f32(void* s, const float* x, size_t n, int32_t rate) {
    if (getenv("FAKE_SLOW")) sleep(60);
    if (rate != 16000) { strcpy(error, "wrong rate"); return 1; }
    ((stream_t*)s)->samples += (int64_t)n;
    return 0;
}
int nemo_speech_diar_stream_finish(void* s) { return 0; }
void nemo_speech_diar_stream_close(void* s) { free(s); }
int64_t nemo_speech_diar_frame_count(const void* s) {
    return ((const stream_t*)s)->samples / 160;
}
/* Like the real library on a long stream, keep only recent frames. */
int64_t nemo_speech_diar_frame_probs_start(const void* s) {
    int64_t count = nemo_speech_diar_frame_count(s);
    const char* keep = getenv("FAKE_RETAIN");
    int64_t retain = keep ? atoll(keep) : count;
    return count > retain ? count - retain : 0;
}
/* Every 4 s: A for 2 s, B for 1 s, then 1 s of silence. */
static float prob(int64_t frame, int speaker) {
    int phase = (int)((frame / 100) % 4);
    if (speaker == 0) return phase < 2 ? 0.9f : 0.05f;
    if (speaker == 1) return phase == 2 ? 0.8f : 0.05f;
    return 0.01f;
}
int nemo_speech_diar_frame_probs(const void* s, float* out, size_t capacity) {
    int64_t first = nemo_speech_diar_frame_probs_start(s);
    int64_t count = nemo_speech_diar_frame_count(s);
    if (capacity < (size_t)((count - first) * 8)) { strcpy(error, "too small"); return 1; }
    for (int64_t f = first; f < count; f++)
        for (int k = 0; k < 8; k++) out[(f - first) * 8 + k] = prob(f, k);
    return 0;
}
"""

needs_compiler = pytest.mark.skipif(
    os.name == "nt" or shutil.which("cc") is None,
    reason="builds a stand-in C library with cc",
)


@pytest.fixture(scope="session")
def fake_library(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fake-nemo")
    source = folder / "fake.c"
    source.write_text(FAKE_LIBRARY, encoding="utf-8")
    library = folder / "libnemo_speech_asr_c.so.1"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-o", str(library), str(source)], check=True
    )
    return library


def _wav(path, seconds):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


@pytest.fixture
def bundle(tmp_path, monkeypatch, fake_library):
    """A Nemotron asset folder holding the stand-in library and a fake model."""
    assets = tmp_path / "whispers_assets"
    folder = assets / "nemotron"
    (folder / "lib").mkdir(parents=True)
    library = folder / "lib" / "libnemo_speech_asr_c.so.1"
    shutil.copyfile(fake_library, library)
    model = folder / resources.NEMOTRON_MODEL_FILENAME
    model.write_bytes(b"gguf-weights")
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    log = tmp_path / "invocation.json"
    monkeypatch.setenv("FAKE_LOG", str(log))
    return {
        "assets": assets,
        "folder": folder,
        "library": library,
        "model": model,
        "log": log,
        "wav": _wav(tmp_path / "call.wav", 8),
    }


def _invocation(bundle):
    return json.loads(bundle["log"].read_text())


def _expected(seconds):
    turns = []
    for cycle in range(0, int(seconds), 4):
        turns += [
            (float(cycle), cycle + 2.0, "SPEAKER_00"),
            (cycle + 2.0, cycle + 3.0, "SPEAKER_01"),
        ]
    return turns


@needs_compiler
def test_turns_come_from_the_librarys_probabilities(bundle):
    turns = diarization.diarize(bundle["wav"], backend="nemotron")
    assert [
        (round(t.start, 2), round(t.end, 2), t.speaker) for t in turns
    ] == _expected(8)


@needs_compiler
def test_library_gets_the_bundled_model_cpu_and_the_published_configuration(bundle):
    diarization.diarize(bundle["wav"], backend="nemotron")
    seen = _invocation(bundle)
    assert seen["model"] == str(bundle["model"])
    assert seen["gpu"] == -1 and seen["preset"] is None
    assert (
        seen["chunk"],
        seen["rc"],
        seen["lc"],
        seen["fifo"],
        seen["cache"],
        seen["update"],
    ) == (
        340,
        40,
        -1,
        40,
        264,
        300,
    )


@needs_compiler
def test_a_model_path_that_is_not_plain_ascii_is_passed_by_a_plain_name(
    bundle, tmp_path, monkeypatch
):
    moved = tmp_path / "José" / "whispers_assets"
    shutil.copytree(bundle["assets"], moved)
    monkeypatch.setenv(resources.ENV_ASSETS, str(moved))
    turns = diarization.diarize(bundle["wav"], backend="nemotron")
    assert _invocation(bundle)["model"] == "model.gguf"
    assert len(turns) == 4


@needs_compiler
def test_probabilities_are_kept_for_a_recording_longer_than_the_library_retains(
    bundle, tmp_path, monkeypatch
):
    # The real library drops old probabilities from long streams; the worker
    # copies them as it goes, so nothing early in the recording is lost.
    monkeypatch.setenv("FAKE_RETAIN", "3000")
    wav = _wav(tmp_path / "long.wav", 64)
    turns = diarization.diarize(wav, backend="nemotron")
    assert [
        (round(t.start, 2), round(t.end, 2), t.speaker) for t in turns
    ] == _expected(64)


@needs_compiler
def test_progress_is_reported_as_the_recording_is_processed(bundle, tmp_path):
    seen = []
    nemotron.run(
        _wav(tmp_path / "long.wav", 64),
        bundle["library"],
        bundle["model"],
        on_progress=seen.append,
        poll_seconds=0.01,
    )
    assert seen == sorted(seen) and all(0.0 <= f <= 1.0 for f in seen)


@needs_compiler
def test_failure_says_what_the_library_said(bundle, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(RuntimeError, match="loading the model failed: cannot load"):
        diarization.diarize(bundle["wav"], backend="nemotron")


@needs_compiler
def test_cancel_stops_the_worker(bundle, monkeypatch):
    monkeypatch.setenv("FAKE_SLOW", "1")
    began = time.monotonic()
    with pytest.raises(CancelledError):
        diarization.diarize(
            bundle["wav"], backend="nemotron", cancelled=lambda: bundle["log"].exists()
        )
    assert time.monotonic() - began < 20
    with pytest.raises(ProcessLookupError):
        os.kill(_invocation(bundle)["pid"], 0)


@needs_compiler
def test_a_given_count_is_honoured_and_a_larger_one_reported(bundle):
    said = []
    turns = diarization.diarize(
        bundle["wav"], backend="nemotron", num_speakers=1, progress=said.append
    )
    assert {t.speaker for t in turns} == {"SPEAKER_00"}
    assert any("heard 2 voices; as 1 were expected" in line for line in said)
    said.clear()
    diarization.diarize(
        bundle["wav"], backend="nemotron", num_speakers=3, progress=said.append
    )
    assert any("cannot split a voice" in line for line in said)


def test_without_the_bundle_it_says_to_choose_another_method(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "asset_dirs", lambda: [])
    wav = _wav(tmp_path / "call.wav", 1)
    with pytest.raises(RuntimeError, match="Choose another Method"):
        diarization.diarize(wav, backend="nemotron")


@needs_compiler
def test_auto_takes_nemotron_where_the_build_includes_it(bundle):
    assert "nemotron" in diarization.available_backends()
    assert diarization.chosen_backend("auto") == "nemotron"
    assert diarization.chosen_backend("auto", 2) == "nemotron"
    # More people than it can hold apart: one of the others.
    assert diarization.chosen_backend("auto", nemotron.MAX_SPEAKERS + 1) != "nemotron"
    # A method chosen by name is the method run.
    assert diarization.chosen_backend("sherpa", 2) == "sherpa"
    bundle["model"].unlink()
    assert resources.bundled_nemotron() is None
    assert "nemotron" not in diarization.available_backends()
    assert diarization.chosen_backend("auto") != "nemotron"


def test_auto_runs_nemotron_when_diarizing(bundle, monkeypatch):
    calls = []
    monkeypatch.setattr(
        nemotron,
        "diarize",
        lambda wav, **kwargs: calls.append(kwargs) or [(0.0, 1.0, "SPEAKER_00")],
    )
    turns = diarization.diarize("a.wav", num_speakers=2)
    assert calls and calls[0]["num_speakers"] == 2
    assert turns[0].speaker == "SPEAKER_00"


def test_shipped_copy_starts_the_worker_through_its_own_executable(monkeypatch):
    monkeypatch.setattr(nemotron.sys, "frozen", True, raising=False)
    command = nemotron.worker_command("lib.dll", "model.gguf", "a.wav", "out")
    assert command[1:3] == ["--nemotron-worker", "lib.dll"]
    monkeypatch.delattr(nemotron.sys, "frozen")
    command = nemotron.worker_command("lib.dll", "model.gguf", "a.wav", "out")
    assert command[1:3] == ["-m", "whispers.nemotron_worker"]


# -- what the record says --------------------------------------------------------


@needs_compiler
def test_record_names_runtime_model_revision_method_and_hash(bundle):
    from whispers.hashing import sha256_file

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
    assert "most probable speaker in each 10 ms frame" in record.pipeline
    assert record.segmentation_model == resources.NEMOTRON_MODEL_FILENAME
    assert record.segmentation_model_sha256 == sha256_file(bundle["model"])
    assert record.embedding_model == "" and record.embedding_model_sha256 is None

    lines = AnalysisProvenance(diarization=record).describe()
    assert f"Diarization model: {resources.NEMOTRON_MODEL_FILENAME}" in lines
    assert not any(
        line.startswith(("Speaker embedding", "Segmentation")) for line in lines
    )


@needs_compiler
def test_record_invents_no_revision_the_build_did_not_write(bundle):
    record = DiarizationProvenance.from_bundle(engine="nemotron")
    assert record.engine_version == ""
    assert "revision not recorded" in record.pipeline


@needs_compiler
def test_method_list_offers_nemotron_only_when_bundled(bundle):
    pytest.importorskip("tkinter")
    from whispers.gui import transcribe_tab

    offered = transcribe_tab.engine_labels()
    # Straight after Auto, which runs it.
    assert [transcribe_tab.ENGINE_CHOICES[label] for label in offered][:2] == [
        "auto",
        "nemotron",
    ]
    bundle["model"].unlink()
    offered = transcribe_tab.engine_labels()
    assert "nemotron" not in {transcribe_tab.ENGINE_CHOICES[label] for label in offered}
    assert offered[0] == transcribe_tab.ENGINE_LABELS[0]
