"""Speaker diarization backend using sherpa-onnx (ONNX / CPU, no PyTorch).

Diarization answers "who spoke when": it splits the audio into speaker turns and
labels each with a speaker id. We use `sherpa-onnx
<https://github.com/k2-fsa/sherpa-onnx>`_, which runs on onnxruntime (already a
dependency for VAD), needs no PyTorch, and uses freely downloadable models that
bundle cleanly for air-gapped use.

``sherpa_onnx`` and ``numpy`` are imported lazily so that importing this module -
and the rest of the package - does not require the optional dependencies.
"""

from __future__ import annotations

import bisect
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import hub_cache
from .resources import bundled_diarization_models, pyannote_cache_dir
from .thresholds import DIARIZATION_CLUSTERING_THRESHOLD
from .transcription import (
    CancelCallback,
    CancelledError,
    ProgressCallback,
    Segment,
    Word,
)

PathLike = Union[str, Path]

# Diarization models operate on 16 kHz mono audio (the format convert_to_wav emits).
DIARIZATION_SAMPLE_RATE = 16000


@dataclass
class SpeakerSegment:
    """A span of audio attributed to a single speaker."""

    start: float
    end: float
    speaker: str


def _resolve_models(
    segmentation_model: Optional[PathLike],
    embedding_model: Optional[PathLike],
) -> Tuple[str, str]:
    if segmentation_model and embedding_model:
        return str(segmentation_model), str(embedding_model)
    bundled = bundled_diarization_models()
    if bundled is None:
        raise RuntimeError(
            "No diarization models found. Bundle them under "
            "whispr_assets/diarization/ (segmentation.onnx + embedding.onnx), set "
            "WHISPR_ASSETS, or pass explicit model paths."
        )
    seg, emb = bundled
    return str(seg), str(emb)


def _read_wav_mono16k(path: PathLike):
    """Read a 16-bit PCM WAV as a mono float32 numpy array, with its sample rate."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise RuntimeError("diarization expects a 16-bit PCM WAV")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sample_rate


BACKENDS = ("auto", "pyannote", "sherpa", "nemotron")


def diarize(
    wav_path: PathLike,
    *,
    backend: str = "auto",
    segmentation_model: Optional[PathLike] = None,
    embedding_model: Optional[PathLike] = None,
    num_speakers: Optional[int] = None,
    threshold: float = DIARIZATION_CLUSTERING_THRESHOLD,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> List[SpeakerSegment]:
    """Diarize a WAV into speaker-labelled segments.

    ``backend`` selects the engine:

    * ``"auto"`` (default) - pyannote.audio if installed, else sherpa-onnx.
    * ``"pyannote"`` - force pyannote (best on hard/low-quality audio).
    * ``"sherpa"`` - force the sherpa-onnx ONNX pipeline (no PyTorch, faster;
      good enough for clean audio).
    * ``"nemotron"`` - NVIDIA's end-to-end model (:mod:`whispr.nemotron`), when
      the build includes it. Never chosen by "auto": it has not been measured
      on this deployment's recordings, and cannot take a speaker count.

    Every engine returns the same ``SpeakerSegment`` list, so the rest of the
    app is unaffected.
    """
    backend = (backend or "auto").lower()
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown diarization backend {backend!r}; choose from {', '.join(BACKENDS)}"
        )

    if backend == "pyannote" and not _pyannote_available():
        raise RuntimeError(
            "The pyannote backend was requested but pyannote.audio is not "
            "installed. Choose the sherpa backend or install the pyannote extra."
        )
    if backend == "pyannote" and not _pyannote_usable_offline():
        raise RuntimeError(
            "This copy does not include the pyannote speaker-separation models, "
            "so that method cannot run here. Choose the sherpa method under "
            "Advanced options, or untick \u201cIdentify who is speaking\u201d."
        )

    if backend == "nemotron":
        from . import nemotron

        return [
            SpeakerSegment(start=start, end=end, speaker=speaker)
            for start, end, speaker in nemotron.diarize(
                wav_path,
                num_speakers=num_speakers,
                progress=progress,
                on_progress=on_progress,
                cancelled=cancelled,
            )
        ]

    # Not just "is the library installed": in a bundle without the pyannote
    # cache, Pipeline.from_pretrained would go to the Hugging Face Hub for it.
    use_pyannote = chosen_backend(backend) == "pyannote"
    if use_pyannote:
        return _diarize_pyannote(
            wav_path,
            num_speakers=num_speakers,
            progress=progress,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    return _diarize_sherpa(
        wav_path,
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        num_speakers=num_speakers,
        threshold=threshold,
        progress=progress,
        on_progress=on_progress,
        cancelled=cancelled,
    )


def available_backends() -> List[str]:
    """The diarization engines this build can actually run, preferred first.

    An engine counts only when both halves are present: the library *and* the
    models it needs, bundled. pyannote installed with no offline cache is not a
    working diarizer on an air-gapped machine, and neither is sherpa-onnx with
    no segmentation model beside it.
    """
    # Reached through the module rather than the names imported above, so a
    # caller that redirects whispr.resources - a test, or a build pointed at a
    # different asset root - sees the same answer here as everywhere else.
    from . import resources

    out: List[str] = []
    if _pyannote_available() and resources.pyannote_cache_dir() is not None:
        out.append("pyannote")
    if (
        _module_available("sherpa_onnx")
        and resources.bundled_diarization_models() is not None
    ):
        out.append("sherpa")
    if resources.bundled_nemotron() is not None:
        out.append("nemotron")
    return out


def is_available() -> bool:
    """True when speaker separation can run at all in this build."""
    return bool(available_backends())


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _pyannote_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        # find_spec raises (rather than returning None) when the parent package
        # itself is absent.
        return False


def chosen_backend(backend: str = "auto") -> str:
    """The engine :func:`diarize` will run for ``backend`` - never "auto".

    One answer for two questions: which engine to run, and which engine the
    analysis record says ran. The record used to store the *setting*, so a
    report could only ever say "auto", and could not say which engine's
    models to name.
    """
    backend = (backend or "auto").lower()
    if backend == "auto":
        return "pyannote" if _pyannote_usable_offline() else "sherpa"
    return backend


def _pyannote_usable_offline() -> bool:
    """Whether pyannote can run without reaching for the network.

    The library alone is not enough in a bundle: its models come from the
    bundled cache or from the Hub, and a bundle may not go to the Hub. From
    source the Hub is still allowed, as it always was.
    """
    from . import resources

    if not _pyannote_available():
        return False
    return resources.pyannote_cache_dir() is not None or not resources.is_bundle()


def _log_cache_contents(hub, progress: ProgressCallback) -> None:
    """Report what's actually in the bundled HF cache (diagnostic).

    Lets us see from the status log whether each model's weight file really made
    it into the bundle, instead of inferring it.
    """
    from pathlib import Path

    hub = Path(hub)
    files = sorted(p for p in hub.rglob("*") if p.is_file())
    progress(f"Bundled cache: {len(files)} file(s) under {hub}")
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        # Highlight the weight files we depend on.
        rel = p.relative_to(hub)
        if p.name.endswith((".bin", ".yaml", ".onnx", ".pt", ".safetensors")):
            progress(f"  {rel}  ({size:,} B)")


def _diarize_pyannote(
    wav_path: PathLike,
    *,
    num_speakers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> List[SpeakerSegment]:
    """Diarize with pyannote.audio (speaker-diarization-3.1) on CPU.

    Loads from a bundled offline HF cache (``whispr_assets/pyannote``) when
    present; otherwise from the normal HF cache using ``HF_TOKEN``.

    pyannote runs inference as a single call; where the installed version accepts
    a progress hook, that call reports its steps through one - which is both how
    the bar advances during the pass and how cancellation is honoured inside it.
    Without hook support it stays a single opaque call, cancellable only at the
    phase boundaries either side.
    """
    if cancelled is not None and cancelled():
        raise CancelledError("Diarization cancelled.")
    # NB: HF offline mode + the bundled cache are configured at startup by
    # resources.configure_offline_hf_cache(); it has to run before huggingface_hub
    # is imported (which has already happened by here, via faster-whisper), so do
    # NOT try to set HF_HOME / HF_HUB_OFFLINE in this function - it's too late.
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # When the offline cache is bundled, point pyannote at it. Passing cache_dir
    # only covers the top-level config; pyannote 3.1.1 loads the sub-models
    # (segmentation / embedding) via hf_hub_download WITHOUT forwarding cache_dir,
    # so they read huggingface_hub's HF_HUB_CACHE *constant* - which was frozen at
    # import (before our startup env override). Patch the constant (and offline)
    # at runtime so the sub-models also resolve from the bundle.
    cache = pyannote_cache_dir()
    load_kwargs = {}
    if cache is not None and (cache / "hub").is_dir():
        hub = cache / "hub"
        load_kwargs["cache_dir"] = str(hub)
        # A bundled cache can hold every weight and still be unreadable offline
        # (see whispr.hub_cache). Where the missing piece can be restated from
        # what is on disk, restate it; where it cannot, say so here in plain
        # words rather than letting huggingface_hub fail several layers down
        # with advice about checking the internet connection.
        repaired = hub_cache.repair(hub, hub_cache.PYANNOTE_REPOS)
        if repaired and progress is not None:
            progress(f"Repaired the bundled model cache for {len(repaired)} model(s).")
        problems = hub_cache.unreadable(hub, hub_cache.PYANNOTE_REPOS)
        if problems:
            raise RuntimeError(
                "The bundled speaker models cannot be read. This build needs to "
                "be rebuilt; nothing you can change on this machine will fix "
                "it, and the app will not go looking on the internet for them.\n  "
                + "\n  ".join(problems)
            )
        try:
            import huggingface_hub.constants as hf_constants

            hf_constants.HF_HUB_CACHE = str(hub)
            hf_constants.HF_HUB_OFFLINE = True
            patched = hf_constants.HF_HUB_CACHE
        except Exception as exc:  # noqa: BLE001
            patched = f"<patch failed: {exc}>"
        if progress is not None:
            progress(f"Using bundled pyannote cache: {hub}")
            # Verbose per-file cache listing is dev diagnostics; opt in with
            # WHISPR_DEBUG to keep the status log clean in normal use.
            if os.environ.get("WHISPR_DEBUG"):
                progress(f"HF_HUB_CACHE -> {patched}")
                _log_cache_contents(hub, progress)

    if progress is not None:
        progress("Loading pyannote pipeline...")
    model_id = "pyannote/speaker-diarization-3.1"
    try:
        # pyannote 3.x / current huggingface_hub use `token`.
        pipeline = Pipeline.from_pretrained(model_id, token=token, **load_kwargs)
    except TypeError:
        # Older pyannote used `use_auth_token`.
        pipeline = Pipeline.from_pretrained(
            model_id, use_auth_token=token, **load_kwargs
        )
    if pipeline is None:
        raise RuntimeError(
            "Could not load the pyannote pipeline. Accept the licenses for "
            "pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0 and "
            "pyannote/wespeaker-voxceleb-resnet34-LM on Hugging Face and set "
            "HF_TOKEN, or bundle the models under whispr_assets/pyannote."
        )
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    pipeline.to(torch.device("cpu"))

    if cancelled is not None and cancelled():
        raise CancelledError("Diarization cancelled.")
    if progress is not None:
        progress("Identifying speakers (pyannote)...")
    params: "dict[str, Any]" = {"num_speakers": num_speakers} if num_speakers else {}
    if _accepts_hook(pipeline):
        params["hook"] = _pyannote_hook(progress, on_progress, cancelled)
    output = pipeline(str(wav_path), **params)
    annotation = _as_annotation(output)

    segments = [
        SpeakerSegment(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda seg: seg.start)
    if on_progress is not None:
        on_progress(1.0)
    return segments


# How much of a pyannote pass each step accounts for, so the bar advances
# through the run instead of jumping from nothing to done at the end. The names
# are pyannote's own and could change between versions: a step this build does
# not recognise moves the label but holds the bar, which is honest about not
# knowing rather than inventing a number.
_PYANNOTE_STEPS = {
    "segmentation": (0.0, 0.45),
    "embeddings": (0.45, 0.95),
}


def _pyannote_hook(
    progress: Optional[ProgressCallback],
    on_progress: Optional[Callable[[float], None]],
    cancelled: Optional[CancelCallback],
):
    """Build the callback pyannote reports its progress through.

    pyannote runs its pipeline as one call, so without this the whole pass is a
    single opaque wait - on a long recording, minutes during which the interface
    can only say that something is happening. The hook is also the only place
    cancellation can be honoured mid-pass rather than at the phase boundaries
    either side of it.
    """
    fraction = 0.0
    current_step = ""

    def hook(
        step_name: str,
        step_artifact: Any = None,
        file: Any = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ) -> None:
        nonlocal fraction, current_step
        if cancelled is not None and cancelled():
            raise CancelledError("Diarization cancelled.")
        if step_name != current_step:
            current_step = step_name
            if progress is not None:
                progress(f"Identifying speakers: {step_name}")
        band = _PYANNOTE_STEPS.get(step_name)
        if band is None:
            moved = fraction
        elif total:
            low, high = band
            moved = low + (high - low) * min(1.0, max(0, completed or 0) / total)
        else:
            moved = band[0]
        # Never backwards: steps overlap and report unevenly, and a bar that
        # retreats reads as something having gone wrong.
        fraction = max(fraction, float(moved))
        if on_progress is not None:
            on_progress(fraction)

    return hook


def _accepts_hook(pipeline: Any) -> bool:
    """Whether this pyannote build takes a progress hook.

    Checked rather than attempted: calling with an unsupported keyword raises
    TypeError, which is indistinguishable from a TypeError raised by a genuine
    fault inside the pipeline.
    """
    import inspect

    try:
        return "hook" in inspect.signature(pipeline.apply).parameters
    except (AttributeError, TypeError, ValueError):
        # No apply, or one that cannot be introspected: assume no hook rather
        # than fail the diarization over the question.
        return False


def _as_annotation(output):
    """Return the pyannote ``Annotation`` from a pipeline result.

    Older pyannote returns the ``Annotation`` directly; newer versions wrap it in
    a ``DiarizeOutput`` exposing it as ``.speaker_diarization`` (or ``.diarization``).
    """
    if hasattr(output, "itertracks"):
        return output
    for attr in ("speaker_diarization", "diarization"):
        inner = getattr(output, attr, None)
        if inner is not None and hasattr(inner, "itertracks"):
            return inner
    raise RuntimeError(f"Unexpected pyannote output type: {type(output).__name__}")


def _diarize_sherpa(
    wav_path: PathLike,
    *,
    segmentation_model: Optional[PathLike] = None,
    embedding_model: Optional[PathLike] = None,
    num_speakers: Optional[int] = None,
    threshold: float = DIARIZATION_CLUSTERING_THRESHOLD,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> List[SpeakerSegment]:
    """Diarize a 16 kHz mono WAV into speaker-labelled segments.

    Parameters
    ----------
    wav_path
        Path to a 16 kHz mono PCM WAV (see ``transcription.convert_to_wav``).
    segmentation_model, embedding_model
        Paths to the sherpa-onnx segmentation and speaker-embedding ONNX models.
        If omitted, bundled models under ``whispr_assets/diarization/`` are used.
    num_speakers
        Expected number of speakers. ``None`` (default) auto-detects via
        clustering with ``threshold``.
    threshold
        Clustering threshold used when ``num_speakers`` is not given.
    progress
        Optional status callback.
    """
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "sherpa-onnx is not installed. Install the GUI/transcription extras "
            "with:  pip install 'silvance-whisper[gui]'"
        ) from exc

    seg_model, emb_model = _resolve_models(segmentation_model, embedding_model)

    if progress is not None:
        progress("Loading diarization models...")

    # sherpa-onnx defaults each model to a single thread, which makes CPU
    # diarization extremely slow; use all available cores.
    num_threads = max(1, os.cpu_count() or 1)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=emb_model, num_threads=num_threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers else -1,
            threshold=threshold,
        ),
        min_duration_on=0.3,
        # Gaps shorter than this inside one speaker's speech are bridged. At
        # 0.5 s that can bridge right over a quick reply from the other person,
        # leaving the two turns overlapping. 0.2 s lowered the diarization
        # error rate in 14 of 22 test runs (whispr.diarization_eval) and
        # raised none by more than a tenth of a point.
        min_duration_off=0.2,
    )
    if not config.validate():
        raise RuntimeError(
            "Invalid diarization configuration; check the model paths "
            f"(segmentation={seg_model}, embedding={emb_model})."
        )

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    samples, sample_rate = _read_wav_mono16k(wav_path)
    if sample_rate != sd.sample_rate:
        raise RuntimeError(
            f"diarization expects {sd.sample_rate} Hz audio, got {sample_rate} Hz"
        )

    if cancelled is not None and cancelled():
        raise CancelledError("Diarization cancelled.")
    if progress is not None:
        progress("Identifying speakers...")

    def _on_chunk(num_processed: int, num_total: int, *_extra) -> int:
        if cancelled is not None and cancelled():
            raise CancelledError("Diarization cancelled.")
        if on_progress is not None and num_total:
            on_progress(min(num_processed / num_total, 1.0))
        return 0

    # Always pass the callback so cancellation is checked even without on_progress.
    result = sd.process(samples, callback=_on_chunk).sort_by_start_time()
    if on_progress is not None:
        on_progress(1.0)
    return [
        SpeakerSegment(start=s.start, end=s.end, speaker=f"SPEAKER_{s.speaker:02d}")
        for s in result
    ]


def _best_overlap_speaker(
    start: float, end: float, speaker_segments: Sequence[SpeakerSegment]
) -> Optional[str]:
    best_speaker: Optional[str] = None
    best_overlap = 0.0
    for sp in speaker_segments:
        overlap = min(end, sp.end) - max(start, sp.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = sp.speaker
    return best_speaker


class _TurnIndex:
    """Speaker turns sorted by start, for finding those that touch a span quickly.

    A long recording has thousands of turns and tens of thousands of words;
    scanning every turn for every word was the slow part of assignment.
    """

    def __init__(self, turns: Sequence[SpeakerSegment]) -> None:
        self.turns = sorted(turns, key=lambda t: (t.start, t.end))
        self._starts = [t.start for t in self.turns]
        self._longest = max((t.end - t.start for t in self.turns), default=0.0)

    def touching(self, start: float, end: float) -> List[SpeakerSegment]:
        """Turns that overlap ``[start, end]`` (or contain the point when equal)."""
        lo = bisect.bisect_left(self._starts, start - self._longest)
        hi = bisect.bisect_right(self._starts, end)
        return [t for t in self.turns[lo:hi] if t.end >= start and t.start <= end]


def exclusive_coverage(
    start: float, end: float, turns: Sequence[SpeakerSegment]
) -> Dict[str, float]:
    """Seconds of ``[start, end]`` during which each speaker is the only one active.

    Time where two people are both marked as speaking counts for neither: it is
    evidence that someone spoke, not of who.
    """
    out: Dict[str, float] = {}
    if end <= start:
        return out
    relevant = [t for t in turns if t.start < end and t.end > start]
    if not relevant:
        return out
    edges = {start, end}
    for turn in relevant:
        edges.add(min(max(turn.start, start), end))
        edges.add(min(max(turn.end, start), end))
    points = sorted(edges)
    for a, b in zip(points, points[1:]):
        if b <= a:
            continue
        middle = (a + b) / 2.0
        active = {t.speaker for t in relevant if t.start <= middle < t.end}
        if len(active) == 1:
            speaker = next(iter(active))
            out[speaker] = out.get(speaker, 0.0) + (b - a)
    return out


# A word sitting more than this far from any speaker turn is essentially
# unattributable; return None (which assign_speakers handles) rather than
# confidently snapping it to a distant speaker.
_MAX_GAP_SECONDS = 2.0


def _speaker_for_word(start: float, end: float, index: _TurnIndex) -> Optional[str]:
    """The speaker a word belongs to.

    The speaker heard alone for most of the word's span. It used to be whoever
    was active at the word's midpoint, taking the first turn listed when two
    overlapped - and the first listed is the turn that started earlier, so a
    quick reply that the diarizer marked as overlapping the end of the other
    person's turn was credited to the person who had been talking. Only when
    the whole word sits in overlapped speech (a backchannel under someone
    else's sentence, say) or in a gap does it fall back to that rule.
    """
    nearby = index.touching(start, end)
    coverage = exclusive_coverage(start, end, nearby)
    if coverage:
        return max(coverage, key=lambda speaker: coverage[speaker])
    return _speaker_at((start + end) / 2.0, index.turns)


def _speaker_at(t: float, speaker_segments: Sequence[SpeakerSegment]) -> Optional[str]:
    """Speaker active at time ``t``.

    Falls back to the nearest turn when ``t`` lands in a small gap, but returns
    ``None`` past ``_MAX_GAP_SECONDS`` rather than guessing across a long silence.
    """
    nearest: Optional[str] = None
    nearest_distance = float("inf")
    for sp in speaker_segments:
        if sp.start <= t <= sp.end:
            return sp.speaker
        distance = sp.start - t if t < sp.start else t - sp.end
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = sp.speaker
    if nearest_distance > _MAX_GAP_SECONDS:
        return None
    return nearest


@dataclass
class _Run:
    """A run of consecutive words attributed to one speaker (pre-segment)."""

    speaker: Optional[str]
    words: List[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start


def _coalesce(runs: List[_Run]) -> List[_Run]:
    """Merge adjacent runs that share a speaker."""
    out: List[_Run] = []
    for run in runs:
        if out and out[-1].speaker == run.speaker:
            out[-1].words.extend(run.words)
        else:
            out.append(_Run(speaker=run.speaker, words=list(run.words)))
    return out


# A short run is the diarizer's own finding, and kept, when its speaker's
# turns cover at least this share of the run.
SHORT_RUN_SUPPORT = 0.5


def _supported(run: _Run, index: Optional[_TurnIndex]) -> bool:
    """True when the diarizer has the run's speaker talking over most of it.

    Overlapped time counts: a reply that starts under the end of the question
    is still a turn the diarizer found. A word that merely clips the edge of
    someone else's turn is not.
    """
    if index is None or run.speaker is None:
        return False
    # A one-word run can be a few milliseconds long; judge it over a span long
    # enough for "most of it" to mean something.
    start, end = run.start, max(run.end, run.start + 0.1)
    covered = sum(
        min(t.end, end) - max(t.start, start)
        for t in index.touching(start, end)
        if t.speaker == run.speaker
    )
    return covered >= SHORT_RUN_SUPPORT * (end - start)


def _merge_short_runs(
    runs: List[_Run],
    index: Optional[_TurnIndex] = None,
    *,
    min_seconds: float = 0.8,
    max_words: int = 2,
) -> List[_Run]:
    """Absorb stray sub-second / 1-2 word runs into an adjacent speaker.

    Whisper's word timings and the diarizer's turn edges rarely agree to the
    tenth of a second, so a word can land on a sliver of someone else's turn.
    Such a fragment is relabelled to a neighbouring speaker - the only
    neighbour at the edges, or the longer (by word count) neighbour in the
    middle - and then coalesced.

    It used to relabel every short run, which is exactly what a quick "Yes."
    or "No." from the other person looks like: two words or fewer, under a
    second, between someone else's sentences. A short run is now left alone
    when the diarizer itself has that speaker talking over most of it
    (:data:`SHORT_RUN_SUPPORT`) - the reply was found, not inferred from a
    misaligned word. On spliced test conversations with the reference turns
    given, the old rule lost 12 of 72 replies under a second; this loses none.
    """
    # Single linear pass: relabel each short run to a neighbour, then coalesce
    # once. We work on a copy so the caller's runs are left untouched.
    runs = [_Run(speaker=r.speaker, words=list(r.words)) for r in runs]
    for i, run in enumerate(runs):
        if run.duration >= min_seconds or len(run.words) > max_words:
            continue
        if _supported(run, index):
            continue
        prev = runs[i - 1] if i > 0 else None
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if prev is None and nxt is None:
            continue
        if prev is None:
            target = nxt
        elif nxt is None:
            target = prev
        else:
            target = prev if len(prev.words) >= len(nxt.words) else nxt
        if target is not None and target.speaker != run.speaker:
            run.speaker = target.speaker
    return _coalesce(runs)


def assign_speakers(
    segments: List[Segment],
    speaker_segments: Sequence[SpeakerSegment],
) -> List[Segment]:
    """Attach speaker labels to a transcript, splitting on speaker changes.

    When per-word timestamps are available, each word is assigned to the speaker
    heard alone for most of it (:func:`_speaker_for_word`) and consecutive
    same-speaker words are grouped into runs - so a single Whisper segment that
    spans a speaker change is split correctly. A smoothing pass
    (:func:`_merge_short_runs`) then absorbs stray fragments the diarizer did
    not itself support. Each surviving run becomes one segment.
    Segments without word timestamps fall back to whole-segment overlap.
    """
    out: List[Segment] = []
    pending: List[_Run] = []
    index = _TurnIndex(speaker_segments)

    def _emit_pending() -> None:
        for run in _merge_short_runs(pending, index):
            text = "".join(w.word for w in run.words).strip()
            if not text:
                continue
            out.append(
                Segment(
                    start=run.words[0].start,
                    end=run.words[-1].end,
                    text=text,
                    speaker=run.speaker,
                    words=list(run.words),
                )
            )
        pending.clear()

    for segment in segments:
        if not segment.words:
            # Mixed/no-word segment: flush what we have, then attribute whole.
            _emit_pending()
            segment.speaker = _best_overlap_speaker(
                segment.start, segment.end, speaker_segments
            )
            out.append(segment)
            continue

        for word in segment.words:
            speaker = _speaker_for_word(word.start, word.end, index)
            if pending and pending[-1].speaker == speaker:
                pending[-1].words.append(word)
            else:
                pending.append(_Run(speaker=speaker, words=[word]))

    _emit_pending()
    return out
