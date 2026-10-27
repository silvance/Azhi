"""Provenance for an analysis: what produced this result, from what, with what.

A transcript or a similarity score is only defensible if you can say which
recording it came from, which models and settings processed it, and which build
of the application ran. This module assembles that record so it can be stored in
a project file and printed into a report.

Everything here is derived locally: file hashes, the bundled models already on
disk, the settings the operator chose, and the build manifest. The source media
is hashed, never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .buildinfo import UNKNOWN, BuildInfo, build_info, runtime_summary
from .hashing import sha256_file_or_none
from .thresholds import DEFAULTS, Thresholds, active

PathLike = Union[str, Path]

# Bump when this record's shape changes; readers tolerate older versions.
PROVENANCE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SourceRecord:
    """The recording an analysis was performed on."""

    filename: str = ""
    original_path: str = ""
    file_size: Optional[int] = None
    sha256: Optional[str] = None

    @classmethod
    def from_path(cls, path: Optional[PathLike]) -> "SourceRecord":
        """Describe and hash ``path`` (read-only; missing files degrade cleanly)."""
        if path is None:
            return cls()
        source = Path(path)
        size: Optional[int] = None
        try:
            size = source.stat().st_size
        except OSError:
            size = None
        return cls(
            filename=source.name,
            original_path=str(source),
            file_size=size,
            sha256=sha256_file_or_none(source),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "original_path": self.original_path,
            "file_size": self.file_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRecord":
        size = data.get("file_size")
        return cls(
            filename=str(data.get("filename") or ""),
            original_path=str(data.get("original_path") or ""),
            file_size=int(size) if isinstance(size, int) else None,
            sha256=(str(data["sha256"]) if data.get("sha256") else None),
        )


@dataclass
class TranscriptionProvenance:
    """How the words were produced."""

    model_name: str = UNKNOWN
    model_sha256: Optional[str] = None
    device: str = "cpu"
    compute_type: str = "int8"
    language_setting: str = "auto"
    detected_language: str = ""
    vad: bool = True
    beam_size: int = 5
    initial_prompt: str = ""
    # Whether the words came from a cleaned copy rather than the recording, and
    # which model cleaned it. A transcript has to be able to say this: cleaning
    # changes the audio the model heard.
    denoised: bool = False
    denoiser_sha256: str = ""
    # A listening copy written from this recording, if one was. It did not
    # produce these words - the transcript is made from the recording, or from
    # the cleaned copy above - so this records an artefact that exists rather
    # than a step the words went through. It is here so that a file found
    # beside a transcript months later can be accounted for, and so the filters
    # that made it can be read back rather than guessed at.
    listening_copy: str = ""
    listening_filters: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "device": self.device,
            "compute_type": self.compute_type,
            "language_setting": self.language_setting,
            "detected_language": self.detected_language,
            "vad": self.vad,
            "beam_size": self.beam_size,
            "initial_prompt": self.initial_prompt,
            "denoised": self.denoised,
            "denoiser_sha256": self.denoiser_sha256,
            "listening_copy": self.listening_copy,
            "listening_filters": self.listening_filters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionProvenance":
        return cls(
            model_name=str(data.get("model_name") or UNKNOWN),
            model_sha256=(
                str(data["model_sha256"]) if data.get("model_sha256") else None
            ),
            device=str(data.get("device") or "cpu"),
            compute_type=str(data.get("compute_type") or "int8"),
            language_setting=str(data.get("language_setting") or "auto"),
            detected_language=str(data.get("detected_language") or ""),
            vad=bool(data.get("vad", True)),
            beam_size=int(data.get("beam_size") or 5),
            initial_prompt=str(data.get("initial_prompt") or ""),
            denoised=bool(data.get("denoised") or False),
            denoiser_sha256=str(data.get("denoiser_sha256") or ""),
            listening_copy=str(data.get("listening_copy") or ""),
            listening_filters=str(data.get("listening_filters") or ""),
        )


@dataclass
class DiarizationProvenance:
    """How speakers were separated (absent when diarization was not run)."""

    engine: str = ""
    engine_version: str = ""
    segmentation_model: str = ""
    segmentation_model_sha256: Optional[str] = None
    embedding_model: str = ""
    embedding_model_sha256: Optional[str] = None
    expected_speaker_count: Optional[int] = None
    clustering_threshold: Optional[float] = None
    # pyannote's pipeline, as repo@commit; empty for sherpa, which has none.
    pipeline: str = ""
    # Names put on speakers by voice similarity to learned voices. Each was
    # shown as an "(unconfirmed voice lead)"; listed here so the record says
    # which names came from the recogniser rather than from the operator.
    voice_leads: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "voice_leads": list(self.voice_leads),
            "engine": self.engine,
            "engine_version": self.engine_version,
            "segmentation_model": self.segmentation_model,
            "segmentation_model_sha256": self.segmentation_model_sha256,
            "embedding_model": self.embedding_model,
            "embedding_model_sha256": self.embedding_model_sha256,
            "expected_speaker_count": self.expected_speaker_count,
            "clustering_threshold": self.clustering_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiarizationProvenance":
        count = data.get("expected_speaker_count")
        threshold = data.get("clustering_threshold")
        return cls(
            engine=str(data.get("engine") or ""),
            engine_version=str(data.get("engine_version") or ""),
            segmentation_model=str(data.get("segmentation_model") or ""),
            segmentation_model_sha256=(
                str(data["segmentation_model_sha256"])
                if data.get("segmentation_model_sha256")
                else None
            ),
            embedding_model=str(data.get("embedding_model") or ""),
            embedding_model_sha256=(
                str(data["embedding_model_sha256"])
                if data.get("embedding_model_sha256")
                else None
            ),
            expected_speaker_count=int(count) if isinstance(count, int) else None,
            clustering_threshold=(
                float(threshold) if isinstance(threshold, (int, float)) else None
            ),
            pipeline=str(data.get("pipeline") or ""),
            voice_leads=[str(name) for name in (data.get("voice_leads") or []) if name],
        )

    @classmethod
    def from_bundle(
        cls,
        *,
        engine: str,
        expected_speaker_count: Optional[int] = None,
        clustering_threshold: Optional[float] = None,
        voice_leads: "Optional[List[str]]" = None,
    ) -> "DiarizationProvenance":
        """Describe the speaker split that ran, from the models it actually loaded.

        ``engine`` may be the setting - "auto" is resolved exactly as
        :func:`whispers.diarization.diarize` resolves it - so the record names the
        engine that ran rather than the option that was picked.

        It used to fill sherpa's fields whichever engine ran, so a pyannote run
        was described by sherpa's files. Each engine is now described by its
        own. (For sherpa the old record was right: the Speaker Profiles model
        and sherpa's diarization embedding are the same file.)
        """
        from .diarization import chosen_backend

        record = cls(
            engine=chosen_backend(engine),
            expected_speaker_count=expected_speaker_count,
            clustering_threshold=clustering_threshold,
            voice_leads=sorted({name for name in (voice_leads or []) if name}),
        )
        if record.engine == "pyannote":
            record._describe_pyannote()
        elif record.engine == "nemotron":
            record._describe_nemotron()
        else:
            record._describe_sherpa()
        return record

    def _describe_nemotron(self) -> None:
        """The runtime and model as the build recorded them, and the file's hash.

        One model does the whole job, so there is no separate embedding model
        to name. The configuration it ran with, and how its probabilities were
        made into turns, go in ``pipeline``: the same model gives different
        turns under either changed.
        """
        from .nemotron import describe_method
        from .resources import bundled_nemotron, bundled_nemotron_info

        info = bundled_nemotron_info()
        runtime = info.get("runtime")
        if isinstance(runtime, dict) and runtime.get("version"):
            self.engine_version = f"NeMo-Speech.cpp {runtime['version']}"
        model = info.get("model")
        repo = model.get("repo") if isinstance(model, dict) else None
        revision = model.get("revision") if isinstance(model, dict) else None
        name = (
            f"{repo}@{revision}"
            if repo and revision
            else f"{repo} (revision not recorded)"
            if repo
            else "Nemotron 3 Diarization (revision not recorded)"
        )
        self.pipeline = f"{name}; {describe_method()}"
        bundled = bundled_nemotron()
        if bundled is None:
            return
        self.segmentation_model = bundled[1].name
        self.segmentation_model_sha256 = sha256_file_or_none(bundled[1])

    def _describe_sherpa(self) -> None:
        from .resources import bundled_diarization_models, bundled_embedding_model_name

        self.engine_version = _installed_version("sherpa-onnx")
        bundled = bundled_diarization_models()
        if bundled is None:
            return
        segmentation, embedding = bundled
        self.segmentation_model = segmentation.name
        self.segmentation_model_sha256 = sha256_file_or_none(segmentation)
        # The name the build recorded for it (embedding_model.txt), which says
        # what the model is; the file name only says where it is.
        self.embedding_model = bundled_embedding_model_name() or embedding.name
        self.embedding_model_sha256 = sha256_file_or_none(embedding)

    def _describe_pyannote(self) -> None:
        """Name each repository at the commit the bundled cache holds.

        A Hugging Face commit identifies the files exactly, so it stands in for
        a hash here. Where there is no bundled cache - running from source -
        the revision is not known, and the record says so rather than guess.
        """
        from . import hub_cache
        from .resources import pyannote_cache_dir

        self.engine_version = _installed_version("pyannote.audio")
        cache = pyannote_cache_dir()
        hub = cache / "hub" if cache is not None else None

        def at_commit(repo: str) -> str:
            snapshot = hub_cache.resolve(hub, repo) if hub is not None else None
            if snapshot is None:
                return f"{repo} (revision not recorded)"
            return f"{repo}@{snapshot.name}"

        pipeline, segmentation, embedding = hub_cache.PYANNOTE_REPOS
        self.pipeline = at_commit(pipeline)
        self.segmentation_model = at_commit(segmentation)
        self.embedding_model = at_commit(embedding)


def _installed_version(distribution: str) -> str:
    """The installed version of ``distribution``, or "" when it is not known."""
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:  # noqa: BLE001 - absent or unreadable: not known
        return ""


@dataclass
class AnalysisProvenance:
    """The full record: source, build, models and settings for one analysis."""

    source: SourceRecord = field(default_factory=SourceRecord)
    created_utc: str = field(default_factory=_utc_now)
    build: BuildInfo = field(default_factory=build_info)
    runtime: Dict[str, str] = field(default_factory=runtime_summary)
    transcription: Optional[TranscriptionProvenance] = None
    diarization: Optional[DiarizationProvenance] = None
    # The thresholds in force when the analysis was made. It defaulted to the
    # shipped constants, so a report printed 0.62 while 0.75 was configured and
    # was what recognition had actually used.
    thresholds: Thresholds = field(default_factory=active)
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "source": self.source.to_dict(),
            "application": self.build.to_dict(),
            "runtime": dict(self.runtime),
            "transcription": (
                self.transcription.to_dict() if self.transcription else None
            ),
            "diarization": (self.diarization.to_dict() if self.diarization else None),
            "speaker_matching": {
                "acceptance_threshold": self.thresholds.recognition_acceptance,
                "margin_threshold": self.thresholds.recognition_margin,
                "comparison_high": self.thresholds.comparison_high,
                "comparison_intermediate": self.thresholds.comparison_intermediate,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisProvenance":
        application = data.get("application")
        matching = data.get("speaker_matching")
        thresholds = DEFAULTS
        if isinstance(matching, dict):
            thresholds = Thresholds(
                recognition_acceptance=float(
                    matching.get(
                        "acceptance_threshold", DEFAULTS.recognition_acceptance
                    )
                ),
                recognition_margin=float(
                    matching.get("margin_threshold", DEFAULTS.recognition_margin)
                ),
                comparison_high=float(
                    matching.get("comparison_high", DEFAULTS.comparison_high)
                ),
                comparison_intermediate=float(
                    matching.get(
                        "comparison_intermediate", DEFAULTS.comparison_intermediate
                    )
                ),
            )
        transcription = data.get("transcription")
        diarization = data.get("diarization")
        runtime = data.get("runtime")
        return cls(
            source=SourceRecord.from_dict(data.get("source") or {}),
            # Not stamped with the time of reading: a record with no creation
            # time does not know it, and must not be given one.
            created_utc=str(data.get("created_utc") or ""),
            build=(
                BuildInfo(**_build_kwargs(application))
                if isinstance(application, dict)
                else BuildInfo()
            ),
            runtime=(
                {str(k): str(v) for k, v in runtime.items()}
                if isinstance(runtime, dict)
                else {}
            ),
            transcription=(
                TranscriptionProvenance.from_dict(transcription)
                if isinstance(transcription, dict)
                else None
            ),
            diarization=(
                DiarizationProvenance.from_dict(diarization)
                if isinstance(diarization, dict)
                else None
            ),
            thresholds=thresholds,
            schema_version=int(data.get("schema_version") or PROVENANCE_SCHEMA_VERSION),
        )

    @staticmethod
    def _staged_model_lines(diar: "DiarizationProvenance") -> List[str]:
        """Segmentation and embedding models, for the engines that have both."""
        lines = [f"Segmentation model: {diar.segmentation_model or 'not recorded'}"]
        if diar.segmentation_model_sha256:
            lines.append(f"Segmentation SHA-256: {diar.segmentation_model_sha256}")
        lines.append(
            f"Speaker embedding model: {diar.embedding_model or 'not recorded'}"
        )
        if diar.embedding_model_sha256:
            # Always shown when recorded: a report never hides a fact the
            # record holds.
            lines.append(f"Speaker embedding SHA-256: {diar.embedding_model_sha256}")
        elif diar.engine != "pyannote":
            # sherpa is identified by file hash, so a missing one is worth
            # saying; pyannote is identified by the commit above instead.
            lines.append("Speaker embedding SHA-256: not recorded")
        return lines

    def describe(self) -> List[str]:
        """Report-ready lines covering source, build and models."""
        lines = [
            f"Source file: {self.source.filename or 'unknown'}",
            f"Source SHA-256: {self.source.sha256 or 'not recorded'}",
            f"Analysed (UTC): {self.created_utc or 'not recorded'}",
        ]
        lines += self.build.describe()
        if self.transcription:
            lines += [
                f"Transcription model: {self.transcription.model_name}",
                f"Transcription model SHA-256: "
                f"{self.transcription.model_sha256 or 'not recorded'}",
                f"Device / compute: {self.transcription.device} / "
                f"{self.transcription.compute_type}",
                f"Language: {self.transcription.language_setting} "
                f"(detected: {self.transcription.detected_language or 'n/a'})",
            ]
        if self.diarization:
            diar = self.diarization
            version = f" {diar.engine_version}" if diar.engine_version else ""
            lines.append(
                f"Diarization engine: {diar.engine or 'not recorded'}{version}"
            )
            if diar.pipeline:
                lines.append(f"Diarization pipeline: {diar.pipeline}")
            if diar.engine == "nemotron":
                # One end-to-end model: no segmentation or embedding stage.
                lines.append(
                    f"Diarization model: {diar.segmentation_model or 'not recorded'}"
                )
                lines.append(
                    "Diarization model SHA-256: "
                    f"{diar.segmentation_model_sha256 or 'not recorded'}"
                )
            else:
                lines += self._staged_model_lines(diar)
            if diar.voice_leads:
                lines.append(
                    "Names suggested by voice similarity (leads, not "
                    f"identifications): {', '.join(diar.voice_leads)}"
                )
        lines += [
            f"Speaker acceptance threshold: "
            f"{self.thresholds.recognition_acceptance:.2f}",
            f"Speaker margin threshold: {self.thresholds.recognition_margin:.2f}",
        ]
        return lines


def _build_kwargs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a stored application record down to BuildInfo's fields."""
    allowed = {
        "build_id",
        "git_commit",
        "build_timestamp",
        "application_version",
        "platform",
        "python_version",
        "dependencies",
        "models",
    }
    return {key: value for key, value in data.items() if key in allowed}


def transcription_model_sha256(model: PathLike) -> Optional[str]:
    """SHA-256 of a CTranslate2 model's weights, when the path resolves to one.

    Bundled models are directories containing ``model.bin``; a bare size name
    (``small.en``) has no local file to hash and returns ``None`` rather than a
    fabricated value.
    """
    candidate = Path(model)
    if candidate.is_dir():
        weights = candidate / "model.bin"
        if weights.is_file():
            return sha256_file_or_none(weights)
    if candidate.is_file():
        return sha256_file_or_none(candidate)
    return None
