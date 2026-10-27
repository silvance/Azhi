"""Locate bundled, offline resources (ffmpeg binary and Whisper models).

This lets the GUI run on an air-gapped machine with no network access. It works
both when running from source and when frozen with PyInstaller (one-dir or
one-file), and an explicit override directory can be supplied via the
``WHISPERS_ASSETS`` environment variable.

Expected layout of an assets directory::

    whispers_assets/
        ffmpeg/ffmpeg            (or ffmpeg.exe on Windows)
        models/
            small/model.bin ...
            medium/model.bin ...
            large-v3/model.bin ...
        nemotron/                (optional speaker separation)
            bin/ or lib/         (the runtime's C library and what it loads)
            Nemotron-3-Diarization.q8_0.gguf
            nemotron.json
        alignment/               (optional English word timings)
            model.onnx
            vocab.json
            alignment.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from shutil import which
from typing import Dict, List, Optional, Set, Tuple

ASSETS_DIRNAME = "whispers_assets"
ENV_ASSETS = "WHISPERS_ASSETS"


def _candidate_asset_dirs() -> List[Path]:
    """Ordered candidate ``whispers_assets`` directories (existing or not)."""
    candidates: List[Path] = []

    override = os.environ.get(ENV_ASSETS)
    if override:
        candidates.append(Path(override))

    # PyInstaller one-file: resources are unpacked under sys._MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / ASSETS_DIRNAME)

    # PyInstaller one-dir: resources sit next to the executable.
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / ASSETS_DIRNAME)

    # Running from source: repo root (parent of the whispers package).
    candidates.append(Path(__file__).resolve().parent.parent / ASSETS_DIRNAME)

    # Last resort: current working directory.
    candidates.append(Path.cwd() / ASSETS_DIRNAME)

    return candidates


def asset_dirs() -> List[Path]:
    """Existing assets directories, de-duplicated and in priority order."""
    out: List[Path] = []
    seen = set()
    for candidate in _candidate_asset_dirs():
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            out.append(resolved)
    return out


def find_bundled_ffmpeg() -> Optional[Path]:
    """Return the bundled ffmpeg executable, if one is shipped in the assets."""
    names = ("ffmpeg.exe", "ffmpeg") if os.name == "nt" else ("ffmpeg",)
    for base in asset_dirs():
        for directory in (base / "ffmpeg", base):
            for name in names:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
    return None


def find_ffmpeg() -> Optional[Path]:
    """Return the ffmpeg to use: a bundled one if present, else one on PATH."""
    bundled = find_bundled_ffmpeg()
    if bundled is not None:
        return bundled
    on_path = which("ffmpeg")
    return Path(on_path) if on_path else None


def _frozen_roots() -> List[Path]:
    """Bundle root dir(s) when frozen with PyInstaller (one-file / one-dir).

    PyInstaller reclassifies executables/shared libraries shipped as data into the
    binary set, which can land at the bundle root rather than under
    ``whispers_assets/...``. These roots let us still find such a relocated binary.
    """
    roots: List[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    return roots


def find_bundled_tesseract() -> Optional[Path]:
    """Return the bundled Tesseract OCR executable, if shipped in the assets."""
    names = ("tesseract.exe", "tesseract") if os.name == "nt" else ("tesseract",)
    search_dirs: List[Path] = []
    for base in asset_dirs():
        search_dirs += [base / "tesseract", base]
    # Also the frozen bundle root(s): PyInstaller may relocate the .exe there.
    for root in _frozen_roots():
        search_dirs += [root / "tesseract", root]
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def find_tesseract() -> Optional[Path]:
    """Return the Tesseract to use: a bundled one if present, else one on PATH."""
    bundled = find_bundled_tesseract()
    if bundled is not None:
        return bundled
    on_path = which("tesseract")
    return Path(on_path) if on_path else None


def bundled_tessdata_dir() -> Optional[Path]:
    """Return the bundled Tesseract ``tessdata`` directory, if present.

    The build populates ``whispers_assets/tesseract/tessdata`` with the
    ``<lang>.traineddata`` files; at runtime OCR points Tesseract at it.
    """
    for base in asset_dirs():
        directory = base / "tesseract" / "tessdata"
        if directory.is_dir():
            return directory
    return None


def configure_offline_ocr() -> Optional[Path]:
    """Point Tesseract at the bundled tessdata and co-located libraries, offline.

    Returns the bundled tesseract directory, or ``None`` if none is bundled
    (so running from a system Tesseract is unchanged). Sets ``TESSDATA_PREFIX``
    and prepends the bundle's tesseract directory to the OS library search path
    (``PATH`` on Windows, ``LD_LIBRARY_PATH`` on Linux) so the binary we spawn
    finds the shared libraries shipped beside it.
    """
    tessdata = bundled_tessdata_dir()
    if tessdata is None:
        return None
    os.environ["TESSDATA_PREFIX"] = str(tessdata)
    tess_dir = tessdata.parent
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    existing = os.environ.get(var, "").split(os.pathsep)
    # Put the tesseract folder, the actual binary's folder, and the bundle root(s)
    # on the library search path. PyInstaller can scatter the binary and its DLLs
    # between whispers_assets/tesseract and the bundle root, so cover both.
    candidates: List[Path] = [tess_dir]
    binary = find_bundled_tesseract()
    if binary is not None:
        candidates.append(binary.parent)
    candidates += _frozen_roots()
    new_entries = [str(p) for p in candidates if str(p) not in existing]
    if new_entries:
        os.environ[var] = os.pathsep.join(new_entries + existing).strip(os.pathsep)
    return tess_dir


def bundled_models() -> Dict[str, Path]:
    """Map of bundled model name -> directory (a CTranslate2 model folder).

    A directory is considered a model if it contains a ``model.bin`` file.
    """
    models: Dict[str, Path] = {}
    for base in asset_dirs():
        models_dir = base / "models"
        if not models_dir.is_dir():
            continue
        for child in sorted(models_dir.iterdir()):
            if child.is_dir() and (child / "model.bin").is_file():
                models.setdefault(child.name, child)
    return models


def bundled_diarization_models() -> Optional[Tuple[Path, Path]]:
    """Return ``(segmentation.onnx, embedding.onnx)`` if both are bundled."""
    for base in asset_dirs():
        directory = base / "diarization"
        segmentation = directory / "segmentation.onnx"
        embedding = directory / "embedding.onnx"
        if segmentation.is_file() and embedding.is_file():
            return segmentation, embedding
    return None


# NVIDIA's end-to-end diarizer and the runtime that runs it, both optional.
NEMOTRON_DIRNAME = "nemotron"
NEMOTRON_MODEL_FILENAME = "Nemotron-3-Diarization.q8_0.gguf"


def bundled_nemotron() -> Optional[Tuple[Path, Path]]:
    """Return ``(runtime C library, model)`` if this build ships both.

    Kept in one folder, ``whispers_assets/nemotron``: the library loads the
    others it needs from beside itself, and the build copies the folder as it
    is. Windows keeps the libraries in ``bin``, Linux in ``lib``.
    """
    library = (
        Path("bin") / "nemo_speech_asr_c.dll"
        if os.name == "nt"
        else Path("lib") / "libnemo_speech_asr_c.so.1"
    )
    for base in asset_dirs():
        directory = base / NEMOTRON_DIRNAME
        model = directory / NEMOTRON_MODEL_FILENAME
        if model.is_file() and (directory / library).is_file():
            return directory / library, model
    return None


def bundled_nemotron_info() -> Dict[str, object]:
    """What the build recorded about the Nemotron files (``nemotron.json``).

    Runtime version and archive digest, model repository, revision, digest and
    licence, as downloaded. ``{}`` when nothing was recorded; nothing is guessed.
    """
    for base in asset_dirs():
        note = base / NEMOTRON_DIRNAME / "nemotron.json"
        if note.is_file():
            try:
                data = json.loads(note.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}
    return {}


ALIGNMENT_DIRNAME = "alignment"


def bundled_aligner() -> Optional[Path]:
    """The English word-timing model's folder (``model.onnx``, ``vocab.json``)."""
    for base in asset_dirs():
        directory = base / ALIGNMENT_DIRNAME
        if (directory / "model.onnx").is_file() and (
            directory / "vocab.json"
        ).is_file():
            return directory
    return None


def bundled_aligner_info() -> Dict[str, object]:
    """What the build recorded about the word-timing model (``alignment.json``).

    Repository, revision, file, digest and licence, as downloaded; ``{}`` when
    nothing was recorded.
    """
    for base in asset_dirs():
        note = base / ALIGNMENT_DIRNAME / "alignment.json"
        if note.is_file():
            try:
                data = json.loads(note.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}
    return {}


def bundled_embedding_model() -> Optional[Path]:
    """Return the bundled speaker-embedding model (for voiceprints), if present.

    Independent of the segmentation model, so voiceprints and speaker comparison
    work even on a pyannote-only build that ships just the embedding model.
    """
    for base in asset_dirs():
        embedding = base / "diarization" / "embedding.onnx"
        if embedding.is_file():
            return embedding
    return None


def bundled_denoiser_model() -> Optional[Path]:
    """Return the bundled noise-reduction model, if this build ships one.

    Optional, like every other model here: a build without it simply cannot
    offer the option, and says so rather than reaching for the network.
    """
    for base in asset_dirs():
        model = base / "enhancement" / "denoiser.onnx"
        if model.is_file():
            return model
    return None


def bundled_embedding_model_name() -> Optional[str]:
    """Name of the bundled speaker-embedding model, if recorded at build time.

    The build writes it to ``diarization/embedding_model.txt`` so the app can show
    which model a bundle uses - two builds must share it to compare voiceprints.
    """
    for base in asset_dirs():
        note = base / "diarization" / "embedding_model.txt"
        if note.is_file():
            try:
                text = note.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            return text or None
    return None


def bundled_embedding_model_info() -> Dict[str, object]:
    """What is known about the bundled speaker-embedding model.

    The build writes ``diarization/embedding_model.json`` describing the exact
    file that was downloaded - architecture, the data it was trained on, source
    URL and SHA-256 - so the app can name the model precisely instead of applying
    a broad label to it. Older bundles have only the name sidecar, and bundles
    built from a direct URL know only the file name; whatever is missing is left
    out rather than guessed. Returns ``{}`` when nothing is recorded.
    """
    for base in asset_dirs():
        note = base / "diarization" / "embedding_model.json"
        if note.is_file():
            try:
                data = json.loads(note.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}
    name = bundled_embedding_model_name()
    return {"name": name} if name else {}


def describe_bundled_embedding_model() -> Optional[str]:
    """A one-line description of the embedding model, for the UI and reports.

    Names the architecture and what the file was trained on when the build
    recorded them, e.g. ``CAM++ (3D-Speaker, zh-cn 16k common) - trained on
    Mandarin (zh-cn) 16 kHz common-domain data``. Falls back to the bare alias.
    """
    info = bundled_embedding_model_info()
    if not info:
        return None
    label = str(info.get("label") or info.get("name") or "").strip()
    if not label:
        return None
    trained = str(info.get("training_data") or "").strip()
    return f"{label} - trained on {trained}" if trained else label


def pyannote_cache_dir() -> Optional[Path]:
    """Return a bundled offline Hugging Face cache for pyannote, if present.

    The build copies the downloaded pyannote models into ``whispers_assets/pyannote``
    (an HF cache layout); at runtime this is used with ``HF_HUB_OFFLINE`` so the
    air-gapped app needs no network or token.
    """
    for base in asset_dirs():
        directory = base / "pyannote"
        if directory.is_dir():
            return directory
    return None


def is_bundle() -> bool:
    """Whether this is a shipped bundle, which must never fetch anything.

    A PyInstaller build is what goes to an air-gapped machine, so every path in
    one that could reach the network is closed, rather than each relying on
    there being no network to reach. QA found three that were not: a model name
    the bundle lacked went to faster-whisper as a download, the Live page passed
    any typed name the same way, and "auto" speaker separation reached for the
    Hugging Face Hub whenever the pyannote cache was missing.

    Running from source keeps the developer conveniences - a model fetched on
    first use, a pyannote pipeline from the Hub - because that is not deployed.
    """
    return bool(getattr(sys, "frozen", False))


# Every switch the Hugging Face libraries read to stay off the network.
_OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")


class NetworkRefused(OSError):
    """A download was attempted in a copy that must never use the network."""


def refuse_network_downloads() -> None:
    """Make every ``urllib`` download in this process fail, plainly, instead.

    The backstop under the switches below. Argos Translate's sentence splitter
    (MiniSBD) fetches a model from GitHub with ``urllib.request.urlretrieve``
    whenever the one it wants is not in its cache - and no environment switch
    turns that off. The application itself makes no ``urllib`` requests, so in
    a shipped copy any that happens is a library reaching for something the
    build did not include, and it is stopped here with a sentence that says so.
    """
    import urllib.request

    def refuse(self, request):  # noqa: ANN001 - urllib handler signature
        raise NetworkRefused(
            "This copy tried to download "
            f"{request.full_url} and was stopped: a shipped copy never uses "
            "the network. A component it needed was not included when this "
            "copy was built; it needs rebuilding."
        )

    handlers = [
        type(f"Refuse{base.__name__}", (base,), {f"{scheme}_open": refuse})()
        for base, scheme in (
            (urllib.request.HTTPHandler, "http"),
            (urllib.request.HTTPSHandler, "https"),
            (urllib.request.FTPHandler, "ftp"),
        )
    ]
    urllib.request.install_opener(urllib.request.build_opener(*handlers))


def enforce_offline() -> bool:
    """In a bundle, close every route a library could take to the network.

    Set unconditionally, not only when a pyannote cache is bundled: a bundle
    missing that cache is exactly the one that would otherwise go looking for
    it. Must run before huggingface_hub or argostranslate is imported - both
    read these into module state at import. Returns whether it did anything.

    Argos reads its settings from the environment first and from a
    settings.json under the user's config folder second, so setting the model
    provider here pins it however that file has been edited: a provider of
    LIBRETRANSLATE or OPENAI would send the text being translated to a server.
    """
    if not is_bundle():
        return False
    for flag in _OFFLINE_FLAGS:
        os.environ[flag] = "1"
    os.environ["ARGOS_MODEL_PROVIDER"] = "OPENNMT"
    os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
    refuse_network_downloads()
    return True


def configure_offline_hf_cache() -> Optional[Path]:
    """Point Hugging Face at the bundled pyannote cache, in offline mode.

    Returns the configured cache directory, or ``None`` if none is bundled.

    This MUST run at process startup, before ``huggingface_hub`` (or anything that
    imports it - ``faster_whisper`` and ``pyannote.audio`` both do) is first
    imported: huggingface_hub reads ``HF_HOME`` / ``HF_HUB_OFFLINE`` into module
    constants at import time, so setting them later (e.g. inside the diarization
    call) has no effect and the app falls back to the network.

    No-op when no bundled cache is present (e.g. running from source), so normal
    online Hugging Face behaviour is unchanged.

    When a bundle IS present these are set unconditionally (not setdefault): the
    air-gapped app must use its own cache regardless of any HF_HOME / HF_HUB_CACHE
    / HF_HUB_OFFLINE the operator's machine happens to have set, which would
    otherwise point Hugging Face at the wrong (empty) cache in offline mode.
    """
    enforce_offline()
    cache = pyannote_cache_dir()
    if cache is None:
        return None
    hub = str(cache / "hub")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = hub
    os.environ["HF_HUB_OFFLINE"] = "1"
    # pyannote.audio loads its sub-models (segmentation + speaker embedding) via
    # Model.from_pretrained, whose cache_dir defaults to PYANNOTE_CACHE (NOT
    # HF_HUB_CACHE) - and that default is captured into a module constant at
    # import time. Point it at the bundled cache here, before pyannote is imported,
    # so the sub-models resolve from the bundle too.
    os.environ["PYANNOTE_CACHE"] = hub
    return cache


def bundled_argos_data_dir() -> Optional[Path]:
    """Return the bundled Argos Translate data dir, if present.

    The build populates ``whispers_assets/argos/argos-translate`` (Argos's data dir,
    holding ``packages/`` and the ``minisbd/`` sentence-splitter cache). Argos
    derives all of these from ``XDG_DATA_HOME``, so the parent is used at runtime.
    """
    for base in asset_dirs():
        directory = base / "argos" / "argos-translate"
        if directory.is_dir():
            return directory
    return None


def bundled_sentence_splitters() -> List[str]:
    """The offline sentence-splitter models this build holds for translation.

    Argos uses a language pack's own ``minisbd/*.onnx`` if it ships one, and
    otherwise the MiniSBD cache beside the packs - which it fills by download.
    A build whose warm-up failed holds packs but no splitter, and would have
    tried that download the first time an operator translated. Names are the
    model files' stems: ``de``, ``es`` and so on.
    """
    data = bundled_argos_data_dir()
    if data is None:
        return []
    found: Set[str] = set()
    for folder in [data / "minisbd", *sorted((data / "packages").glob("*/minisbd"))]:
        if folder.is_dir():
            found.update(path.stem for path in folder.glob("*.onnx"))
    return sorted(found)


def configure_offline_translation() -> Optional[Path]:
    """Point Argos Translate at the bundled data dir, offline.

    Returns the data dir, or ``None`` if none is bundled. Like the HF config this
    MUST run at startup before ``argostranslate`` is imported (it reads
    ``XDG_DATA_HOME`` / ``ARGOS_CHUNK_TYPE`` into module state at import). No-op
    when no bundle is present, so from-source use is unchanged.

    Argos resolves its data dir as ``XDG_DATA_HOME/argos-translate`` and puts the
    language packs and the MiniSBD model cache under it, so pointing XDG_DATA_HOME
    at the bundle wires up both. ``ARGOS_CHUNK_TYPE=MINISBD`` forces the lightweight
    sentence splitter whose models we pre-cache at build time (no stanza models, no
    runtime downloads).
    """
    data = bundled_argos_data_dir()
    if data is None:
        return None
    os.environ["XDG_DATA_HOME"] = str(data.parent)
    os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
    return data
