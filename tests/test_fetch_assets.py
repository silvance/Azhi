import importlib.util
from pathlib import Path

import pytest

# fetch_assets.py is a build script under packaging/ (not an installed module);
# load it by path. Its heavy imports are all inside functions, so this is cheap.
_PATH = Path(__file__).resolve().parent.parent / "packaging" / "fetch_assets.py"
_spec = importlib.util.spec_from_file_location("fetch_assets", _PATH)
fetch_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_assets)


def test_every_model_the_application_offers_can_be_bundled():
    """The dropdown and the bundler must agree on the model names.

    They are two lists in two files that have to stay in step, and they drifted:
    the application offered "turbo" while the bundler had no repository for it,
    so a release build that asked for it failed - after downloading every model
    named before it.
    """
    from sotto.transcription import MODEL_SIZES

    missing = [name for name in MODEL_SIZES if name not in fetch_assets.MODEL_REPOS]
    assert not missing, (
        f"{missing} can be chosen in the application but cannot be bundled; "
        "add them to MODEL_REPOS in packaging/fetch_assets.py"
    )


def test_unknown_model_is_refused_before_anything_is_downloaded(monkeypatch):
    """A bad name fails in seconds, not after gigabytes."""
    downloads = []

    def _never(*args, **kwargs):
        downloads.append(kwargs.get("repo_id"))

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("m", (), {"snapshot_download": _never}),
    )
    with pytest.raises(SystemExit) as excinfo:
        fetch_assets.fetch_models(["base.en", "small", "not-a-model"])
    assert "not-a-model" in str(excinfo.value)
    assert downloads == []


def test_turbo_resolves_to_the_repository_faster_whisper_itself_uses():
    """A bundled model must be the weights a non-bundled run would have fetched."""
    assert (
        fetch_assets.MODEL_REPOS["turbo"]
        == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    )


def test_resolve_embedding_alias():
    name, url = fetch_assets._resolve_embedding("titanet-large")
    assert name == "titanet-large"
    assert url.endswith("nemo_en_titanet_large.onnx")


def test_resolve_embedding_zh_cn_alias():
    name, url = fetch_assets._resolve_embedding("campplus")
    assert name == "campplus"
    assert url.startswith(fetch_assets.EMBEDDING_RELEASE_BASE)
    assert url.endswith(".onnx")


def test_resolve_embedding_direct_url():
    url_in = "https://example.com/models/my_custom_embed.onnx"
    name, url = fetch_assets._resolve_embedding(url_in)
    assert url == url_in
    assert name == "my_custom_embed"


def test_resolve_embedding_unknown_raises():
    with pytest.raises(SystemExit):
        fetch_assets._resolve_embedding("not-a-real-model")


def test_default_embedding_is_a_known_alias():
    assert fetch_assets.DEFAULT_EMBEDDING in fetch_assets.EMBEDDING_MODELS


def test_embedding_metadata_records_what_the_file_actually_is():
    name, url = fetch_assets._resolve_embedding("campplus")
    meta = fetch_assets.embedding_metadata(name, url)
    # The published CAM++/ERes2Net assets are the zh-cn checkpoints; record that
    # rather than calling them broadly multilingual.
    assert "zh-cn" in meta["label"]
    assert "Mandarin" in meta["training_data"]
    assert meta["filename"].endswith(".onnx")
    assert meta["source_url"] == url
    assert "varies with language" in meta["caveat"]


def test_no_embedding_alias_claims_to_be_multilingual():
    for entry in fetch_assets.EMBEDDING_MODELS.values():
        blob = " ".join(str(v) for v in entry.values()).lower()
        assert "multilingual" not in blob


def test_embedding_metadata_invents_nothing_for_a_direct_url():
    name, url = fetch_assets._resolve_embedding("https://example.invalid/custom.onnx")
    meta = fetch_assets.embedding_metadata(name, url)
    assert meta["name"] == "custom"
    # Nothing is known about an arbitrary file beyond its name and where it came
    # from, so the descriptive fields stay empty.
    assert meta["architecture"] == ""
    assert meta["training_data"] == ""
    assert "sha256" not in meta


def test_embedding_metadata_hashes_the_downloaded_file(tmp_path):
    model = tmp_path / "embedding.onnx"
    model.write_bytes(b"weights")
    name, url = fetch_assets._resolve_embedding("titanet-large")
    meta = fetch_assets.embedding_metadata(name, url, model)
    from sotto.hashing import sha256_file

    assert meta["sha256"] == sha256_file(model)
    assert meta["size"] == len(b"weights")


# -- Nemotron: runtime archive + model, both checked against pinned digests ----


def _runtime_archive(path, platform):
    """A small stand-in for NVIDIA's release archive, laid out as theirs is."""
    import io
    import os
    import tarfile
    import zipfile

    root = f"nemo-speech-0.2.0-{platform}-x86_64-cpu"
    files = {
        f"{root}/bin/nemo-speech"
        + (".exe" if platform == "windows" else ""): b"program",
        f"{root}/lib/libggml.so.0.25.1": b"library",
        f"{root}/lib/cmake/NeMoSpeech/NeMoSpeechConfig.cmake": b"cmake",
        f"{root}/share/licenses/nemo-speech/LICENSE": b"Apache-2.0",
        f"{root}/share/doc/nemo-speech/README.md": b"docs",
    }
    if platform == "windows":
        with zipfile.ZipFile(path, "w") as zipped:
            for name, data in files.items():
                zipped.writestr(name, data)
        return
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"{root}/lib/libggml.so.0")
        link.type = tarfile.SYMTYPE
        link.linkname = "libggml.so.0.25.1"
        tar.addfile(link)
    assert os.path.getsize(path)


@pytest.fixture
def nemotron_fetch(tmp_path, monkeypatch):
    import sys
    import types

    from sotto import asset_lock
    from sotto.hashing import sha256_file

    platform = "windows" if __import__("os").name == "nt" else "linux"
    archive_name, _ = fetch_assets.NEMO_SPEECH_ARCHIVES[platform]
    source = tmp_path / "source" / archive_name
    source.parent.mkdir()
    _runtime_archive(source, platform)
    model_bytes = b"gguf-weights"
    model_source = tmp_path / "source" / "model.gguf"
    model_source.write_bytes(model_bytes)

    monkeypatch.setattr(fetch_assets, "ASSETS", tmp_path / "sotto_assets")
    monkeypatch.setattr(fetch_assets, "LOCK", asset_lock.empty())
    monkeypatch.setattr(fetch_assets, "save_lock", lambda: None)
    monkeypatch.setitem(
        fetch_assets.NEMO_SPEECH_ARCHIVES, platform, (archive_name, sha256_file(source))
    )
    monkeypatch.setattr(fetch_assets, "NEMOTRON_SHA256", sha256_file(model_source))
    monkeypatch.setattr(fetch_assets, "NEMOTRON_SIZE", len(model_bytes))
    monkeypatch.setattr(
        fetch_assets.urllib.request,
        "urlretrieve",
        lambda url, dest: __import__("shutil").copyfile(source, dest),
    )

    def hf_hub_download(repo_id, filename, revision, local_dir):
        assert (repo_id, revision) == (
            fetch_assets.NEMOTRON_REPO,
            fetch_assets.NEMOTRON_REVISION,
        )
        target = __import__("pathlib").Path(local_dir) / filename
        if filename == fetch_assets.NEMOTRON_CARD:
            target.write_text(
                "## License/Terms of Use\nGoverned by OpenMDW 1.1.\n", encoding="utf-8"
            )
        else:
            target.write_bytes(model_bytes)
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    return {"platform": platform, "out": tmp_path / "sotto_assets" / "nemotron"}


def test_nemotron_bundle_holds_program_libraries_licence_model_and_record(
    nemotron_fetch,
):
    import json

    fetch_assets.fetch_nemotron()
    out = nemotron_fetch["out"]
    linux = nemotron_fetch["platform"] == "linux"
    assert (out / "bin" / ("nemo-speech" if linux else "nemo-speech.exe")).is_file()
    assert (out / "licenses" / "nemo-speech" / "LICENSE").is_file()
    assert (out / fetch_assets.NEMOTRON_FILENAME).is_file()
    assert not (out / "doc").exists() and not (out / "lib" / "cmake").exists()
    if linux:
        # A link arrives as a plain file: not every tool that copies a bundle
        # keeps links.
        library = out / "lib" / "libggml.so.0"
        assert library.is_file() and not library.is_symlink()
    record = json.loads((out / "nemotron.json").read_text(encoding="utf-8"))
    assert record["runtime"]["version"] == fetch_assets.NEMO_SPEECH_VERSION
    assert record["model"]["revision"] == fetch_assets.NEMOTRON_REVISION
    assert record["model"]["licence"] == "OpenMDW-1.1"
    assert fetch_assets.LOCK["huggingface"][fetch_assets.NEMOTRON_REPO] == (
        fetch_assets.NEMOTRON_REVISION
    )


def test_nemotron_bundle_keeps_what_its_licence_asks_a_copy_to_keep(nemotron_fetch):
    """OpenMDW-1.1: a copy keeps the agreement and the notices of origin."""
    import json

    fetch_assets.fetch_nemotron()
    out = nemotron_fetch["out"]
    text = (out / "licenses" / "OpenMDW-1.1.txt").read_text(encoding="utf-8")
    assert text.startswith("OpenMDW License Agreement, version 1.1")
    assert "you shall retain in your\ndistribution (1) a copy of this agreement" in text
    card = out / fetch_assets.NEMOTRON_CARD_COPY
    assert "OpenMDW 1.1" in card.read_text(encoding="utf-8")
    record = json.loads((out / "nemotron.json").read_text(encoding="utf-8"))
    assert record["model"]["licence_text"] == "licenses/OpenMDW-1.1.txt"
    assert record["model"]["model_card"] == card.name
    assert len(record["model"]["model_card_sha256"]) == 64
    assert not (out / fetch_assets.NEMOTRON_CARD).exists()


def test_the_notices_reproduce_the_openmdw_agreement_as_shipped():
    """The notices beside the program carry the same text as the bundle's copy."""
    root = Path(__file__).resolve().parent.parent
    agreement = fetch_assets.OPENMDW_TEXT.read_text(encoding="utf-8")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert agreement.strip() in notices


def test_nemotron_runtime_with_another_digest_is_not_bundled(
    nemotron_fetch, monkeypatch
):
    platform = nemotron_fetch["platform"]
    name, _ = fetch_assets.NEMO_SPEECH_ARCHIVES[platform]
    monkeypatch.setitem(fetch_assets.NEMO_SPEECH_ARCHIVES, platform, (name, "0" * 64))
    with pytest.raises(SystemExit, match="does not match its pinned digest"):
        fetch_assets.fetch_nemotron()
    assert not (nemotron_fetch["out"] / "bin").exists()


def test_nemotron_model_with_another_digest_is_not_bundled(nemotron_fetch, monkeypatch):
    monkeypatch.setattr(fetch_assets, "NEMOTRON_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="does not match its pinned digest"):
        fetch_assets.fetch_nemotron()


@pytest.fixture
def alignment_fetch(tmp_path, monkeypatch):
    import sys
    import types
    from pathlib import Path

    from sotto import asset_lock

    weights = b"onnx-weights"
    calls = []
    monkeypatch.setattr(fetch_assets, "ASSETS", tmp_path / "sotto_assets")
    monkeypatch.setattr(fetch_assets, "LOCK", asset_lock.empty())
    monkeypatch.setattr(fetch_assets, "save_lock", lambda: None)
    monkeypatch.setattr(fetch_assets, "ALIGNMENT_SIZE", len(weights))
    monkeypatch.setattr(fetch_assets, "_hub_revision", lambda repo: "head123")

    def hf_hub_download(repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(weights if filename.endswith(".onnx") else b"{}")
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    return {"out": tmp_path / "sotto_assets" / "alignment", "calls": calls}


def test_alignment_bundle_holds_model_vocabulary_and_record(alignment_fetch):
    import json

    from sotto.hashing import sha256_bytes

    fetch_assets.fetch_alignment()
    out = alignment_fetch["out"]
    assert (out / "model.onnx").read_bytes() == b"onnx-weights"
    assert (out / "vocab.json").is_file()
    record = json.loads((out / "alignment.json").read_text(encoding="utf-8"))
    assert record["model"]["revision"] == "head123"
    assert record["model"]["sha256"] == sha256_bytes(b"onnx-weights")
    assert record["model"]["licence"] == "Apache-2.0"
    # The revision the build resolved is what every later build gets.
    assert fetch_assets.LOCK["huggingface"][fetch_assets.ALIGNMENT_REPO] == "head123"


def test_alignment_follows_the_pinned_revision(alignment_fetch):
    fetch_assets.LOCK["huggingface"][fetch_assets.ALIGNMENT_REPO] = "pinned456"
    fetch_assets.fetch_alignment()
    assert {revision for _repo, _file, revision in alignment_fetch["calls"]} == {
        "pinned456"
    }


def test_an_alignment_model_of_another_size_is_not_bundled(
    alignment_fetch, monkeypatch
):
    monkeypatch.setattr(fetch_assets, "ALIGNMENT_SIZE", 1)
    with pytest.raises(SystemExit, match="is not 1 bytes"):
        fetch_assets.fetch_alignment()
    assert fetch_assets.ALIGNMENT_REPO not in fetch_assets.LOCK["huggingface"]
