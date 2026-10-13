import pytest

# whispr.gui imports tkinter (via the package __init__); skip cleanly where the
# Tk runtime isn't available. errors.py itself has no GUI dependency.
errors = pytest.importorskip("whispr.gui.errors")
friendly_error = errors.friendly_error


def test_model_not_in_build_surfaces_message_verbatim():
    exc = RuntimeError(
        "The model 'medium.en' isn't in this build. Open Self-test… to see the "
        "bundled models and pick one of those, or rebuild the bundle with "
        "'medium.en' included."
    )
    out = friendly_error(exc)
    assert out.startswith("The model 'medium.en' isn't in this build.")
    # Not wrapped in the generic "Something went wrong" fallback.
    assert "Something went wrong" not in out


def test_faster_whisper_missing_is_friendly():
    out = friendly_error(RuntimeError("faster-whisper is not installed"))
    assert "transcription engine isn't installed" in out


def test_unknown_error_falls_back():
    out = friendly_error(ValueError("some obscure failure"))
    assert out.startswith("Something went wrong (ValueError):")


# -- Everyday file failures (QA-21) -------------------------------------------


def test_access_denied_names_the_place_and_what_to_do():
    exc = PermissionError(13, "Access is denied", r"\\fileserver\cases\op.whispr.json")
    out = friendly_error(exc)
    assert r"\\fileserver\cases\op.whispr.json" in out
    assert "Choose another folder" in out
    assert "Errno" not in out and "PermissionError" not in out


def test_a_full_disk_says_so():
    import errno

    out = friendly_error(
        OSError(errno.ENOSPC, "No space left on device", "/mnt/usb/a.txt")
    )
    assert out.startswith("The disk is full")
    assert "/mnt/usb/a.txt" in out


def test_other_os_errors_keep_the_fallback():
    out = friendly_error(OSError(5, "Input/output error"))
    assert out.startswith("Something went wrong (OSError):")
