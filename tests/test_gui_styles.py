"""The shipped theme keeps the application's own widget styles (QA-18).

ttkbootstrap rewrites any ``style=`` it has not registered, so 170 labels and
144 frames were built as plain TLabel/TFrame: titles at body size, muted text
not muted, and low-emphasis buttons drawn as the filled primary button. This
builds the real window under ttkbootstrap and checks what each widget got.
Needs a display, so it is skipped on a headless runner.
"""

import os
from collections import Counter
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")
tb = pytest.importorskip("ttkbootstrap")

pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="needs a display",
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import whispr.gui.app as app_mod

    # Build the speaker pages too, without needing the model file.
    monkeypatch.setattr(app_mod, "bundled_embedding_model", lambda: Path("e.onnx"))
    root = tb.Window(themename="darkly")
    try:
        yield app_mod.WhisprApp(root, drag_and_drop=False, bootstrap=True)
    finally:
        root.destroy()


def _styles(widget):
    seen = Counter()
    for child in widget.winfo_children():
        try:
            seen[child.cget("style")] += 1
        except (tk.TclError, AttributeError):
            pass
        seen.update(_styles(child))
    return seen


def test_custom_styles_survive_ttkbootstrap(app):
    from whispr.gui.theme import Style

    seen = _styles(app.root)
    for name in (
        Style.BODY,
        Style.META,
        Style.MUTED,
        Style.SECTION_TITLE,
        Style.PAGE,
        Style.CARD,
        Style.CARD_INNER,
        Style.SUBTLE,
    ):
        assert seen[name] > 0, f"no widget kept {name}"
    # The ones that were stripped to ttkbootstrap's defaults.
    assert seen["TLabel"] == 0
    assert seen["TFrame"] == 0
