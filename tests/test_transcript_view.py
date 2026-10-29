"""Editing the transcript: the view stays put, and lines can be dragged.

Builds the real transcript widget, so it needs a display; skipped without one.
"""

import os
from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter")

pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="needs a display",
)


@pytest.fixture
def view():
    from azhi.gui.theme import init_theme
    from azhi.gui.transcript_view import TranscriptView
    from azhi.transcription import Segment, TranscriptionResult, Word

    root = tk.Tk()
    root.geometry("700x300")
    init_theme(root)
    changes = []
    try:
        view = TranscriptView(
            root, root, tk.BooleanVar(value=True), lambda: changes.append(1)
        )
        view.widget.pack(fill="both", expand=True)
        segments = []
        for n in range(80):
            start = n * 4.0
            words = [
                Word(start + i, start + i + 0.8, f" w{n}_{i}", 0.9) for i in range(3)
            ]
            segments.append(
                Segment(
                    start,
                    start + 3.0,
                    "".join(w.word for w in words).strip(),
                    speaker=f"SPEAKER_0{n % 2}",
                    words=words,
                )
            )
        result = TranscriptionResult(
            text="",
            language="en",
            language_probability=1.0,
            duration=320.0,
            segments=segments,
        )
        view.set_result(result, {})
        _settle(root)
        yield SimpleNamespace(view=view, root=root, result=result, changes=changes)
    finally:
        root.destroy()


def _settle(root):
    for _ in range(5):
        root.update_idletasks()
        root.update()


def _event_at(view, index):
    x, y, w, h = view.widget.bbox(index)
    return SimpleNamespace(x=x + 2, y=y + h // 2)


def test_moving_a_word_keeps_the_view_where_it_was(view):
    view.view.widget.yview_moveto(0.6)
    _settle(view.root)
    before = view.view.widget.yview()[0]
    # A word on a line in view, given to the other speaker.
    top = view.view._segment_at(view.view.widget.index("@0,0 +2 lines"))
    other = (
        "SPEAKER_01"
        if view.result.segments[top].speaker == "SPEAKER_00"
        else "SPEAKER_00"
    )
    view.view._reassign_word_span(top, 1, other, False)
    _settle(view.root)
    after = view.view.widget.yview()[0]
    assert after > 0.4, "the view went back to the top"
    assert abs(after - before) < 0.05
    assert view.changes


def test_the_moved_words_are_marked_for_a_moment(view):
    view.view._reassign_word_span(3, 2, "SPEAKER_00", False)
    _settle(view.root)
    marked = view.view.widget.tag_ranges("moved")
    assert marked
    assert view.view.widget.get(marked[0], marked[1]).strip() == "w3_2"


def test_a_speaker_tag_dragged_onto_another_line_moves_that_line(view):
    w = view.view.widget
    w.yview("line::10.first linestart")
    _settle(view.root)
    assert view.result.segments[10].speaker == "SPEAKER_00"
    start = _event_at(view.view, "line::10.first")
    target = _event_at(view.view, "word::11::1.first")  # Speaker 2's line
    assert view.view._on_drag_press(start) == "break"
    view.view._on_drag_motion(target)
    assert w.tag_ranges("droptarget"), "the line it would land on is shaded"
    view.view._on_drag_release(target)
    _settle(view.root)
    # Line 10 joined the other speaker (merged with its neighbours).
    owner = next(
        seg.speaker
        for seg in view.view._result.segments
        if any(word.start == 40.0 for word in seg.words)
    )
    assert owner == "SPEAKER_01"
    assert not w.tag_ranges("droptarget")


def test_a_click_on_a_speaker_tag_is_still_a_click(view):
    start = _event_at(view.view, "line::2.first")
    view.view._on_drag_press(start)
    view.view._on_drag_motion(SimpleNamespace(x=start.x + 1, y=start.y))
    view.view._on_drag_release(start)
    _settle(view.root)
    assert view.result.segments[2].speaker == "SPEAKER_00"
    assert not view.changes


def test_dropping_on_the_same_speaker_changes_nothing(view):
    start = _event_at(view.view, "line::2.first")
    same = _event_at(view.view, "word::4::0.first")  # also Speaker 1
    view.view._on_drag_press(start)
    view.view._on_drag_motion(same)
    assert not view.view.widget.tag_ranges("droptarget")
    view.view._on_drag_release(same)
    _settle(view.root)
    assert not view.changes


def test_a_real_drag_drops_without_opening_the_speaker_menu(view):
    w = view.view.widget
    w.yview("line::10.first linestart")
    _settle(view.root)
    menus = []
    view.view._popup = lambda menu, _event: menus.append(menu)
    start = _event_at(view.view, "line::10.first")
    target = _event_at(view.view, "word::11::1.first")
    w.focus_force()
    w.event_generate("<Motion>", x=start.x, y=start.y, warp=True)
    w.event_generate("<ButtonPress-1>", x=start.x, y=start.y)
    w.event_generate("<B1-Motion>", x=start.x + 10, y=start.y, state=0x100)
    w.event_generate("<B1-Motion>", x=target.x, y=target.y, state=0x100)
    w.event_generate("<Motion>", x=target.x, y=target.y)
    w.event_generate("<ButtonRelease-1>", x=target.x, y=target.y, state=0x100)
    _settle(view.root)
    assert not menus, "a drop opened a menu as well"
    owner = next(
        seg.speaker
        for seg in view.view._result.segments
        if any(word.start == 40.0 for word in seg.words)
    )
    assert owner == "SPEAKER_01"


def test_a_drag_scrolls_near_an_edge_but_not_where_it_starts(view):
    w = view.view.widget
    w.yview("line::10.first linestart")
    _settle(view.root)
    top = w.yview()[0]
    start = _event_at(view.view, "line::10.first")  # on the top line
    view.view._on_drag_press(start)
    view.view._on_drag_motion(SimpleNamespace(x=start.x + 10, y=start.y))
    _settle(view.root)
    assert w.yview()[0] == top, "scrolled away as soon as the drag began"
    middle = w.winfo_height() // 2
    view.view._on_drag_motion(SimpleNamespace(x=start.x, y=middle))
    view.view._on_drag_motion(SimpleNamespace(x=start.x, y=w.winfo_height() - 2))
    _settle(view.root)
    assert w.yview()[0] > top, "held at the bottom edge, it should scroll down"
    view.view._end_drag()


@pytest.fixture
def playing(view):
    """The same view with a player: records what it was asked to play."""
    played = []
    view.view._on_play = lambda start, end: played.append((start, end))
    view.view.render()
    _settle(view.root)
    view.played = played
    return view


def _wait_click(root):
    import time

    deadline = time.time() + 1.0
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)


def _click(view, index):
    event = _event_at(view.view, index)
    w = view.view.widget
    w.event_generate("<Motion>", x=event.x, y=event.y, warp=True)
    w.event_generate("<ButtonPress-1>", x=event.x, y=event.y)
    w.event_generate("<ButtonRelease-1>", x=event.x, y=event.y, state=0x100)
    return event


def test_a_click_on_a_word_plays_from_it(playing):
    menus = []
    playing.view._popup = lambda menu, _event: menus.append(menu)
    _click(playing, "word::2::1.first")
    _wait_click(playing.root)
    assert playing.played == [(9.0, 11.0)]  # word 1 of line 2, to the line's end
    assert not menus, "a click plays; corrections are on the right button"


def test_a_click_on_a_speaker_tag_plays_the_line(playing):
    _click(playing, "line::3.first")
    _wait_click(playing.root)
    assert playing.played == [(12.0, 15.0)]


def test_a_double_click_selects_without_playing(playing):
    event = _event_at(playing.view, "word::2::1.first")
    w = playing.view.widget
    w.event_generate("<Motion>", x=event.x, y=event.y, warp=True)
    for _ in range(2):  # two clicks in quick succession: a double-click
        w.event_generate("<ButtonPress-1>", x=event.x, y=event.y)
        w.event_generate("<ButtonRelease-1>", x=event.x, y=event.y, state=0x100)
    _wait_click(playing.root)
    assert playing.played == []
    assert w.tag_ranges("sel"), "the double-click selected the word"


def test_right_click_opens_the_corrections(playing):
    menus = []
    playing.view._popup = lambda menu, _event: menus.append(menu)

    def labels(menu):
        return [
            menu.entrycget(i, "label")
            for i in range(menu.index("end") + 1)
            if menu.type(i) == "command"
        ]

    playing.view._context_menu(_event_at(playing.view, "line::2.first"))
    assert any(label.startswith("Rename") for label in labels(menus[-1]))
    playing.view._context_menu(_event_at(playing.view, "word::2::1.first"))
    assert any(label.startswith("Move") for label in labels(menus[-1]))
    assert playing.played == []


def test_without_a_player_a_click_still_opens_the_corrections(view):
    menus = []
    view.view._popup = lambda menu, _event: menus.append(menu)
    _click(view, "word::2::1.first")
    _settle(view.root)
    assert menus


@pytest.fixture
def saving(view):
    """The same view, able to save passages: records what it was asked for."""
    saved = []
    view.view._on_save_clip = lambda start, end: saved.append((start, end))
    menus = []
    view.view._popup = lambda menu, _event: menus.append(menu)
    view.saved, view.menus = saved, menus
    return view


def _choose(menu, label):
    for i in range(menu.index("end") + 1):
        if menu.type(i) == "command" and menu.entrycget(i, "label") == label:
            menu.invoke(i)
            return True
    return False


def test_a_selection_across_lines_saves_as_one_passage(saving):
    w = saving.view.widget
    w.tag_add("sel", "word::2::1.first", "word::4::0.last")
    saving.view._context_menu(_event_at(saving.view, "word::3::1.first"))
    assert _choose(saving.menus[-1], "Save the selection as audio…")
    # From the first selected word's start to the last one's end.
    assert saving.saved == [(9.0, 16.8)]


def test_a_selection_within_a_line_still_offers_the_moves(saving):
    w = saving.view.widget
    w.tag_add("sel", "word::2::0.first", "word::2::1.last")
    saving.view._context_menu(_event_at(saving.view, "word::2::1.first"))
    menu = saving.menus[-1]
    labels = [
        menu.entrycget(i, "label")
        for i in range(menu.index("end") + 1)
        if menu.type(i) == "command"
    ]
    assert labels[0] == "Save the selection as audio…"
    assert any(label.startswith("Move selection") for label in labels)


def test_a_line_saves_from_its_speaker_tag(saving):
    saving.view._context_menu(_event_at(saving.view, "line::3.first"))
    assert _choose(saving.menus[-1], "Save this line as audio…")
    assert saving.saved == [(12.0, 15.0)]


def test_watch_list_hits_are_marked_but_never_in_speaker_tags(view):
    from azhi.watchlist import Matcher

    counts = []
    view.view._on_watch = counts.append
    # "Speaker" is in every tag, and "w3_1" is said once.
    view.view.set_watch(Matcher(["Speaker", "w3_1"]))
    w = view.view.widget
    marked = w.tag_ranges("watch")
    assert [w.get(a, b) for a, b in zip(marked[0::2], marked[1::2])] == ["w3_1"]
    assert counts[-1] == {"w3_1": 1}
    # Shown again after a re-render, as it is for every transcript.
    view.view.render()
    _settle(view.root)
    assert w.tag_ranges("watch")
    # Below the selection and Find's highlight, so neither is hidden.
    assert w.tag_names().index("watch") < w.tag_names().index("sel")


def test_next_hit_steps_through_and_wraps(view):
    from azhi.watchlist import Matcher

    view.view.set_watch(Matcher(["w2_0", "w40_2"]))
    w = view.view.widget
    shown = []
    for _ in range(3):
        assert view.view.next_watch_hit()
        current = w.tag_ranges("watch_current")
        shown.append(w.get(current[0], current[1]))
    assert shown == ["w2_0", "w40_2", "w2_0"]
    view.view.set_watch(None)
    assert not view.view.next_watch_hit()


@pytest.fixture
def noting(view, monkeypatch):
    """The view, with the note dialog answering as the test says."""
    import azhi.gui.transcript_view as tv

    answers = []
    asked = []

    def ask(root, *, line, when, text=""):
        asked.append((line, when, text))
        return answers.pop(0) if answers else None

    monkeypatch.setattr(tv, "ask_note", ask)
    menus = []
    view.view._popup = lambda menu, _event: menus.append(menu)
    view.answers, view.asked, view.menus = answers, asked, menus
    return view


def _labels(menu):
    return [
        menu.entrycget(i, "label")
        for i in range(menu.index("end") + 1)
        if menu.type(i) == "command"
    ]


def test_a_note_is_added_from_the_speaker_tag_and_shown_under_its_line(noting):
    noting.answers.append(("save", "Sounds rehearsed."))
    noting.view._context_menu(_event_at(noting.view, "line::3.first"))
    assert _choose(noting.menus[-1], "Add a note…")
    _settle(noting.root)
    assert noting.asked[0][:2] == ("w3_0 w3_1 w3_2", "00:00:12")
    note = noting.view._result.notes[0]
    assert (note.at, note.text) == (12.0, "Sounds rehearsed.")
    assert note.written_utc  # when it was written, for the record
    w = noting.view.widget
    shown = w.get("note::0.first", "note::0.last")
    assert shown.strip() == "✎ Sounds rehearsed."
    # Directly under line 3, before line 4 starts.
    assert w.compare("note::0.first", ">=", "word::3::2.last")
    assert w.compare("note::0.last", "<", "line::4.first")
    assert noting.changes  # saved like any other edit


def test_a_note_is_changed_by_clicking_it_and_deleted_from_its_menu(noting):
    from azhi.transcription import Note

    noting.view._result.notes.append(Note(12.0, "first thought"))
    noting.view.render()
    _settle(noting.root)
    noting.answers.append(("save", "second thought"))
    noting.view._note_clicked(0)
    _settle(noting.root)
    assert noting.asked[-1][2] == "first thought"
    assert noting.view._result.notes[0].text == "second thought"
    noting.view._context_menu(_event_at(noting.view, "note::0.first +3c"))
    assert _labels(noting.menus[-1]) == ["Change the note…", "Delete the note"]
    assert _choose(noting.menus[-1], "Delete the note")
    _settle(noting.root)
    assert noting.view._result.notes == []
    assert not noting.view.widget.tag_ranges("note")


def test_a_note_stays_with_its_moment_when_the_line_is_split(noting):
    from azhi.transcription import Note

    noting.view._result.notes.append(Note(12.0, "about the start"))
    other = (
        "SPEAKER_00"
        if noting.view._result.segments[3].speaker == "SPEAKER_01"
        else "SPEAKER_01"
    )
    noting.view._reassign_word_span(3, 2, other, False)  # split off the last word
    _settle(noting.root)
    w = noting.view.widget
    # Still after the word it was written at, not after the moved one.
    assert w.compare("note::0.first", ">=", "word::3::1.last")
    # ...and before the word that moved away to the other speaker.
    moved = next(
        tag
        for tag in w.tag_names()
        if tag.startswith("word::")
        and w.tag_ranges(tag)
        and w.get(f"{tag}.first", f"{tag}.last").strip() == "w3_2"
    )
    assert w.compare("note::0.last", "<=", f"{moved}.first")


def test_watch_list_hits_are_never_inside_a_note(noting):
    from azhi.transcription import Note
    from azhi.watchlist import Matcher

    noting.view._result.notes.append(Note(12.0, "w3_1 again"))
    noting.view.render()
    _settle(noting.root)
    noting.view.set_watch(Matcher(["w3_1"]))
    w = noting.view.widget
    marked = w.tag_ranges("watch")
    assert len(marked) == 2  # the word said, not the note about it
    assert "note" not in w.tag_names(marked[0])
