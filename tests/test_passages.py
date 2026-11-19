"""Passages: text grouped by where it sits on the page."""

import pytest

from whispers.passages import _segments, group, ocr_passages


def _line(text, x, y, size=8.0, step=None, rtl=False):
    """Characters of one line from ``x`` at baseline ``y``, as pdfium gives them."""
    step = step or size * 0.6
    chars = []
    for i, ch in enumerate(text):
        # Right to left: the text in reading order, each letter further left.
        left = x + (len(text) - 1 - i if rtl else i) * step
        box = None if ch == " " else (left, y, left + step * 0.9, y + size)
        chars.append((ch, box, size))
    chars.append(("\n", None, 0.0))
    return chars


def _row(*cells, y, size=8.0):
    """One table row: cells at their column x positions, read left to right."""
    chars = []
    for text, x in cells:
        chars += _line(text, x, y, size)[:-1]
    chars.append(("\n", None, 0.0))
    return chars


def test_a_table_row_is_cut_into_its_cells():
    chars = _row(
        ("1", 60), ("3/12/2023", 90), ("Zhang", 175), ("hello there", 270), y=700
    )
    segments = _segments(chars)
    assert [s.text for s in segments] == ["1", "3/12/2023", "Zhang", "hello there"]


def test_a_word_space_does_not_cut_a_line():
    (segment,) = _segments(_line("meet at nine", 100, 700))
    assert segment.text == "meet at nine"


def test_a_wrapped_cell_is_put_back_together_and_rows_stay_apart():
    cjk = 8.0
    chars = (
        _row(("1", 60), ("明天上午九点在老地方见面请把", 270), y=700)
        + _line("那份文件带上。", 270, 700 - 10, step=cjk)
        + _row(("2", 60), ("货已经到了一共三箱", 270), y=670)
    )
    passages = group(_segments(chars), page=0)
    texts = [p.text for p in passages if p.script == "han"]
    assert texts == ["明天上午九点在老地方见面请把那份文件带上。", "货已经到了一共三箱"]
    first = next(p for p in passages if p.text.startswith("明天"))
    assert first.box[1] == pytest.approx(690) and first.box[3] == pytest.approx(708)


def test_a_full_stop_wrapped_alone_joins_its_passage():
    chars = _line("不要告诉任何人", 270, 700) + _line("。", 270, 690)
    (passage,) = group(_segments(chars), page=0)
    assert passage.text == "不要告诉任何人。"


def test_lines_in_another_script_or_column_start_their_own_passage():
    chars = (
        _line("Meeting notes prepared by the officer", 72, 700)
        + _line("会议定于下周三上午九点举行", 72, 690)
        + _line("Read", 400, 680)
    )
    passages = group(_segments(chars), page=0)
    assert [p.script for p in passages] == ["latin", "han", "latin"]


def test_numbers_do_not_gather_the_text_below_them():
    chars = _line("3/12/2023", 90, 700) + _line("10:15", 90, 690)
    passages = group(_segments(chars), page=0)
    assert [p.text for p in passages] == ["3/12/2023", "10:15"]


def test_a_right_to_left_line_is_measured_from_the_right():
    # pdfium orders a mostly right-to-left line from its right end.
    chars = _line("مرحبا بكم في", 300, 700, rtl=True)
    (segment,) = _segments(chars)
    assert segment.text == "مرحبا بكم في"
    assert segment.box[0] == pytest.approx(300)


def test_passages_are_in_reading_order():
    chars = (
        _line("Lower", 72, 600)
        + _line("Upper right", 300, 700)
        + _line("Upper left", 72, 700)
    )
    passages = group(_segments(chars), page=0)
    assert [p.text for p in passages] == ["Upper left", "Upper right", "Lower"]


def test_ocr_boxes_become_pdf_points_from_the_bottom():
    (passage,) = ocr_passages(
        [("会议", (300, 600, 900, 660))], page=2, height=792, dpi=300
    )
    assert passage.page == 2 and passage.script == "han"
    assert passage.box == pytest.approx((72, 792 - 158.4, 216, 792 - 144))
