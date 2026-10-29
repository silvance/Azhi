"""Recording conditions: what is set side by side, and what is said about it."""

import re

from azhi.conditions import Side, compare, side


def _piece(**values):
    data = {
        "bandwidth_hz": 7000,
        "level_margin_db": 30.0,
        "level_spread_low": False,
        "clipping_percent": 0.0,
    }
    data.update(values)
    return data


def _matched():
    return Side(
        speech_seconds=40.0,
        pieces=5,
        bandwidth_hz=7000.0,
        bandwidth_pieces=5,
        margin_db=30.0,
        clipping_percent=0.0,
    )


def test_a_side_takes_medians_and_the_worst_clipping():
    measured = side(
        [
            _piece(bandwidth_hz=6800, level_margin_db=20.0),
            _piece(bandwidth_hz=7000, level_margin_db=30.0, clipping_percent=2.5),
            _piece(bandwidth_hz=7200, level_margin_db=40.0),
        ],
        speech_seconds=21.0,
    )
    assert measured.pieces == 3
    assert measured.bandwidth_hz == 7000
    assert measured.margin_db == 30.0
    assert measured.clipping_percent == 2.5


def test_a_margin_with_nothing_to_measure_against_is_left_out():
    """No pause, or pauses of digital silence: neither says how noisy it is."""
    measured = side(
        [
            _piece(level_margin_db=8.0, level_spread_low=True),
            _piece(level_margin_db=0.0),
            _piece(level_margin_db=30.0),
        ],
        speech_seconds=20.0,
    )
    # One measurable piece of three is not enough to speak for the side.
    assert measured.margin_db is None
    assert compare(measured, _matched()).rows()[2][1] == "not measurable"


def test_samples_enrolled_before_the_range_was_measured_are_counted_not_guessed():
    old = {k: v for k, v in _piece().items() if k != "bandwidth_hz"}
    measured = side([old, _piece(bandwidth_hz=7000)], speech_seconds=20.0)
    assert measured.bandwidth_hz == 7000 and measured.bandwidth_pieces == 1
    rows = compare(_matched(), measured).rows()
    assert rows[1][2] == "up to 7.0 kHz (from 1 of 2 samples)"

    none = side([old, old], speech_seconds=20.0)
    result = compare(_matched(), none)
    assert result.rows()[1][2] == "not measured"
    assert any("enrolled before" in note for note in result.notes)


def test_matched_recordings_need_no_notes():
    result = compare(_matched(), _matched())
    assert result.notes == []
    assert result.lines()[0] == "Recording conditions (questioned / reference):"


def test_a_telephone_recording_against_a_full_band_one_is_pointed_out():
    phone = _matched()
    phone.bandwidth_hz = 3750.0
    result = compare(phone, _matched())
    assert result.rows()[1][1] == "up to 3.8 kHz (telephone-like)"
    (note,) = result.notes
    assert "questioned audio reaches only 3.8 kHz against 7.0 kHz" in note
    assert "whoever is speaking" in note


def test_noise_clipping_and_little_speech_are_each_pointed_out():
    questioned = _matched()
    questioned.margin_db = 12.0
    questioned.clipping_percent = 3.0
    questioned.speech_seconds = 6.0
    notes = compare(questioned, _matched()).notes
    assert any("questioned speech stands much less clear" in n for n in notes)
    assert any("questioned audio is clipped (3.0% of samples)" in n for n in notes)
    assert any("Only 6.0 sec of questioned speech" in n for n in notes)


def test_nothing_said_about_the_recordings_reads_as_a_finding_about_the_speaker():
    worst = Side(
        speech_seconds=4.0,
        pieces=1,
        bandwidth_hz=3500.0,
        bandwidth_pieces=1,
        margin_db=5.0,
        clipping_percent=5.0,
    )
    text = "\n".join(compare(worst, _matched()).lines()).lower()
    for word in ("match", "same person", "identity", "identified", "confirmed"):
        assert word not in text
    # The only percentage is a share of clipped samples, never a likelihood.
    assert re.findall(r"\d+(?:\.\d+)?%", text) == ["5.0%", "0.0%", "5.0%"]
