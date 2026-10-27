import csv

import pytest

from whispr import diarization_eval as ev
from whispr.diarization import SpeakerSegment
from whispr.transcription import Segment, Word


def turn(start, end, speaker):
    return SpeakerSegment(start=start, end=end, speaker=speaker)


# -- reference files ------------------------------------------------------------


def test_reads_audacity_labels(tmp_path):
    labels = tmp_path / "call.txt"
    labels.write_text(
        "0.500000\t2.250000\tAlice\n"
        "\\\t200.0\t3000.0\n"  # frequency row Audacity adds; no time span
        "2,400000\t2,900000\tBob\n"  # comma decimal separator
        "3.000000\t3.000000\tnote\n",  # point label: a moment, not a turn
        encoding="utf-8",
    )
    assert ev.read_reference(labels) == [
        turn(0.5, 2.25, "Alice"),
        turn(2.4, 2.9, "Bob"),
    ]


def test_reads_rttm(tmp_path):
    rttm = tmp_path / "call.rttm"
    rttm.write_text(
        "SPEAKER call 1 1.00 0.50 <NA> <NA> B <NA> <NA>\n"
        "SPKR-INFO call 1 <NA> <NA> <NA> unknown B <NA> <NA>\n"
        "SPEAKER call 1 0.00 1.00 <NA> <NA> A <NA> <NA>\n"
    )
    assert ev.read_reference(rttm) == [turn(0.0, 1.0, "A"), turn(1.0, 1.5, "B")]


def test_unreadable_reference_names_the_line(tmp_path):
    labels = tmp_path / "call.txt"
    labels.write_text("0.5\t1.0\tAlice\nsoon\t2.0\tBob\n")
    with pytest.raises(ValueError, match=r"call.txt, line 2: 'soon'"):
        ev.read_reference(labels)
    labels.write_text("0.5 1.0 Alice\n")
    with pytest.raises(ValueError, match="separated by tabs"):
        ev.read_reference(labels)
    labels.write_text("\n")
    with pytest.raises(ValueError, match="no speaker turns"):
        ev.read_reference(labels)


def test_finds_recordings_with_a_reference_beside_them(tmp_path):
    for name in ("a.wav", "a.txt", "b.m4a", "b.rttm", "c.wav", "notes.txt"):
        (tmp_path / name).write_text("x")
    found = [(c.name, c.audio.name, c.reference.name) for c in ev.find_cases(tmp_path)]
    assert found == [("a", "a.wav", "a.txt"), ("b", "b.m4a", "b.rttm")]


# -- scoring --------------------------------------------------------------------


def test_mapping_pairs_by_shared_time_not_by_name():
    reference = [turn(0, 10, "Alice"), turn(10, 12, "Bob")]
    found = [turn(0, 9, "SPEAKER_01"), turn(9, 12, "SPEAKER_00")]
    assert ev.speaker_mapping(reference, found) == {
        "SPEAKER_01": "Alice",
        "SPEAKER_00": "Bob",
    }


def test_mapping_is_one_to_one_when_one_speaker_was_split():
    reference = [turn(0, 10, "Alice")]
    found = [turn(0, 6, "SPEAKER_00"), turn(6, 10, "SPEAKER_01")]
    assert ev.speaker_mapping(reference, found) == {"SPEAKER_00": "Alice"}


def test_diarization_error_parts():
    reference = [turn(0, 4, "A"), turn(4, 6, "B")]
    perfect = [turn(0, 4, "x"), turn(4, 6, "y")]
    mapping = ev.speaker_mapping(reference, perfect)
    missed, false_alarm, confusion, total = ev.diarization_error(
        reference, perfect, mapping
    )
    assert (missed, false_alarm, confusion) == (0, 0, 0)
    assert total == pytest.approx(6.0)

    # B's two seconds given to A's speaker, one second missed, one invented.
    found = [turn(1, 6, "x"), turn(7, 8, "x")]
    missed, false_alarm, confusion, total = ev.diarization_error(
        reference, found, ev.speaker_mapping(reference, found)
    )
    assert missed == pytest.approx(1.0)
    assert false_alarm == pytest.approx(1.0)
    assert confusion == pytest.approx(2.0)


def _words(*spans):
    return [Word(start=s, end=e, word=f" w{i}") for i, (s, e) in enumerate(spans)]


def test_score_counts_words_and_quick_replies():
    reference = [turn(0, 3, "Alice"), turn(3.2, 3.8, "Bob"), turn(4, 7, "Alice")]
    found = [turn(0, 3, "S0"), turn(3.2, 3.8, "S1"), turn(4, 7, "S0")]
    attributed = [
        Segment(0, 3, "", words=_words((0, 1), (1, 2), (2, 3)), speaker="S0"),
        # The quick reply given to the wrong person.
        Segment(3.2, 3.8, "", words=_words((3.2, 3.7)), speaker="S0"),
        Segment(4, 7, "", words=_words((4, 5), (5, 6)), speaker="S0"),
        # Words outside every reference turn are not scored.
        Segment(9, 10, "", words=_words((9, 10)), speaker="S0"),
    ]
    score = ev.score_case("call", reference, found, attributed)
    assert (score.words_scored, score.words_right, score.words_not_scored) == (6, 5, 1)
    assert (score.short_turns, score.short_turns_with_words) == (1, 1)
    assert score.short_turns_right == 0
    # The separation had Bob's turn; the word was still given to Alice.
    assert (score.short_lost_assignment, score.short_lost_separation) == (1, 0)
    assert score.der == pytest.approx(0.0)


def test_a_reply_the_separation_never_found_is_counted_as_its_loss():
    reference = [turn(0, 3, "Alice"), turn(3.2, 3.8, "Bob"), turn(4, 7, "Alice")]
    # Bob has turns elsewhere, but none at the reply: Alice's runs straight over it.
    found = [turn(0, 7, "S0"), turn(8, 9, "S1")]
    reference.append(turn(8, 9, "Bob"))
    attributed = [Segment(3.2, 3.8, "", words=_words((3.2, 3.7)), speaker="S0")]
    score = ev.score_case("call", reference, found, attributed)
    assert (score.short_lost_separation, score.short_lost_assignment) == (1, 0)
    assert "1 the speaker separation did not find" in "\n".join(
        ev.summary_lines([score])
    )


def test_word_in_overlapping_speech_is_not_scored():
    reference = [turn(0, 2, "Alice"), turn(1, 3, "Bob")]
    attributed = [Segment(1.1, 1.9, "", words=_words((1.1, 1.9)), speaker="S0")]
    score = ev.score_case("call", reference, [turn(0, 3, "S0")], attributed)
    assert (score.words_scored, score.words_not_scored) == (0, 1)


def test_unattributed_word_counts_as_wrong():
    reference = [turn(0, 2, "Alice")]
    attributed = [Segment(0, 2, "", words=_words((0, 1)), speaker=None)]
    score = ev.score_case("call", reference, [turn(0, 2, "S0")], attributed)
    assert (score.words_scored, score.words_right, score.words_unattributed) == (
        1,
        0,
        1,
    )


def test_total_pools_every_word_and_second():
    a = ev.CaseScore(
        "a", reference_seconds=10, confusion=1, words_scored=10, words_right=9
    )
    b = ev.CaseScore(
        "b", reference_seconds=30, missed=3, words_scored=30, words_right=15
    )
    pooled = ev.total([a, b])
    assert pooled.der == pytest.approx(0.1)
    assert pooled.word_accuracy == pytest.approx(0.6)


# -- cache and command line -------------------------------------------------------


def _cached_run(tmp_path):
    testset = tmp_path / "testset"
    testset.mkdir()
    (testset / "call.wav").write_bytes(b"not read when rescoring")
    (testset / "call.txt").write_text("0\t3\tAlice\n3.2\t3.8\tBob\n")
    out = tmp_path / "results"
    segments = [
        Segment(0, 3.8, "w0 w1", words=_words((0, 2), (3.2, 3.7))),
    ]
    found = [turn(0, 3, "SPEAKER_00"), turn(3.2, 3.8, "SPEAKER_01")]
    ev.save_hypothesis(out / "cache" / "call.json", segments, found, {"model": "m"})
    return testset, out


def test_cache_round_trip_holds_no_audio(tmp_path):
    _testset, out = _cached_run(tmp_path)
    cached = out / "cache" / "call.json"
    segments, found, meta = ev.load_hypothesis(cached)
    assert [w.word for w in segments[0].words] == [" w0", " w1"]
    assert found[1] == turn(3.2, 3.8, "SPEAKER_01")
    assert meta["model"] == "m"
    assert b"not read" not in cached.read_bytes()


def test_cache_from_another_version_is_refused(tmp_path):
    _testset, out = _cached_run(tmp_path)
    cached = out / "cache" / "call.json"
    cached.write_text(cached.read_text().replace('"version": 1', '"version": 99'))
    with pytest.raises(ValueError, match="another version"):
        ev.load_hypothesis(cached)


def test_rescore_writes_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "_offline", lambda: None)
    testset, out = _cached_run(tmp_path)
    assert ev.main([str(testset), "--out", str(out), "--rescore"]) == 0
    rows = list(csv.DictReader((out / "scores.csv").open(encoding="utf-8")))
    assert rows[0]["recording"] == "call"
    assert (rows[0]["words_scored"], rows[0]["words_right"]) == ("2", "2")
    assert rows[0]["short_turns_right"] == "1"
    summary = (out / "summary.txt").read_text(encoding="utf-8")
    assert "Words to right person" in summary
    assert "100.0% of 2" in capsys.readouterr().out


def test_rescore_without_a_first_run_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "_offline", lambda: None)
    testset, out = _cached_run(tmp_path)
    (out / "cache" / "call.json").unlink()
    assert ev.main([str(testset), "--out", str(out), "--rescore"]) == 2
    assert "run once without --rescore" in capsys.readouterr().out


def test_empty_test_set_says_what_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "_offline", lambda: None)
    (tmp_path / "call.wav").write_bytes(b"x")
    assert ev.main([str(tmp_path), "--out", str(tmp_path / "out"), "--rescore"]) == 2
    assert "has a reference file beside it" in capsys.readouterr().out


def test_shipped_copy_runs_it_from_the_application_entry(tmp_path, monkeypatch):
    from whispr import app

    monkeypatch.setattr(ev, "_offline", lambda: None)
    monkeypatch.setattr(app, "launch", lambda: pytest.fail("opened the window"))
    testset, out = _cached_run(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        app.main(["--score-test-set", str(testset), "--out", str(out), "--rescore"])
    assert excinfo.value.code == 0
    assert (out / "summary.txt").is_file()
