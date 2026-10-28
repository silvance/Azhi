import pytest

from sotto.matching import ComparisonResult, GalleryMatch, GalleryResult
from sotto.provenance import (
    AnalysisProvenance,
    DiarizationProvenance,
    SourceRecord,
    TranscriptionProvenance,
)
from sotto.quality import GOOD
from sotto.reports import (
    DISCLAIMER_HEADING,
    GallerySearch,
    build_report_sections,
    render_text,
    write_analysis_report,
)
from sotto.thresholds import (
    BAND_HIGH,
    BAND_INSUFFICIENT,
    BAND_LOW,
    BAND_NOT_ASSESSED,
    DISCLAIMER,
    NO_LEAD_EXPLAINED,
)
from sotto.transcription import Segment, TranscriptionResult
from sotto.voiceprints import MatchDecision

SOURCE_SHA = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def _provenance():
    return AnalysisProvenance(
        source=SourceRecord(
            filename="operation.wav",
            original_path="/cases/operation.wav",
            file_size=1024,
            sha256=SOURCE_SHA,
        ),
        transcription=TranscriptionProvenance(
            model_name="small.en", model_sha256="a" * 64, detected_language="en"
        ),
        diarization=DiarizationProvenance(
            engine="pyannote",
            embedding_model="titanet-large",
            embedding_model_sha256="c" * 64,
        ),
    )


def _result():
    return TranscriptionResult(
        text="hello there",
        language="en",
        language_probability=0.98,
        duration=12.0,
        segments=[
            Segment(start=0.0, end=6.0, text="hello", speaker="SPEAKER_00"),
            Segment(start=6.0, end=12.0, text="there", speaker="voice::Alpha"),
        ],
    )


def _comparison(**kw):
    defaults = dict(
        reference_name="Subject Alpha",
        questioned_label="operation.wav",
        score=0.78,
        band=BAND_HIGH,
        reference_seconds=38.4,
        questioned_seconds=14.2,
        reference_quality=GOOD,
        questioned_quality="Fair",
        embedding_model="titanet-large, dim 192, sha256 cccccccccccc",
    )
    defaults.update(kw)
    return ComparisonResult(**defaults)


def _text(**kw):
    return render_text(build_report_sections(**kw))


# -- content ---------------------------------------------------------------


def test_report_includes_source_hash_and_build():
    text = _text(result=_result(), provenance=_provenance())
    assert SOURCE_SHA in text
    assert "operation.wav" in text
    assert "Application version:" in text
    assert "Build ID:" in text


def test_report_includes_model_information():
    text = _text(result=_result(), provenance=_provenance())
    assert "small.en" in text
    assert "a" * 64 in text  # transcription model hash
    assert "titanet-large" in text
    assert "c" * 64 in text  # embedding model hash


def test_report_includes_transcript_with_timestamps_and_speakers():
    text = _text(
        result=_result(),
        speaker_names={"voice::Alpha": "Alpha"},
        provenance=_provenance(),
    )
    assert "[00:00-00:06]" in text
    assert "[Speaker 1] hello" in text
    assert "[Alpha] there" in text
    assert "Detected language: en" in text


def test_report_includes_comparison_details():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert "Reference subject: Subject Alpha" in text
    assert "Similarity score: 0.78 / 1.00" in text
    assert f"Assessment: {BAND_HIGH}" in text
    assert "Operational threshold:" in text
    assert "Reference speech: 38.4 sec" in text
    assert "Questioned speech: 14.2 sec" in text


def test_a_comparison_reports_its_windows_and_recording_conditions():
    from sotto.conditions import Side, compare
    from sotto.matching import WindowScore

    comparison = _comparison(
        windows=[
            WindowScore(
                spans=[(0.0, 8.0)], speech_seconds=7.1, quality=GOOD, score=0.8
            ),
            WindowScore(spans=[(8.0, 16.0)], skipped="poor audio (1.2s of speech)"),
        ],
        conditions=compare(
            Side(speech_seconds=14.2, pieces=2, bandwidth_hz=3750.0),
            Side(speech_seconds=38.4, pieces=5, bandwidth_hz=7000.0),
        ),
    )
    text = _text(comparisons=[comparison])
    assert "  0:00-0:08: 0.80 (7.1 sec of speech, good)" in text
    assert "  0:08-0:16: not used (poor audio (1.2s of speech))" in text
    assert "Frequency range: up to 3.8 kHz (telephone-like) / up to 7.0 kHz" in text
    assert "questioned audio reaches only 3.8 kHz" in text


def test_high_similarity_is_phrased_as_a_lead_not_an_identification():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert "Further review is warranted" in text
    lowered = text.lower()
    assert "positive identification" not in lowered
    assert "confirmed speaker" not in lowered
    assert "is john doe" not in lowered


def test_analysis_body_never_asserts_identity_or_a_percentage():
    """The findings must not claim identity.

    The disclaimer section is excluded on purpose: it legitimately contains
    phrases like "biometric probability of identity" *in order to deny them*.
    What matters is that the analysis body never asserts them.
    """
    sections = build_report_sections(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    body = render_text([s for s in sections if s.heading != DISCLAIMER_HEADING]).lower()
    for banned in (
        "probability of identity",
        "same person",
        "match probability",
        "biometric",
        "positive identification",
        "confirmed speaker",
    ):
        assert banned not in body
    # The score is a raw value out of 1.00, never a percentage.
    assert "similarity score: 0.78 / 1.00" in body
    assert "78%" not in body


def test_disclaimer_is_present_verbatim():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert DISCLAIMER in text
    assert "Interpretation and limitations" in text
    assert "require validation against representative" in text


def test_report_records_active_thresholds():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert "Active thresholds:" in text
    assert "Recognition acceptance:" in text
    assert "Recognition margin:" in text


def test_refused_comparison_is_reported_as_refused():
    refused = _comparison(
        refused=True, refusal_reason="Different speaker-embedding models."
    )
    text = _text(result=_result(), provenance=_provenance(), comparisons=[refused])
    assert "comparison refused" in text.lower()
    assert "Different speaker-embedding models." in text
    assert "Similarity score:" not in text


def test_insufficient_comparison_does_not_claim_a_lead():
    weak = _comparison(band=BAND_INSUFFICIENT, questioned_seconds=1.0)
    text = _text(result=_result(), provenance=_provenance(), comparisons=[weak])
    assert BAND_INSUFFICIENT in text
    assert "Further review is warranted" not in text


def test_missing_provenance_is_stated_not_invented():
    text = _text(result=_result())
    assert "not recorded" in text
    assert "cannot be confirmed from this report" in text


def test_report_without_comparisons():
    text = _text(result=_result(), provenance=_provenance())
    assert "No speaker comparisons were performed." in text


def test_report_without_transcript():
    text = _text(provenance=_provenance(), comparisons=[_comparison()])
    assert "No transcript is attached" in text


def test_multiple_comparisons_are_numbered():
    text = _text(
        result=_result(),
        provenance=_provenance(),
        comparisons=[_comparison(), _comparison(reference_name="Subject Bravo")],
    )
    assert "Comparison 1" in text
    assert "Comparison 2" in text
    assert "Subject Bravo" in text


# -- writing ---------------------------------------------------------------


def test_write_text_report(tmp_path):
    path = tmp_path / "report.txt"
    write_analysis_report(
        path,
        result=_result(),
        provenance=_provenance(),
        comparisons=[_comparison()],
        case_notes="Operation Nightfall, tasking 42.",
    )
    text = path.read_text(encoding="utf-8")
    assert SOURCE_SHA in text
    assert "Operation Nightfall, tasking 42." in text
    assert DISCLAIMER in text


def test_write_docx_report(tmp_path):
    pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    write_analysis_report(
        path, result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert path.exists() and path.stat().st_size > 0


# -- A report must not describe the wrong recording --------------------------


def test_a_comparison_names_the_recording_it_measured():
    comparison = _comparison(
        questioned_source_filename="intercept-042.wav",
        questioned_source_sha256="d" * 64,
        questioned_selection="diarized speaker (SPEAKER_01, 6 turn(s))",
        questioned_window_count=4,
    )
    text = render_text(build_report_sections(comparisons=[comparison]))
    assert "intercept-042.wav" in text
    assert ("d" * 64) in text
    assert "diarized speaker (SPEAKER_01, 6 turn(s))" in text
    assert "measured across 4 window(s)" in text


def test_a_transcript_of_another_recording_is_flagged_not_implied():
    """The transcribed file and the compared file are not assumed to be one."""
    comparison = _comparison(
        questioned_source_filename="intercept-042.wav",
        questioned_source_sha256="d" * 64,
    )
    text = render_text(
        build_report_sections(
            result=_result(), provenance=_provenance(), comparisons=[comparison]
        )
    )
    assert "is not the same as every recording compared below" in text
    assert "operation.wav" in text and "intercept-042.wav" in text


def test_no_note_when_the_transcript_is_of_the_compared_recording():
    comparison = _comparison(
        questioned_source_filename="operation.wav",
        questioned_source_sha256=SOURCE_SHA,
    )
    text = render_text(
        build_report_sections(
            result=_result(), provenance=_provenance(), comparisons=[comparison]
        )
    )
    assert "is not the same as every recording" not in text


def test_comparisons_without_a_transcript_say_the_source_section_is_not_theirs():
    comparison = _comparison(
        questioned_source_filename="intercept-042.wav",
        questioned_source_sha256="d" * 64,
    )
    text = render_text(build_report_sections(comparisons=[comparison]))
    assert "does not describe the questioned recording" in text


def test_a_comparison_with_no_recorded_source_makes_no_claim():
    text = render_text(build_report_sections(comparisons=[_comparison()]))
    assert "Questioned recording:" not in text
    assert "does not describe the questioned recording" not in text


# -- Searches of every profile (QA-14) ----------------------------------------
# Exporting from Compare with a search on screen wrote an earlier one-to-one
# comparison instead, because a search could not be reported at all.


def _search(**kw):
    defaults = dict(
        matches=[
            GalleryMatch("s1", "Subject Alpha", 0.81, BAND_HIGH),
            GalleryMatch("s2", "Subject Bravo", 0.32, BAND_LOW),
        ],
        decision=MatchDecision(
            best_name="Subject Alpha", best_score=0.81, accepted=True
        ),
        searched=2,
        questioned_source_filename="intercept-0914.wav",
        questioned_source_sha256="d" * 64,
    )
    defaults.update(kw)
    return GallerySearch(
        GalleryResult(**defaults),
        questioned_seconds=22.5,
        questioned_quality="Fair",
        questioned_window_count=4,
    )


def test_a_search_is_reported_with_its_ranking_and_recording():
    text = _text(comparisons=[_search()])
    assert "Search of all reference profiles 1" in text
    assert "Questioned recording: intercept-0914.wav" in text
    assert "d" * 64 in text
    assert "Questioned speech: 22.5 sec (Fair), measured across 4 window(s)" in text
    assert "1. Subject Alpha   0.81" in text
    assert "Lead: the questioned speaker produced high similarity" in text
    assert "No speaker comparisons were performed." not in text


def test_a_search_without_a_lead_says_so():
    search = _search(
        decision=MatchDecision(
            best_name="Subject Alpha", best_score=0.5, accepted=False
        )
    )
    text = _text(comparisons=[search])
    assert NO_LEAD_EXPLAINED in text
    assert "Further review is warranted" not in text


def test_a_search_on_too_little_speech_claims_nothing():
    search = _search(
        matches=[GalleryMatch("s1", "Subject Alpha", 0.81, BAND_NOT_ASSESSED)],
        inadequate_reason="Only 1.8s of usable questioned speech.",
    )
    text = _text(comparisons=[search])
    assert BAND_INSUFFICIENT in text
    assert "supports no conclusion about any subject" in text
    assert "Lead:" not in text


def test_comparisons_and_searches_keep_their_order():
    text = _text(comparisons=[_comparison(), _search()])
    assert text.index("Comparison 1") < text.index("Search of all reference profiles 2")


def test_a_searched_recording_other_than_the_transcript_is_flagged():
    text = _text(result=_result(), provenance=_provenance(), comparisons=[_search()])
    assert "is not the same as every recording compared below" in text
    assert "intercept-0914.wav" in text


def test_the_report_shows_notes_under_their_lines_unless_left_out():
    from sotto.reports import _transcript_lines
    from sotto.transcription import Note, Segment, TranscriptionResult

    result = TranscriptionResult(
        "", "en", 1.0, 10.0, segments=[Segment(1.0, 3.0, "Meet at ten.")]
    )
    result.notes = [Note(1.0, "Which ten?")]
    assert _transcript_lines(result)[-1] == "    Analyst note: Which ten?"
    assert "Analyst note" not in "\n".join(_transcript_lines(result, notes=False))
