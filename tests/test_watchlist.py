"""The watch list: what counts as a hit, and what does not."""

from azhi.watchlist import Matcher, describe, matcher, parse


def _hits(terms, text):
    return [text[s:e] for s, e, _term in Matcher(terms).finditer(text)]


def test_one_term_per_line_tidied_and_without_repeats():
    assert parse("  Ann \n\nthe   bridge\nANN\nO'Brien\n") == [
        "Ann",
        "the bridge",
        "O'Brien",
    ]
    assert matcher("\n  \n") is None


def test_whole_words_in_any_case():
    text = "Ann met ANNA, Joanne and ann."
    assert _hits(["Ann"], text) == ["Ann", "ann"]


def test_a_phrase_matches_across_any_spacing():
    assert _hits(["the bridge"], "at the\nbridge, then THE  BRIDGE") == [
        "the\nbridge",
        "THE  BRIDGE",
    ]


def test_speaker_tags_and_times_are_not_searched():
    text = "[Ann] I'm here.\n[00:12:34] [Speaker 2] Where's Ann?\n[…] Ann again\n"
    found = Matcher(["Ann"])
    assert [text[s:e] for s, e, _ in found.finditer(text)] == ["Ann", "Ann"]
    assert found.counts(text) == {"Ann": 2}


def test_punctuation_in_a_term_is_taken_literally():
    assert _hits(["C++", "J.D."], "Ask J.D. about C++ and JxD.") == ["J.D.", "C++"]


def test_other_scripts_match_as_whole_words():
    assert _hits(["Москва"], "в Москва, не Москвадом") == ["Москва"]
    assert _hits(["بغداد"], "من بغداد الى") == ["بغداد"]


def test_the_longest_term_wins_where_two_overlap():
    found = Matcher(["bridge", "the old bridge"])
    assert [t for _, _, t in found.finditer("by the old bridge")] == ["the old bridge"]


def test_hits_are_reported_as_the_term_was_written():
    found = Matcher(["Ann", "the bridge"])
    counts = found.counts("ANN and the  BRIDGE and ann")
    assert counts == {"Ann": 2, "the bridge": 1}
    assert describe(counts, listed=True) == "Watch list: Ann ×2, the bridge ×1."
    assert describe({}, listed=True) == "Watch list: nothing in this transcript."
    assert describe({}, listed=False) == ""
