"""Documents: cleaning extracted text, page furniture, emails, paragraphs, routing."""

from whispers.documents import (
    char_script,
    choose_language,
    clean,
    dominant_script,
    field_label,
    parse,
    remove_page_furniture,
)

INSTALLED = {"zh", "ar", "fa", "ru"}


def _never(text):
    raise AssertionError(f"detector consulted for {text!r}")


# -- clean ---------------------------------------------------------------------


def test_clean_takes_out_what_word_cannot_save():
    text = "a\r\nb\rc\x0cd\x0be\x00f\x07g￾h"
    assert clean(text) == "a\nb\nc\n\nd\nef" + "gh"


def test_clean_mends_a_lone_surrogate_and_a_word_broken_at_a_line_end():
    assert clean("bro­\nken \ud800x") == "broken �x"
    assert clean("con\x02 \ntinued") == "continued"


# -- page furniture ------------------------------------------------------------


BANNER = "CUI // Exported from A365 eDiscovery - Case 2025-0117"
BODIES = [
    "The convoy left at dawn.\nIt reached the river by noon.",
    "Radio traffic was light.\nTwo calls were logged.",
    "The meeting was moved.\nNo reason was given.",
    "Supplies arrived late.\nThe count was short by two crates.",
]


def test_running_headers_and_page_numbers_are_taken_out_and_reported():
    pages = [f"{BANNER}\n{body}\nPage {n} of 4" for n, body in enumerate(BODIES, 1)]
    out, removed = remove_page_furniture(pages)
    assert out == BODIES
    assert len(removed) == 8
    assert "Page 3 of 4" in removed


def test_text_repeated_only_inside_pages_is_kept():
    pages = [
        f"Top {n}\nTop line two\nThe same sentence.\nBottom one\nBottom {n}"
        for n in range(4)
    ]
    out, removed = remove_page_furniture(pages)
    assert out[0] == "The same sentence."
    assert len(removed) == 16


def test_a_long_line_that_differs_only_in_a_number_is_not_a_footer():
    line = "The convoy left the base at {} hours and arrived at the checkpoint on time today"
    pages = [f"Intro {n}.\n{line.format(n)}\nEnd." for n in range(1, 5)]
    out, removed = remove_page_furniture(pages)
    assert removed == ["Intro 1.", "End.", "Intro 2.", "End.", "Intro 3.", "End."] + [
        "Intro 4.",
        "End.",
    ]
    assert out[3] == line.format(4)


def test_email_header_lines_are_never_furniture():
    # A thread from one sender repeats the same From line at the top of
    # every page.
    pages = [
        f"From: Ali <a@example.com>\nSubject: Plan\nNote {n}, which says more than a footer would."
        for n in range(4)
    ]
    out, removed = remove_page_furniture(pages)
    assert removed == []
    assert out == pages


def test_too_few_pages_have_nothing_but_page_numbers_taken_out():
    out, removed = remove_page_furniture(["Banner\ntext 1\n1", "Banner\ntext 2\n2"])
    assert removed == ["1", "2"]
    assert out[0] == "Banner\ntext 1"


# -- header fields ---------------------------------------------------------------


def test_header_fields_in_english_and_arabic():
    assert field_label("Subject: Meeting") == "Subject"
    assert field_label("  sent: Tuesday") == "Sent"
    assert field_label("الموضوع: الاجتماع") == "Subject"
    assert field_label("إلى: علي") == "To"
    assert field_label("Subjects to discuss") is None


def test_an_arabic_subject_extracted_with_its_label_at_the_end():
    # A mostly right-to-left line comes out of the PDF reversed.
    assert field_label("جدول الاجتماع :Subject") == "Subject"
    # English text that happens to end in a label word is not a field.
    assert field_label("Please see the note :subject") is None


# -- scripts ---------------------------------------------------------------------


def test_scripts():
    assert char_script("中") == "han"
    assert char_script("ع") == "arabic"
    assert char_script("é") == "latin"
    assert char_script("5") is None
    # Six Chinese characters outweigh a dozen English letters.
    assert dominant_script("Report 会议定于下周三 attached") == "han"
    assert dominant_script("こんにちは世界") == "kana"
    assert dominant_script("123 !!") is None


# -- parse -----------------------------------------------------------------------

_WIDE = "x" * 60


def test_wrapped_lines_are_joined_into_paragraphs():
    text = (
        "The liaison officer prepared these notes after the meeting and they\n"
        "were reviewed the same day by the section.\n"
        "Short line.\n"
        "\n"
        "会议定于下周三上午九点在第二会议室举行请各部门负责人准时出席如有\n"
        "变动将另行通知。\n"
        "End of excerpt; English commentary continues here."
    )
    (section,) = parse(text)
    assert section.paragraphs == [
        "The liaison officer prepared these notes after the meeting and they "
        "were reviewed the same day by the section.",
        "Short line.",
        "会议定于下周三上午九点在第二会议室举行请各部门负责人准时出席如有变动将另行通知。",
        "End of excerpt; English commentary continues here.",
    ]


def test_a_change_of_script_or_a_bullet_starts_a_new_paragraph():
    text = (
        "English commentary that runs nearly the whole width of the page here\n"
        "会议定于下周三上午九点在第二会议室举行请各部门负责人准时出席\n"
        "- a bullet that follows it and is not part of the paragraph above it\n"
        "- another bullet"
    )
    (section,) = parse(text)
    assert len(section.paragraphs) == 4


def test_emails_are_split_on_their_header_blocks():
    text = (
        "Mailbox export\n"
        "From: Ali <a@example.com>\n"
        "Sent: Tuesday, March 4, 2025 9:12 AM\n"
        "To: Smith, John <john.smith@example.com>;\n"
        "   Ops <ops@example.com>\n"
        "Subject: جدول الاجتماع\n"
        "\n"
        "مرحبا\n"
        "\n"
        "جدول الاجتماع :Subject\n"
        "From: Sara <s@example.com>\n"
        "Date: 5 March 2025\n"
        "\n"
        "Please confirm receipt."
    )
    cover, first, second = parse(text)
    assert not cover.is_email and cover.paragraphs == ["Mailbox export"]
    assert first.is_email
    assert first.value("To") == (
        "Smith, John <john.smith@example.com>; Ops <ops@example.com>"
    )
    assert first.value("Subject") == "جدول الاجتماع"
    assert first.paragraphs == ["مرحبا"]
    assert second.value("Subject") == "جدول الاجتماع"
    assert second.value("From") == "Sara <s@example.com>"
    assert second.paragraphs == ["Please confirm receipt."]


def test_one_field_line_in_running_text_is_not_an_email():
    (section,) = parse("Notes\nSubject: the budget\nMore notes")
    assert not section.is_email
    assert section.paragraphs[1] == "Subject: the budget"


# -- choose_language ---------------------------------------------------------------


def test_the_script_decides_without_the_detector():
    assert choose_language(
        "会议定于下周三", forced=None, installed=INSTALLED, detect=_never
    ) == ("zh", "")
    assert choose_language(
        "Привет, мир", forced=None, installed=INSTALLED, detect=_never
    ) == ("ru", "")


def test_english_names_and_numbers_are_left_alone():
    for text in ("Smith, John", "2025-0117", "OK", ""):
        assert choose_language(
            text, forced="zh", installed=INSTALLED, detect=_never
        ) == (None, "")


def test_long_english_is_left_alone_even_when_a_language_is_chosen():
    text = "End of excerpt one; English commentary continues on the next page."
    detect = lambda _: [("en", 0.99)]  # noqa: E731
    assert choose_language(text, forced="zh", installed=INSTALLED, detect=detect) == (
        None,
        "",
    )


def test_long_text_in_another_latin_language_is_translated_or_named():
    text = "Le convoi a quitté la base à six heures et il est arrivé à l'heure."
    detect = lambda _: [("fr", 0.99)]  # noqa: E731
    assert choose_language(
        text, forced=None, installed=INSTALLED | {"fr"}, detect=detect
    ) == ("fr", "")
    assert choose_language(text, forced=None, installed=INSTALLED, detect=detect) == (
        None,
        "fr",
    )
    unsure = lambda _: [("fr", 0.6)]  # noqa: E731
    assert choose_language(
        text, forced=None, installed=INSTALLED | {"fr"}, detect=unsure
    ) == (None, "")


def test_the_language_chosen_under_from_wins_within_its_script():
    persian = "این یک متن آزمایشی است"
    detect = lambda _: [("ar", 0.8), ("fa", 0.2)]  # noqa: E731
    assert choose_language(
        persian, forced="fa", installed=INSTALLED, detect=detect
    ) == ("fa", "")
    assert choose_language(
        persian, forced=None, installed=INSTALLED, detect=detect
    ) == ("ar", "")
    # A choice in another script does not apply to this paragraph.
    assert choose_language(
        persian, forced="zh", installed=INSTALLED, detect=detect
    ) == ("ar", "")


def test_a_script_with_no_pack_installed_is_named():
    assert choose_language(
        "안녕하세요 세계", forced=None, installed=INSTALLED, detect=_never
    ) == (None, "ko")
