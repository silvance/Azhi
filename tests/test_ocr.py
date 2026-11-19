import importlib.util
import os

import pytest

from whispers import ocr
from whispers.ocr import (
    extract_text,
    is_image,
    is_ocr_file,
    is_pdf,
    ocr_available,
    tesseract_lang,
)


@pytest.mark.parametrize(
    "code,expected",
    [
        ("ar", "ara"),
        ("fa", "fas"),
        ("he", "heb"),
        ("zh", "chi_sim"),
        ("ko", "kor"),
        ("en", "eng"),
        ("chi_tra", "chi_tra"),  # already a Tesseract code -> passthrough
        ("xx", "xx"),  # unknown -> passthrough
    ],
)
def test_tesseract_lang_mapping(code, expected):
    assert tesseract_lang(code) == expected


@pytest.mark.parametrize(
    "name,image,pdf,ocrable",
    [
        ("scan.png", True, False, True),
        ("photo.JPG", True, False, True),
        ("doc.pdf", False, True, True),
        ("notes.txt", False, False, False),
        ("clip.mp3", False, False, False),
    ],
)
def test_file_type_predicates(name, image, pdf, ocrable):
    assert is_image(name) is image
    assert is_pdf(name) is pdf
    assert is_ocr_file(name) is ocrable


def test_ocr_available_false_without_binary(monkeypatch):
    monkeypatch.setattr(ocr, "find_tesseract", lambda: None)
    assert ocr_available() is False


def test_ensure_tessdata_prefix_sets_unquoted_path(tmp_path, monkeypatch):
    # Regression: the tessdata dir must reach Tesseract via TESSDATA_PREFIX with no
    # surrounding quotes (the old --tessdata-dir config string embedded quotes in
    # the path, so Tesseract failed to open "<dir>"/eng.traineddata).
    tessdata = tmp_path / "tesseract" / "tessdata"
    tessdata.mkdir(parents=True)
    monkeypatch.setattr(ocr, "bundled_tessdata_dir", lambda: tessdata)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    ocr._ensure_tessdata_prefix()
    assert os.environ["TESSDATA_PREFIX"] == str(tessdata)
    assert '"' not in os.environ["TESSDATA_PREFIX"]


def test_extract_text_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "missing.png", lang="eng")


def test_extract_text_rejects_non_ocr_file(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="image or PDF"):
        extract_text(notes, lang="eng")


def test_extract_image_requires_pillow(tmp_path):
    if importlib.util.find_spec("PIL") is not None:
        pytest.skip("Pillow is installed; missing-dependency path not exercised")
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="Pillow is not installed"):
        extract_text(image, lang="eng")


def test_extract_pdf_requires_pypdfium2(tmp_path):
    if importlib.util.find_spec("pypdfium2") is not None:
        pytest.skip("pypdfium2 is installed; missing-dependency path not exercised")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError, match="pypdfium2 is not installed"):
        extract_text(pdf, lang="eng")


# -- Reading scans when the language is left to be detected ------------------


class _Image:
    def __init__(self, name="page"):
        self.name = name

    def crop(self, box):
        return _Image(f"{self.name}-crop")


def _data(lines):
    """image_to_data output for lines of (confidence, words)."""
    out = {k: [] for k in ("text", "conf", "block_num", "par_num", "line_num")}
    out.update(left=[], top=[], width=[], height=[])
    for number, (confidence, words) in enumerate(lines):
        for i, word in enumerate(words.split()):
            out["text"].append(word)
            out["conf"].append(confidence)
            out["block_num"].append(1)
            out["par_num"].append(1)
            out["line_num"].append(number)
            out["left"].append(10 + 50 * i)
            out["top"].append(20 * number)
            out["width"].append(40)
            out["height"].append(15)
    return out


class _Tesseract:
    """Stands in for pytesseract: English reads the page, except two lines of
    Chinese, which only chi_sim reads well."""

    class Output:
        DICT = "dict"

    def __init__(self, script="Latin", chinese=True):
        self.script = script
        self.chinese = chinese
        self.calls = []

    def image_to_osd(self, image, output_type=None):
        return {"script": self.script}

    def image_to_data(self, image, lang, output_type=None):
        self.calls.append(("data", image.name, lang))
        if image.name.endswith("crop"):
            return _data([(92.0 if lang == "chi_sim" else 30.0, "会议 定于")])
        lines = [(95.0, "Meeting notes, part one.")]
        if self.chinese:
            lines.append((14.0, "SWEET PRS LFARER"))
        return _data(lines)

    def image_to_string(self, image, lang):
        self.calls.append(("string", image.name, lang))
        return f" text read with {lang} "


@pytest.fixture
def tesseract(monkeypatch):
    def use(fake, installed=("eng", "ara", "chi_sim", "fra", "osd")):
        monkeypatch.setattr(ocr, "_configure_pytesseract", lambda: fake)
        monkeypatch.setattr(ocr, "_ensure_tessdata_prefix", lambda: None)
        monkeypatch.setattr(ocr, "_installed_tessdata", set(installed))
        return fake

    return use


def test_a_chosen_language_is_read_alongside_english(tesseract):
    fake = tesseract(_Tesseract())
    assert ocr._ocr_image(_Image(), "chi_sim") == "text read with chi_sim+eng"
    assert fake.calls == [("string", "page", "chi_sim+eng")]


def test_lines_english_could_not_read_are_read_in_their_own_script(tesseract):
    fake = tesseract(_Tesseract())
    found = []
    assert ocr._ocr_image(_Image(), None, found) == "text read with chi_sim+eng"
    # Only the non-Latin languages are tried, on the unread lines alone.
    tried = [lang for kind, name, lang in fake.calls if name == "page-crop"]
    assert tried == ["ara", "chi_sim"]
    assert found == ["chi_sim"]

    # The next page tries what the last one found first, and stops there.
    fake.calls = []
    ocr._ocr_image(_Image(), None, found)
    assert [lang for _, name, lang in fake.calls if name == "page-crop"] == ["chi_sim"]


def test_a_page_english_reads_well_is_read_once_in_english(tesseract):
    fake = tesseract(_Tesseract(chinese=False))
    assert ocr._ocr_image(_Image(), None) == "text read with eng"
    assert [kind for kind, _, _ in fake.calls] == ["data", "string"]


def test_a_page_mostly_in_another_script_is_read_in_it(tesseract):
    fake = tesseract(_Tesseract(script="Arabic", chinese=False))
    assert ocr._ocr_image(_Image(), None) == "text read with ara+eng"
    assert fake.calls[-1] == ("string", "page", "ara+eng")
