import codecs

import pytest

from sotto.textfiles import UnreadableText, read_text


def test_utf8_with_and_without_bom(tmp_path):
    plain = tmp_path / "a.txt"
    plain.write_bytes("привет".encode("utf-8"))
    marked = tmp_path / "b.txt"
    marked.write_bytes(codecs.BOM_UTF8 + "привет".encode("utf-8"))
    assert read_text(plain) == "привет"
    assert read_text(marked) == "привет"


def test_utf16_with_bom(tmp_path):
    path = tmp_path / "c.txt"
    path.write_bytes("مرحبا".encode("utf-16"))
    assert read_text(path) == "مرحبا"


@pytest.mark.parametrize(
    "text, encoding",
    [("привет мир", "cp1251"), ("مرحبا", "cp1256"), ("你好世界", "gbk")],
)
def test_legacy_encodings_are_refused_not_garbled(tmp_path, text, encoding):
    path = tmp_path / "legacy.txt"
    path.write_bytes(text.encode(encoding))
    with pytest.raises(UnreadableText, match="not saved as UTF-8"):
        read_text(path)
