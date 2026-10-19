from __future__ import annotations

import pytest

from docsage.chunking import chunk_text


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_windows_overlap_and_cover_the_text() -> None:
    chunks = chunk_text(_words(24), "d", chunk_size=10, overlap=3)
    assert [c.text.split()[0] for c in chunks] == ["w0", "w7", "w14"]
    # consecutive chunks share exactly `overlap` tokens
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.split()[-1] == "w23"
    assert [c.index for c in chunks] == [0, 1, 2]


def test_chunk_offsets_point_into_original_text_and_keep_formatting() -> None:
    text = "# Title\n\nFirst paragraph here.\n\nSecond   paragraph."
    (chunk,) = chunk_text(text, "d", chunk_size=50, overlap=5)
    assert text[chunk.start : chunk.end] == chunk.text
    assert "\n\n" in chunk.text
    assert chunk.id == "d:0"


def test_empty_text_and_invalid_arguments() -> None:
    assert chunk_text("   \n ", "d") == []
    with pytest.raises(ValueError):
        chunk_text("a b c", "d", chunk_size=5, overlap=5)
    with pytest.raises(ValueError):
        chunk_text("a b c", "d", chunk_size=0, overlap=0)
