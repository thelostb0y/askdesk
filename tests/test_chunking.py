from askdesk.chunking import chunk_document


def test_small_doc_is_single_chunk():
    chunks = chunk_document("doc.md", "One paragraph.\n\nAnother paragraph.")
    assert len(chunks) == 1
    assert chunks[0].source == "doc.md"
    assert chunks[0].ordinal == 0
    assert "One paragraph." in chunks[0].text


def test_long_doc_splits_with_overlap():
    paragraphs = [f"Paragraph {i}. " + ("filler " * 60) for i in range(8)]
    chunks = chunk_document("doc.md", "\n\n".join(paragraphs), max_chars=800)
    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    # overlap: each later chunk starts with the previous chunk's last paragraph
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.text.split("\n\n")[0] == prev.text.split("\n\n")[-1]


def test_oversized_single_paragraph_is_hard_split():
    chunks = chunk_document("doc.md", "x" * 3000, max_chars=1000)
    assert len(chunks) == 3
    assert all(len(c.text) <= 1000 for c in chunks)


def test_empty_doc_yields_nothing():
    assert chunk_document("doc.md", "\n\n  \n\n") == []
