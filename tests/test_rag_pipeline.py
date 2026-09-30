"""Unit tests for Document-Grounded RAG Engine (src/llm/rag_engine.py)."""

import pytest
from src.llm.rag_engine import (
    RAGEngine,
    chunk_text,
    extract_text_from_file,
    RAG_SIMILARITY_THRESHOLD,
    FALLBACK_UNGROUNDED_CONTEXT,
)


def test_chunk_text_logic():
    # Build text of known word/token length
    sample_text = " ".join([f"word_{i}" for i in range(120)])
    chunks = chunk_text(sample_text, chunk_size=40, overlap=10)
    assert len(chunks) >= 3

    # Verify chunks are non-empty and respect chunking bounds
    for chunk in chunks:
        assert len(chunk) > 0

    # Test with sentence-transformers tokenizer explicitly
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tok = model.tokenizer
    token_chunks = chunk_text(sample_text, chunk_size=40, overlap=10, tokenizer=tok)
    assert len(token_chunks) >= 3

    # Check token count of the first chunk
    c0_tokens = tok.encode(token_chunks[0], add_special_tokens=False)
    assert len(c0_tokens) <= 40



def test_rag_retrieval_grounded():
    engine = RAGEngine()
    engine.clear()

    doc = (
        "Recursion is a programming technique where a function calls itself directly or indirectly. "
        "A recursive function must have a base case to terminate execution and prevent stack overflow. "
        "Classic examples include computing the factorial of a number and traversing tree data structures."
    )
    num_chunks = engine.ingest_text(doc, source="recursion_notes")
    assert num_chunks > 0

    # Query closely matching the document
    res = engine.retrieve("What is a base case in recursive functions?", k=2)
    assert res["grounded"] is True
    assert len(res["context"]) > 0
    assert "base case" in res["context"][0].lower()
    assert len(res["retrieval_scores"]) > 0
    assert res["retrieval_scores"][0] >= RAG_SIMILARITY_THRESHOLD


def test_rag_retrieval_ungrounded_fallback():
    engine = RAGEngine()
    engine.clear()

    doc = "Variables and types in Python: int, float, str, and boolean representations."
    engine.ingest_text(doc, source="python_variables")

    # Completely unrelated query
    res = engine.retrieve("How do deep sea hydrothermal vents sustain extremophile organisms?", k=2)
    assert res["grounded"] is False
    assert res["context"] == [FALLBACK_UNGROUNDED_CONTEXT]


def test_rag_ingest_file(tmp_path):
    engine = RAGEngine()
    engine.clear()

    # Create temporary markdown file
    md_file = tmp_path / "pointers.md"
    md_file.write_text(
        "# Pointers and Memory Management\n\n"
        "A pointer stores the memory address of another variable. "
        "Dereferencing a pointer accesses the value stored at that target address.",
        encoding="utf-8",
    )

    count = engine.ingest_file(md_file)
    assert count >= 1

    res = engine.retrieve("memory address stored in a pointer", k=1)
    assert res["grounded"] is True
    assert "pointer" in res["context"][0].lower()
