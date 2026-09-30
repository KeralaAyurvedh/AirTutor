"""Document-Grounded RAG (Retrieval-Augmented Generation) Engine for AirTutor.

Provides local document ingestion (.txt, .md, .pdf), chunking, FAISS vector indexing
using sentence-transformers/all-MiniLM-L6-v2, and similarity threshold guardrails.

Target execution environment: CPU-only, 8 GB RAM target.
"""

from pathlib import Path
import re
from typing import Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE: int = 400
DEFAULT_CHUNK_OVERLAP: int = 50
RAG_SIMILARITY_THRESHOLD: float = 0.45
FALLBACK_UNGROUNDED_CONTEXT: str = (
    "The requested concept is not covered in the uploaded curriculum materials."
)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into chunks of roughly chunk_size words with overlap."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = cleaned.split(" ")
    if not words or words == [""]:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def extract_text_from_file(file_path: str | Path) -> str:
    """Extract plain text from .txt, .md, or .pdf files."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif suffix == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n\n".join(pages_text)

    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .txt, .md, .pdf")


class RAGEngine:
    """In-memory FAISS vector store and retrieval engine with cosine similarity guardrails."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize lightweight embedding model
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # In-memory FAISS IndexFlatIP (Inner Product over L2-normalized vectors = Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks: list[dict[str, Any]] = []

    def clear(self) -> None:
        """Reset the vector index and stored chunk registry."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks.clear()

    def ingest_text(self, text: str, source: str = "raw_text") -> int:
        """Chunk raw text, compute embeddings, normalize, and add to FAISS index."""
        text_chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not text_chunks:
            return 0

        # Compute embeddings
        embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # L2-normalize vectors for exact cosine similarity in IndexFlatIP
        faiss.normalize_L2(embeddings)

        start_idx = len(self.chunks)
        self.index.add(embeddings)

        for i, chunk in enumerate(text_chunks):
            self.chunks.append(
                {
                    "chunk_id": start_idx + i,
                    "text": chunk,
                    "source": source,
                }
            )

        return len(text_chunks)

    def ingest_file(self, file_path: str | Path) -> int:
        """Extract text from file and index into FAISS."""
        text = extract_text_from_file(file_path)
        return self.ingest_text(text, source=str(file_path))

    def retrieve(self, query: str, k: int = 3) -> dict[str, Any]:
        """Retrieve top-k relevant chunks with cosine similarity scores and guardrail flag.

        Returns:
            {
                "context": list[str],
                "source_chunks": list[int],
                "retrieval_scores": list[float],
                "grounded": bool
            }
        """
        if self.index.ntotal == 0 or not query.strip():
            return {
                "context": [FALLBACK_UNGROUNDED_CONTEXT],
                "source_chunks": [],
                "retrieval_scores": [],
                "grounded": False,
            }

        # Encode and normalize query
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)

        # Top-k search
        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, actual_k)

        top_scores = [float(s) for s in scores[0] if s != -1]
        top_indices = [int(idx) for idx in indices[0] if idx != -1]

        # Guardrail check
        max_score = top_scores[0] if top_scores else 0.0
        grounded = max_score >= self.similarity_threshold

        if grounded:
            context = [self.chunks[idx]["text"] for idx in top_indices]
        else:
            context = [FALLBACK_UNGROUNDED_CONTEXT]

        return {
            "context": context,
            "source_chunks": top_indices,
            "retrieval_scores": top_scores,
            "grounded": bool(grounded),
        }
