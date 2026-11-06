"""
Ingestion pipeline for the maintenance-assistant RAG system.

Pipeline: PDF manuals -> extract text per page (pypdf) -> chunk with overlap
-> embed with sentence-transformers (all-MiniLM-L6-v2, local, no API key) ->
store vectors + metadata in a FAISS index.

Chunking choice (documented in README too):
  CHUNK_SIZE_WORDS = 220   (~roughly 280-320 tokens for this kind of technical
                            prose; small enough that a retrieved chunk is still
                            topically coherent, large enough to keep a full
                            instruction step / warning intact)
  CHUNK_OVERLAP_WORDS = 40 (~15-18% overlap; prevents a procedure or warning
                            that straddles a chunk boundary from being cut in
                            half and becoming unretrievable or ambiguous)
We chunk on whitespace-split word counts rather than raw characters because
word count is a much closer proxy for LLM token count across manuals that mix
dense paragraphs and short tabular/numeric lines (equipment tables, torque
specs), so chunk "density" stays comparable between documents.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).resolve().parent
MANUALS_DIR = APP_DIR / "manuals"
INDEX_DIR = APP_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40


@dataclass
class Chunk:
    chunk_id: int
    doc: str          # source filename
    page: int          # 1-indexed page number within that PDF
    text: str


def extract_pages(pdf_path: Path) -> List[str]:
    """Return list of page texts (index 0 == page 1)."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # normalize whitespace a bit, keep content faithful
        text = " ".join(text.split())
        pages.append(text)
    return pages


def chunk_page_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
                     overlap: int = CHUNK_SIZE_WORDS and CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    i = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def build_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    pdf_files = sorted(MANUALS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDFs found in {MANUALS_DIR}")
    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            for piece in chunk_page_text(page_text):
                chunks.append(Chunk(chunk_id=cid, doc=pdf_path.name, page=page_num, text=piece))
                cid += 1
    return chunks


def build_index(force: bool = False) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists() and META_PATH.exists() and not force:
        print("Index already exists; skipping build (pass force=True to rebuild).")
        return

    print("Extracting + chunking PDFs...")
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from {len(set(c.doc for c in chunks))} documents.")

    print(f"Loading embedding model {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c.text for c in chunks]
    print("Embedding chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True,
                               normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump([asdict(c) for c in chunks], f)

    print(f"Saved FAISS index -> {INDEX_PATH}")
    print(f"Saved chunk metadata -> {META_PATH}")


if __name__ == "__main__":
    build_index(force=True)
