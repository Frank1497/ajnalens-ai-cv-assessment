"""
Retrieval over the FAISS index built by ingest.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, TypedDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import INDEX_PATH, META_PATH, EMBED_MODEL_NAME, build_index

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list | None = None
_page_index: dict | None = None  # (doc, page) -> list of chunk dicts, ordered by chunk_id

# Hard cap on total chunks returned after sibling expansion, so a single
# heavily-chunked page (e.g. a multi-page troubleshooting table) cannot blow
# out the context sent to the LLM.
MAX_CHUNKS_AFTER_EXPANSION = 14


class RetrievedChunk(TypedDict):
    doc: str
    page: int
    text: str
    score: float


def _ensure_loaded() -> None:
    global _model, _index, _chunks, _page_index
    if not INDEX_PATH.exists() or not META_PATH.exists():
        build_index()
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    if _chunks is None:
        with open(META_PATH, "rb") as f:
            _chunks = pickle.load(f)
    if _page_index is None:
        _page_index = {}
        for c in _chunks:
            _page_index.setdefault((c["doc"], c["page"]), []).append(c)
        for key in _page_index:
            _page_index[key].sort(key=lambda c: c["chunk_id"])


def retrieve(query: str, k: int = 4, expand_siblings: bool = True) -> List[RetrievedChunk]:
    """Return top-k chunks most relevant to the query, each with doc/page/score.

    Sibling expansion: a source page that got split into multiple chunks at
    ingest time (e.g. a troubleshooting table where the problem-code list and
    the cause/solution definitions for those codes land in different chunks)
    can have its definition chunks rank far below the top-k purely on
    semantic similarity, since they read as dense lists with few query
    keywords. When expand_siblings is True, every OTHER chunk from the same
    (doc, page) as a retrieved hit is pulled in too (deduped, capped at
    MAX_CHUNKS_AFTER_EXPANSION total), so a table is never returned half-cut.
    Siblings are appended after the primary hits with their own FAISS score
    set to 0.0 (they were not independently ranked high enough to be a top-k
    hit on their own merit; they are included for page-locality).
    """
    _ensure_loaded()
    assert _model is not None and _index is not None and _chunks is not None and _page_index is not None

    q_emb = _model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = _index.search(q_emb, k)

    results: List[RetrievedChunk] = []
    seen_ids = set()
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        c = _chunks[idx]
        if c["chunk_id"] in seen_ids:
            continue
        seen_ids.add(c["chunk_id"])
        results.append(RetrievedChunk(doc=c["doc"], page=c["page"], text=c["text"], score=float(score)))

    if expand_siblings:
        primary_pages = [(c["doc"], c["page"]) for c in results]
        for doc, page in primary_pages:
            if len(results) >= MAX_CHUNKS_AFTER_EXPANSION:
                break
            for sibling in _page_index.get((doc, page), []):
                if len(results) >= MAX_CHUNKS_AFTER_EXPANSION:
                    break
                if sibling["chunk_id"] in seen_ids:
                    continue
                seen_ids.add(sibling["chunk_id"])
                results.append(RetrievedChunk(doc=sibling["doc"], page=sibling["page"],
                                               text=sibling["text"], score=0.0))

    return results
