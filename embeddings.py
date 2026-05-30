"""
embeddings.py — Builds a FAISS index over study materials at startup.

Each material is embedded as "{topic}: {title}" so semantic search can match
vague queries like "equations" or "light and mirrors" to the right material
even without exact keyword overlap.

The index is built once at startup and reused across all requests.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("study_assistant.embeddings")

# ---------------------------------------------------------------------------
# Lazy globals — populated once by build_index()
# ---------------------------------------------------------------------------

_faiss            = None
_openai_client    = None
_index            = None
_indexed_materials: list[dict] = []


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss  # type: ignore
        _faiss = faiss
    return _faiss


def _get_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"


def _embed(texts: list[str]) -> np.ndarray:
    """Calls OpenAI embeddings API. Returns float32 array of shape (N, dim)."""
    logger.info(f"Embedding {len(texts)} text(s) via {EMBEDDING_MODEL}")
    client   = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors  = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Index builder — called once at startup
# ---------------------------------------------------------------------------

def build_index() -> None:
    """
    Loads study_materials.json, embeds each material, and builds a FAISS
    IndexFlatIP (cosine similarity via L2-normalised vectors).
    """
    global _index, _indexed_materials

    data_path = Path(__file__).parent / "data" / "study_materials.json"
    with open(data_path) as f:
        data = json.load(f)

    materials = data["materials"]
    texts     = [f"{m['topic']}: {m['title']}" for m in materials]

    logger.info(f"Building FAISS index over {len(texts)} materials...")
    vectors = _embed(texts)

    faiss = _get_faiss()
    dim   = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    _index             = index
    _indexed_materials = materials
    logger.info(f"FAISS index ready — dim={dim}, vectors={index.ntotal}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_materials(query: str, top_k: int = 2) -> list[dict]:
    """
    Semantically searches the FAISS index for materials matching the query.
    Returns up to top_k results with a similarity score attached.
    Returns [] if the index hasn't been built yet.
    """
    if _index is None or not _indexed_materials:
        logger.warning("search_materials called before index was built — returning []")
        return []

    faiss     = _get_faiss()
    query_vec = _embed([query])
    faiss.normalize_L2(query_vec)

    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        material = dict(_indexed_materials[idx])
        material["similarity_score"] = round(float(score), 4)
        results.append(material)

    logger.info(f"search_materials('{query}') → {[r['title'] for r in results]}")
    return results