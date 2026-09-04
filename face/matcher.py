"""
face/matcher.py
===============
Cosine-similarity based face matching.

ArcFace embeddings are already L2-normalised, so cosine similarity is
equivalent to the dot product.  We still perform explicit normalisation
here as a defensive measure in case embeddings arrive un-normalised.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD: float = 0.40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_embeddings(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Compute cosine similarity between two face embeddings.

    ArcFace embeddings are typically L2-normalised 512-d float32 vectors.
    The cosine similarity is mapped to [-1, 1] but in practice for faces it
    lies in roughly [−0.3, 1.0].  The default *threshold* of 0.40 is a
    conservative value that balances precision and recall on common datasets.

    Parameters
    ----------
    embedding1, embedding2:
        Face embedding vectors of equal length.
    threshold:
        Minimum cosine similarity to declare a match.

    Returns
    -------
    dict with keys:
        - ``"similarity"`` (float) — cosine similarity in [-1, 1].
        - ``"match"``      (bool)  — ``True`` if similarity >= threshold.

    Raises
    ------
    ValueError
        If the embeddings have different shapes or are zero vectors.
    """
    e1 = np.array(embedding1, dtype=np.float32).flatten()
    e2 = np.array(embedding2, dtype=np.float32).flatten()

    if e1.shape != e2.shape:
        raise ValueError(
            f"Embedding shape mismatch: {e1.shape} vs {e2.shape}"
        )

    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)

    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("One or both embeddings are zero vectors.")

    similarity: float = float(np.dot(e1 / n1, e2 / n2))
    is_match: bool = similarity >= threshold

    logger.debug(
        "Cosine similarity=%.4f | threshold=%.2f | match=%s",
        similarity,
        threshold,
        is_match,
    )

    return {"similarity": similarity, "match": is_match}


def rank_candidates(
    source_embedding: np.ndarray,
    candidates: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Rank a list of candidate face embeddings against a source embedding.

    Parameters
    ----------
    source_embedding:
        Embedding of the query / source face.
    candidates:
        List of dicts, each containing at minimum:
        ``{"embedding": np.ndarray, "url": str, "image_path": str}``.
    threshold:
        Similarity threshold passed to :func:`compare_embeddings`.

    Returns
    -------
    list[dict]
        Candidates sorted by descending similarity, each dict augmented with
        ``"similarity"`` and ``"match"`` keys.  Candidates that failed
        embedding extraction (``embedding is None``) are excluded.
    """
    results: list[dict[str, Any]] = []

    for cand in candidates:
        emb = cand.get("embedding")
        if emb is None:
            logger.debug("Skipping candidate (no embedding): %s", cand.get("url"))
            continue

        try:
            cmp = compare_embeddings(source_embedding, emb, threshold)
        except ValueError as exc:
            logger.warning("Skipping candidate due to comparison error: %s", exc)
            continue

        results.append(
            {
                **cand,
                "similarity": cmp["similarity"],
                "match": cmp["match"],
            }
        )

    results.sort(key=lambda d: d["similarity"], reverse=True)

    logger.info(
        "Ranked %d candidate(s); best similarity=%.4f",
        len(results),
        results[0]["similarity"] if results else 0.0,
    )

    return results
