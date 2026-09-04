"""
face/detector.py
================
Face detection and embedding extraction using InsightFace.

Uses the ``buffalo_l`` model pack which provides:
- RetinaFace-based detection
- ArcFace 512-d embedding

Only the *largest* detected face (by bounding-box area) is used to avoid
ambiguity when multiple faces appear in the same photo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton — InsightFace model is expensive to load; we load it once.
# ---------------------------------------------------------------------------
_APP: Any = None  # insightface.app.FaceAnalysis instance


def _get_app() -> Any:
    """Return (and lazily initialise) the shared InsightFace FaceAnalysis app."""
    global _APP  # noqa: PLW0603

    if _APP is not None:
        return _APP

    try:
        import insightface  # noqa: PLC0415

        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        # det_size must be a multiple of 32; 640×640 is the recommended default.
        app.prepare(ctx_id=0, det_size=(640, 640))
        _APP = app
        logger.info("InsightFace FaceAnalysis (buffalo_l) initialised successfully.")
    except ImportError as exc:
        logger.critical("insightface is not installed: %s", exc)
        raise
    except Exception as exc:
        logger.critical("Failed to initialise InsightFace: %s", exc)
        raise

    return _APP


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_embedding(image_path: str | Path) -> dict[str, Any]:
    """Detect the face in *image_path* and return its ArcFace embedding.

    When multiple faces are detected the largest one (by bounding-box area)
    is selected, mimicking a "subject in focus" heuristic.

    Parameters
    ----------
    image_path:
        Path to the source image (JPEG, PNG, BMP, …).

    Returns
    -------
    dict with keys:
        - ``"embedding"`` (np.ndarray, shape (512,)) — L2-normalised face embedding.
        - ``"bbox"``      (list[int])                — [x1, y1, x2, y2] in pixels.
        - ``"det_score"`` (float)                    — Detection confidence (0-1).
        - ``"num_faces"`` (int)                      — Total faces found in image.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If the image cannot be decoded or no face is detected.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # ------------------------------------------------------------------ load
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"cv2 could not decode image: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------------- detect
    app = _get_app()
    faces = app.get(img_rgb)

    if not faces:
        logger.warning("No face detected in: %s", path.name)
        raise ValueError(f"No face detected in image: {path}")

    logger.info("Detected %d face(s) in '%s'.", len(faces), path.name)

    # ----------------------------------------- pick largest face by bbox area
    def _area(face: Any) -> float:
        x1, y1, x2, y2 = face.bbox.astype(int)
        return float((x2 - x1) * (y2 - y1))

    best_face = max(faces, key=_area)

    # -------------------------------------------------------- build result
    embedding: np.ndarray = best_face.embedding.astype(np.float32)
    bbox: list[int] = best_face.bbox.astype(int).tolist()
    det_score: float = float(best_face.det_score)

    logger.debug(
        "Best face | bbox=%s | det_score=%.4f | embedding_norm=%.4f",
        bbox,
        det_score,
        float(np.linalg.norm(embedding)),
    )

    return {
        "embedding": embedding,
        "bbox": bbox,
        "det_score": det_score,
        "num_faces": len(faces),
    }


def draw_bbox(image_path: str | Path, bbox: list[int], save_path: str | Path) -> None:
    """Draw a bounding box on the image and save it (for debugging).

    Parameters
    ----------
    image_path:
        Source image path.
    bbox:
        [x1, y1, x2, y2] bounding box coordinates.
    save_path:
        Destination path for the annotated image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Cannot draw bbox — image unreadable: %s", image_path)
        return

    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(save_path), img)
    logger.debug("Annotated image saved to: %s", save_path)
