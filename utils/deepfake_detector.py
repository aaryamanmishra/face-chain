"""
utils/deepfake_detector.py
==========================
Experimental deepfake detection signal for candidate images using an open-source
pretrained vision transformer (ViT) image classifier.

Model: "prithivMLmods/Deep-Fake-Detector-v2-Model" (Hugging Face)

Lazy-loaded on first use; gracefully degrades if torch/transformers are
not installed or model weights cannot be fetched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable Constants (no magic numbers in logic)
# ---------------------------------------------------------------------------

# Pretrained Hugging Face classification model identifier
DEFAULT_DEEPFAKE_MODEL_NAME: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"

# Default confidence threshold (0.0 to 1.0) for flagging candidate as suspicious.
# Tunable default: predictions with confidence >= this threshold and a synthetic/fake
# label trigger an additional caution warning during the pipeline run.
DEFAULT_SUSPICIOUS_CONFIDENCE_THRESHOLD: float = 0.50

# Generic caveat note attached to all deepfake predictions
FIXED_DEEPFAKE_CAVEAT_NOTE: str = (
    "Prediction from a single-frame image classifier trained primarily on full-face swap "
    "deepfakes. May not reliably detect region-specific manipulation such as "
    "lip-sync/reenactment. Treat as a probabilistic signal, not a determination."
)

# Generic label keywords indicative of synthetic / manipulated content
# (matched case-insensitively against model output labels).
SYNTHETIC_LABEL_KEYWORDS: tuple[str, ...] = (
    "deepfake",
    "fake",
    "synthetic",
    "manipulated",
    "ai",
)

# ---------------------------------------------------------------------------
# Module-level model state (lazy loaded)
# ---------------------------------------------------------------------------

_pipeline_instance: Any = None
_DEEPFAKE_MODEL_AVAILABLE: bool | None = None
_UNAVAILABLE_REASON: str = ""


def _get_classifier():
    """Lazily load and return the image classification pipeline.

    Returns None if transformers/torch are missing or model fails to load.
    """
    global _pipeline_instance, _DEEPFAKE_MODEL_AVAILABLE, _UNAVAILABLE_REASON

    if _pipeline_instance is not None:
        return _pipeline_instance

    if _DEEPFAKE_MODEL_AVAILABLE is False:
        return None

    try:
        from transformers import pipeline  # noqa: PLC0415

        logger.info(
            "Loading deepfake classification model: %s…",
            DEFAULT_DEEPFAKE_MODEL_NAME,
        )
        _pipeline_instance = pipeline(
            "image-classification",
            model=DEFAULT_DEEPFAKE_MODEL_NAME,
        )
        _DEEPFAKE_MODEL_AVAILABLE = True
        return _pipeline_instance
    except Exception as exc:  # noqa: BLE001
        _DEEPFAKE_MODEL_AVAILABLE = False
        _UNAVAILABLE_REASON = str(exc)
        logger.warning(
            "Deepfake detector model unavailable (%s). "
            "Pipeline will continue without deepfake scoring. Details: %s",
            DEFAULT_DEEPFAKE_MODEL_NAME,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_suspicious_deepfake(
    analysis: dict[str, Any],
    threshold: float = DEFAULT_SUSPICIOUS_CONFIDENCE_THRESHOLD,
) -> bool:
    """Evaluate if deepfake analysis meets criteria for a suspicious flag.

    Parameters
    ----------
    analysis:
        Dict returned by ``analyze_deepfake_risk()``.
    threshold:
        Confidence threshold (default: ``DEFAULT_SUSPICIOUS_CONFIDENCE_THRESHOLD``).

    Returns
    -------
    bool
        True if analysis is available, label indicates synthetic media,
        and confidence >= threshold.
    """
    if not analysis.get("deepfake_analysis_available"):
        return False

    label = analysis.get("predicted_label")
    if not label or not isinstance(label, str):
        return False

    confidence = analysis.get("confidence")
    if confidence is None or float(confidence) < threshold:
        return False

    label_lower = label.strip().lower()
    return any(kw in label_lower for kw in SYNTHETIC_LABEL_KEYWORDS)


def analyze_deepfake_risk(image_path: str | Path) -> dict[str, Any]:
    """Classify deepfake likelihood of a candidate image file.

    Parameters
    ----------
    image_path:
        Path to local candidate image.

    Returns
    -------
    dict
        Always contains:
        {
            "deepfake_analysis_available": bool,
            "predicted_label": str | None,
            "confidence": float | None,
            "note": str
        }
    """
    image_str = str(image_path)
    path_obj = Path(image_path)

    # Validate file existence before attempting pipeline load
    if not path_obj.exists():
        return {
            "deepfake_analysis_available": False,
            "predicted_label": None,
            "confidence": None,
            "note": f"{FIXED_DEEPFAKE_CAVEAT_NOTE} (Image file not found: {image_str})",
        }

    classifier = _get_classifier()
    if classifier is None:
        reason = f" (Unavailable: {_UNAVAILABLE_REASON})" if _UNAVAILABLE_REASON else " (Model not loaded or dependencies missing)"
        return {
            "deepfake_analysis_available": False,
            "predicted_label": None,
            "confidence": None,
            "note": f"{FIXED_DEEPFAKE_CAVEAT_NOTE}{reason}",
        }

    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path_obj) as img:
            rgb_img = img.convert("RGB")
            predictions = classifier(rgb_img)

        if not predictions or not isinstance(predictions, list):
            raise ValueError("Classifier returned unexpected format or empty predictions.")

        top_pred = predictions[0]
        raw_label = str(top_pred.get("label", ""))
        raw_score = float(top_pred.get("score", 0.0))

        return {
            "deepfake_analysis_available": True,
            "predicted_label": raw_label,
            "confidence": round(raw_score, 4),
            "note": FIXED_DEEPFAKE_CAVEAT_NOTE,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Deepfake inference failed on image %s: %s", image_str, exc)
        return {
            "deepfake_analysis_available": False,
            "predicted_label": None,
            "confidence": None,
            "note": f"{FIXED_DEEPFAKE_CAVEAT_NOTE} (Inference failed: {exc})",
        }
