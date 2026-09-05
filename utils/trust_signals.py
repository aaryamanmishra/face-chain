"""
utils/trust_signals.py
======================
Independent, additive trust and corroboration signals for search candidates.

Provides:
1. Video content / single-frame extraction detection (is_video_content,
   frame_extraction_note).
2. Content corroboration evaluation (content_corroboration: HIGH / LOW / UNKNOWN).
3. Candidate dictionary annotation.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_EXTRACTION_NOTE = (
    "Match derived from a single extracted video frame/thumbnail, not the "
    "full video. Video content carries higher risk of AI-generated or "
    "manipulated media (e.g. deepfakes, lip-sync) than static images."
)

# Documented heuristic list of clickbait, sensationalized, or unrelated patterns.
# Used when evaluating content corroboration to detect titles that do not
# credibly corroborate identity attribution.
CLICKBAIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"put (?:her|his|their) house up for sale",
        r"you won'?t believe",
        r"day in the life",
        r"wait (?:till|until|for) the end",
        r"shocking",
        r"gone wrong",
        r"top (?:5|10|20)\b",
        r"things you didn'?t know",
        r"must watch",
        r"see what happened",
        r"secret revealed",
        r"what happened next",
        r"unbelievable",
        r"can'?t believe",
        r"try not to laugh",
        r"instant regret",
    ]
]

# URL path & platform patterns indicative of video/reel/shorts content
_VIDEO_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/reel/", re.IGNORECASE),
    re.compile(r"/reels/", re.IGNORECASE),
    re.compile(r"/tv/", re.IGNORECASE),
    re.compile(r"/shorts/", re.IGNORECASE),
    re.compile(r"/video(?:s)?/", re.IGNORECASE),
    re.compile(r"/watch\b", re.IGNORECASE),
    re.compile(r"/live/", re.IGNORECASE),
]

_VIDEO_DOMAINS: set[str] = {
    "tiktok.com",
    "www.tiktok.com",
    "vm.tiktok.com",
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "m.youtube.com",
}

_VIDEO_PLATFORMS: set[str] = {
    "TikTok",
    "YouTube",
}


# ---------------------------------------------------------------------------
# Video Content Detection
# ---------------------------------------------------------------------------


def detect_video_content(
    url: str,
    platform: str | None = None,
) -> tuple[bool, str | None]:
    """Determine whether a candidate URL or platform indicates video content.

    Detects video-based platforms (TikTok, YouTube) and video/reel URL paths
    (e.g., Instagram /reel/, YouTube /shorts/, Facebook /video/).

    Parameters
    ----------
    url:
        Candidate source or post URL.
    platform:
        Canonical platform name if already resolved.

    Returns
    -------
    tuple[bool, str | None]
        (is_video_content, frame_extraction_note if True else None).
    """
    if not url:
        return False, None

    # Check canonical platform name if already classified
    if platform and platform in _VIDEO_PLATFORMS:
        return True, FRAME_EXTRACTION_NOTE

    try:
        parsed = urlparse(url)
        hostname = (parsed.netloc or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        path = parsed.path or ""
    except Exception:
        return False, None

    # Check dedicated video domains
    if hostname in _VIDEO_DOMAINS or any(hostname.endswith("." + d) for d in _VIDEO_DOMAINS):
        return True, FRAME_EXTRACTION_NOTE

    # Check URL path patterns (e.g. /reel/, /shorts/, /video/, watch?v=)
    for pattern in _VIDEO_PATH_PATTERNS:
        if pattern.search(path):
            return True, FRAME_EXTRACTION_NOTE

    # Query string check (e.g. watch?v=)
    if parsed.query and "v=" in parsed.query:
        return True, FRAME_EXTRACTION_NOTE

    return False, None


# ---------------------------------------------------------------------------
# Content Corroboration Evaluation
# ---------------------------------------------------------------------------


def evaluate_content_corroboration(
    title: str,
    identity_context: str | None = None,
) -> str:
    """Evaluate whether post title credibly corroborates the target identity.

    Parameters
    ----------
    title:
        The post or page title string.
    identity_context:
        Optional known name or identity keywords.
        NOTE: identity_context is currently never passed from app.py, so "HIGH"
        is a reserved-for-future-use value — this is intentional, not a bug.

    Returns
    -------
    str
        "HIGH": Title contains matching identity context keywords.
        "LOW": Title matches clickbait / generic patterns, or context was
               provided but completely absent from title.
        "UNKNOWN": No identity context was provided and title does not match
                   any known clickbait heuristic patterns.
    """
    clean_title = (title or "").strip()

    # When identity_context is supplied (reserved for future use)
    if identity_context and identity_context.strip():
        clean_context = identity_context.strip().lower()
        title_lower = clean_title.lower()

        # Check for presence of identity words (length >= 3 to avoid noise)
        context_words = [w for w in re.split(r"\W+", clean_context) if len(w) >= 3]
        if context_words and any(w in title_lower for w in context_words):
            return "HIGH"
        return "LOW"

    # When no identity context is provided:
    # Flag generic clickbait or sensationalist patterns as "LOW"
    for pattern in CLICKBAIT_PATTERNS:
        if pattern.search(clean_title):
            return "LOW"

    # Default when no identity context is available to verify against
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Candidate Annotation
# ---------------------------------------------------------------------------


def annotate_trust_signals(
    candidate: dict[str, Any],
    identity_context: str | None = None,
) -> dict[str, Any]:
    """Add content_corroboration, is_video_content, and frame_extraction_note.

    Mutates and returns *candidate* in-place.
    """
    url = candidate.get("link") or candidate.get("downloaded_url") or candidate.get("url") or ""
    platform = candidate.get("platform")
    title = candidate.get("title") or ""

    is_video, note = detect_video_content(url, platform)
    corroboration = evaluate_content_corroboration(title, identity_context)

    candidate["content_corroboration"] = corroboration
    candidate["is_video_content"] = is_video
    if is_video and note:
        candidate["frame_extraction_note"] = note

    return candidate
