"""
utils/social_media.py
=====================
Social media URL detection, platform classification, and post-specificity
validation for the face-chain pipeline.

Recognised platforms:
    instagram.com, x.com, twitter.com, linkedin.com,
    facebook.com, youtube.com, tiktok.com, reddit.com,
    pinterest.com, snapchat.com

Post-specificity rules (per platform)
--------------------------------------
A URL is considered a *specific post* only when its path matches the
platform's known content-URL pattern:

    Instagram  : /p/<id>  /reel/<id>  /tv/<id>
    X/Twitter  : /<user>/status/<id>
    Facebook   : /posts/<id>  /permalink/<id>  /photos/<id>  /video/<id>
    LinkedIn   : /posts/<slug>  /pulse/<slug>  /feed/update/urn:li:activity:<id>
    YouTube    : watch?v=<id>  /shorts/<id>  youtu.be/<id>
    TikTok     : /@<user>/video/<id>  /video/<id>
    Reddit     : /r/<sub>/comments/<id>  /comments/<id>
    Pinterest  : /pin/<id>
    Snapchat   : /add/<user> (accepted as a profile) - always flagged generic

Generic/rejected path fragments (apply to all platforms):
    /popular/  /explore  /search  /hashtag  /home  /trending
    /directory /discover /tags     /feed$    /reels$  /stories$
    (any path that is just "/" or empty after the domain)
"""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform registry
# ---------------------------------------------------------------------------

SOCIAL_MEDIA_DOMAINS: dict[str, list[str]] = {
    "Instagram": ["instagram.com", "www.instagram.com", "instagr.am"],
    "X / Twitter": ["x.com", "www.x.com", "twitter.com", "www.twitter.com", "t.co"],
    "LinkedIn": ["linkedin.com", "www.linkedin.com", "lnkd.in"],
    "Facebook": [
        "facebook.com",
        "www.facebook.com",
        "fb.com",
        "fb.me",
        "m.facebook.com",
    ],
    "YouTube": ["youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"],
    "TikTok": ["tiktok.com", "www.tiktok.com", "vm.tiktok.com"],
    "Reddit": ["reddit.com", "www.reddit.com", "old.reddit.com", "redd.it"],
    "Pinterest": ["pinterest.com", "www.pinterest.com", "pin.it"],
    "Snapchat": ["snapchat.com", "www.snapchat.com"],
}

_DOMAIN_TO_PLATFORM: dict[str, str] = {
    domain: platform
    for platform, domains in SOCIAL_MEDIA_DOMAINS.items()
    for domain in domains
}

# ---------------------------------------------------------------------------
# Generic-page rejection patterns (path substrings / full segments)
# These apply across all platforms before any post-specific check.
# ---------------------------------------------------------------------------

_GENERIC_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^/?$",                     # bare root "/"
        r"/popular(/|$)",
        r"/explore(/|$)",
        r"/search(/|$)",
        r"/hashtag(/|$)",
        r"/tags(/|$)",
        r"/home(/|$)",
        r"/trending(/|$)",
        r"/directory(/|$)",
        r"/discover(/|$)",
        r"/stories(/|$)",
        r"/feed/?$",                 # /feed at end only — not /feed/update/…
        r"/reels/?$",                # bare /reels (Instagram listing page)
        r"/tv/?$",                   # bare /tv (no content id)
        r"/channel(/|$)",            # YouTube channel listing
        r"/c/[^/]+/?$",              # YouTube custom channel /c/name
        r"/@[^/]+/?$",               # bare profile @handle without /video/…
        r"/user/[^/]+/?$",           # bare /user/name profile pages
        r"/r/[^/]+/?$",              # bare subreddit (no /comments/)
    ]
]

# ---------------------------------------------------------------------------
# Per-platform specific-post path requirements
# ---------------------------------------------------------------------------

# Each value is a compiled regex that must MATCH the URL path (and
# optionally the query string) for the URL to be considered a specific post.

_POST_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "Instagram": [
        re.compile(r"/p/[A-Za-z0-9_-]+", re.IGNORECASE),
        re.compile(r"/reel/[A-Za-z0-9_-]+", re.IGNORECASE),
        re.compile(r"/tv/[A-Za-z0-9_-]+", re.IGNORECASE),
    ],
    "X / Twitter": [
        # /<username>/status/<numeric_id>
        re.compile(r"/[^/]+/status/\d+", re.IGNORECASE),
    ],
    "Facebook": [
        re.compile(r"/posts/[A-Za-z0-9_-]+", re.IGNORECASE),
        re.compile(r"/permalink/", re.IGNORECASE),
        re.compile(r"/photos/[^/]+", re.IGNORECASE),
        re.compile(r"/video(s)?/\d+", re.IGNORECASE),
        re.compile(r"/story\.php", re.IGNORECASE),   # legacy story URLs
    ],
    "LinkedIn": [
        re.compile(r"/posts/[^/]+", re.IGNORECASE),
        re.compile(r"/pulse/[^/]+", re.IGNORECASE),
        # Activity URN: /feed/update/urn:li:activity:<id>
        re.compile(r"/feed/update/", re.IGNORECASE),
    ],
    "YouTube": [
        # Handled separately via query string watch?v=
        re.compile(r"/shorts/[A-Za-z0-9_-]+", re.IGNORECASE),
        re.compile(r"/live/[A-Za-z0-9_-]+", re.IGNORECASE),
    ],
    "TikTok": [
        # /@user/video/<id>  or  /video/<id>
        re.compile(r"/@[^/]+/video/\d+", re.IGNORECASE),
        re.compile(r"/video/\d+", re.IGNORECASE),
    ],
    "Reddit": [
        re.compile(r"/r/[^/]+/comments/[A-Za-z0-9_]+", re.IGNORECASE),
        re.compile(r"/comments/[A-Za-z0-9_]+", re.IGNORECASE),
    ],
    "Pinterest": [
        re.compile(r"/pin/\d+", re.IGNORECASE),
    ],
    # Snapchat has no public embeddable post URLs we can validate; always generic.
    "Snapchat": [],
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class UrlClassification(NamedTuple):
    """Classification result for a single URL."""

    platform: str | None
    """Canonical platform name, or ``None`` if not social media."""

    is_social: bool
    """True if URL belongs to any recognised social platform."""

    is_specific_post: bool
    """True if URL points to a specific piece of content (post/reel/video/…)."""

    rejection_reason: str | None
    """Human-readable reason why the URL was NOT classified as a specific post."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_hostname(url: str) -> tuple[str, str]:
    """Return (hostname_lower, path_lower) for *url*.

    Returns ("", "") on parse failure.
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower(), parsed.path
    except Exception:  # noqa: BLE001
        return "", ""


def _resolve_platform(hostname: str) -> str | None:
    """Map *hostname* to a canonical platform name."""
    stripped = re.sub(r"^www\.", "", hostname)

    if hostname in _DOMAIN_TO_PLATFORM:
        return _DOMAIN_TO_PLATFORM[hostname]
    if stripped in _DOMAIN_TO_PLATFORM:
        return _DOMAIN_TO_PLATFORM[stripped]

    # Suffix / base-name match for regional TLDs
    for domain, platform in _DOMAIN_TO_PLATFORM.items():
        base = domain.lstrip("www.").split(".")[0]
        if base and re.search(rf"(?:^|\.){re.escape(base)}\.", hostname):
            return platform

    return None


def _is_generic_path(path: str) -> bool:
    """Return True if *path* matches any generic/listing rejection pattern."""
    return any(p.search(path) for p in _GENERIC_PATH_PATTERNS)


def _matches_post_pattern(platform: str, path: str, query: str) -> bool:
    """Return True if *path* (and *query*) satisfy the post pattern for *platform*."""
    patterns = _POST_PATTERNS.get(platform, [])

    # YouTube: also check watch?v=
    if platform == "YouTube":
        if "v=" in query:
            # Ensure v= has a non-empty value
            qs = parse_qs(query)
            if qs.get("v", [""])[0]:
                return True

    return any(p.search(path) for p in patterns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_url(url: str) -> UrlClassification:
    """Fully classify *url*: platform, social, and post-specificity.

    Parameters
    ----------
    url:
        The URL to analyse.

    Returns
    -------
    UrlClassification
        Named tuple with ``platform``, ``is_social``, ``is_specific_post``,
        and ``rejection_reason`` fields.
    """
    if not url:
        return UrlClassification(None, False, False, "Empty URL")

    hostname, path = _normalise_hostname(url)
    if not hostname:
        return UrlClassification(None, False, False, "Unparseable URL")

    try:
        query = urlparse(url).query or ""
    except Exception:  # noqa: BLE001
        query = ""

    platform = _resolve_platform(hostname)

    if platform is None:
        return UrlClassification(None, False, False, "Not a recognised social media domain")

    # ------------------------------------------ generic-page rejection
    if _is_generic_path(path):
        reason = f"Generic/listing page path rejected: '{path}'"
        logger.debug("URL rejected as generic | platform=%s | path=%s", platform, path)
        return UrlClassification(platform, True, False, reason)

    # ------------------------------------------ post-pattern check
    if not _POST_PATTERNS.get(platform):
        # Platform has no known post pattern (e.g. Snapchat)
        reason = f"{platform} has no verifiable specific-post URL pattern"
        return UrlClassification(platform, True, False, reason)

    if _matches_post_pattern(platform, path, query):
        logger.debug("Specific post URL confirmed | platform=%s | path=%s", platform, path)
        return UrlClassification(platform, True, True, None)

    reason = (
        f"Path '{path}' does not match any known {platform} post pattern "
        f"(expected e.g. /p/<id>, /status/<id>, /posts/<id>, watch?v=…)"
    )
    logger.debug("URL failed post-pattern check | platform=%s | %s", platform, reason)
    return UrlClassification(platform, True, False, reason)


def get_platform(url: str) -> str | None:
    """Return the canonical platform name for *url*, or ``None`` if not social."""
    return classify_url(url).platform


def is_social_media(url: str) -> bool:
    """Return ``True`` if *url* belongs to a recognised social media platform."""
    return classify_url(url).is_social


def is_specific_post(url: str) -> bool:
    """Return ``True`` if *url* is a specific social-media post/content URL.

    Rejects generic pages such as popular/, explore, hashtag, home, search, etc.
    """
    return classify_url(url).is_specific_post


def annotate_platform(
    candidates: list[dict[str, Any]],
    url_key: str = "link",
) -> list[dict[str, Any]]:
    """Annotate every candidate with classification fields.

    Adds to each dict:
        - ``"platform"``         — canonical name or ``None``
        - ``"is_specific_post"`` — bool
        - ``"post_rejection"``   — human-readable reason when not a specific post

    Parameters
    ----------
    candidates:
        Candidate dicts to annotate.
    url_key:
        Key for the URL in each dict.

    Returns
    -------
    list[dict]
        All input candidates annotated in place (new dicts returned).
    """
    annotated: list[dict[str, Any]] = []

    for cand in candidates:
        url = cand.get(url_key, "") or cand.get("downloaded_url", "")
        cls = classify_url(url)
        annotated.append(
            {
                **cand,
                "platform": cls.platform,
                "is_specific_post": cls.is_specific_post,
                "post_rejection": cls.rejection_reason,
            }
        )

    return annotated


def filter_social_media(
    candidates: list[dict[str, Any]],
    url_key: str = "link",
) -> list[dict[str, Any]]:
    """Return only candidates that are from a social media platform.

    Each returned dict is annotated with ``platform``, ``is_specific_post``,
    and ``post_rejection`` keys.
    """
    results = [c for c in annotate_platform(candidates, url_key) if c.get("platform")]
    logger.info(
        "Social media filter: %d/%d candidates are from social platforms.",
        len(results),
        len(candidates),
    )
    return results
