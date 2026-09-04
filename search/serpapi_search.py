"""
search/serpapi_search.py
========================

Google Lens reverse-image search via SerpAPI.

Current SerpAPI flow for local images:

1. Upload image -> https://serpapi.com/image
2. Receive temporary image_id
3. Search Google Lens -> https://serpapi.com/search
   with engine=google_lens and image_id=<id>

The SERPAPI_KEY environment variable must be set.
A .env file is supported via python-dotenv.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERPAPI_IMAGE_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search"

MAX_RESULTS = 10
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# SerpAPI Image API currently accepts images up to 500 KB.
MAX_IMAGE_SIZE_BYTES = 500 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Load and return the SerpAPI API key."""

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    key = os.getenv("SERPAPI_KEY", "").strip()

    if not key:
        raise EnvironmentError(
            "SERPAPI_KEY environment variable is not set."
        )

    return key


def _extract_visual_matches(
    raw: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract and normalize Google Lens visual matches.

    Returns a list containing:
        title
        link
        thumbnail
        source
        source_icon
    """

    matches: list[dict[str, Any]] = []

    # Current Google Lens response
    for item in raw.get("visual_matches", []):
        if not isinstance(item, dict):
            continue

        link = item.get("link", "")

        if not link:
            continue

        matches.append(
            {
                "title": item.get("title", ""),
                "link": link,
                "thumbnail": item.get("thumbnail", ""),
                "source": item.get("source", ""),
                "source_icon": item.get("source_icon", ""),
            }
        )

    return matches[:MAX_RESULTS]


# ---------------------------------------------------------------------------
# Upload local image
# ---------------------------------------------------------------------------


def _upload_image(path: Path, api_key: str) -> str:
    """
    Upload a local image using the SerpAPI Image API.

    Returns:
        image_id
    """

    file_size = path.stat().st_size

    if file_size > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image is too large for SerpAPI Image API: "
            f"{file_size / 1024:.1f} KB. "
            f"Maximum allowed is 500 KB."
        )

    logger.info(
        "Uploading image to SerpAPI Image API | image=%s | size=%.1f KB",
        path.name,
        file_size / 1024,
    )

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            with open(path, "rb") as fh:

                response = requests.post(
                    SERPAPI_IMAGE_URL,
                    params={"api_key": api_key},
                    files={
                        "image": (
                            path.name,
                            fh,
                            "application/octet-stream",
                        )
                    },
                    timeout=REQUEST_TIMEOUT,
                )

            if response.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))

                logger.warning(
                    "SerpAPI Image API rate limited. "
                    "Waiting %.1f seconds.",
                    wait,
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            raw: dict[str, Any] = response.json()

            if "error" in raw:
                raise RuntimeError(
                    f"SerpAPI Image API error: {raw['error']}"
                )

            image_id = raw.get("image_id")

            if not image_id:
                raise RuntimeError(
                    "SerpAPI Image API did not return an image_id."
                )

            logger.info(
                "Image uploaded successfully | image_id=%s",
                image_id,
            )

            return str(image_id)

        except requests.exceptions.Timeout as exc:
            last_exc = exc

            logger.warning(
                "Image upload timed out | attempt %d/%d",
                attempt,
                MAX_RETRIES,
            )

        except requests.exceptions.RequestException as exc:
            last_exc = exc

            logger.warning(
                "Image upload failed | attempt %d/%d | %s",
                attempt,
                MAX_RETRIES,
                exc,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(
        f"SerpAPI image upload failed after {MAX_RETRIES} attempts: "
        f"{last_exc}"
    )


# ---------------------------------------------------------------------------
# Google Lens search using image_id
# ---------------------------------------------------------------------------


def _search_by_image_id(
    image_id: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Search Google Lens using a SerpAPI image_id.
    """

    params = {
        "engine": "google_lens",
        "image_id": image_id,
        "api_key": api_key,
        "type": "visual_matches",
        "no_cache": "true",
    }

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            logger.info(
                "Searching Google Lens | attempt %d/%d",
                attempt,
                MAX_RETRIES,
            )

            response = requests.get(
                SERPAPI_SEARCH_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:

                wait = RETRY_BACKOFF * (2 ** (attempt - 1))

                logger.warning(
                    "Google Lens rate limited. "
                    "Waiting %.1f seconds.",
                    wait,
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            raw: dict[str, Any] = response.json()

            if "error" in raw:
                raise RuntimeError(
                    f"SerpAPI Google Lens error: {raw['error']}"
                )

            matches = _extract_visual_matches(raw)

            logger.info(
                "Google Lens returned %d visual match(es).",
                len(matches),
            )

            return matches

        except requests.exceptions.Timeout as exc:

            last_exc = exc

            logger.warning(
                "Google Lens request timed out | attempt %d/%d",
                attempt,
                MAX_RETRIES,
            )

        except requests.exceptions.RequestException as exc:

            last_exc = exc

            logger.warning(
                "Google Lens request failed | attempt %d/%d | %s",
                attempt,
                MAX_RETRIES,
                exc,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(
        f"Google Lens search failed after {MAX_RETRIES} attempts: "
        f"{last_exc}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_by_image(
    image_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Perform Google Lens reverse-image search on a local image.

    Flow:

        local image
            ↓
        SerpAPI Image API
            ↓
        image_id
            ↓
        Google Lens API
            ↓
        visual matches
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Source image not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Source image path is not a file: {path}"
        )

    api_key = _get_api_key()

    logger.info(
        "Starting SerpAPI Google Lens search | image=%s",
        path.name,
    )

    # Step 1: upload local image
    image_id = _upload_image(
        path,
        api_key,
    )

    # Step 2: perform Google Lens search
    matches = _search_by_image_id(
        image_id,
        api_key,
    )

    if not matches:
        logger.warning(
            "Google Lens returned 0 visual matches."
        )

    return matches


def search_by_url(
    image_url: str,
) -> list[dict[str, Any]]:
    """
    Perform Google Lens reverse-image search using a public image URL.

    This skips the Image API because Google Lens can directly accept
    a publicly accessible image URL.
    """

    if not image_url:
        raise ValueError("image_url cannot be empty.")

    api_key = _get_api_key()

    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": api_key,
        "type": "visual_matches",
        "no_cache": "true",
    }

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            logger.info(
                "Searching Google Lens by URL | attempt %d/%d",
                attempt,
                MAX_RETRIES,
            )

            response = requests.get(
                SERPAPI_SEARCH_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:

                wait = RETRY_BACKOFF * (2 ** (attempt - 1))

                logger.warning(
                    "Rate limited. Waiting %.1f seconds.",
                    wait,
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            raw: dict[str, Any] = response.json()

            if "error" in raw:
                raise RuntimeError(
                    f"SerpAPI Google Lens error: {raw['error']}"
                )

            matches = _extract_visual_matches(raw)

            logger.info(
                "Google Lens returned %d visual match(es).",
                len(matches),
            )

            return matches

        except requests.exceptions.Timeout as exc:

            last_exc = exc

        except requests.exceptions.RequestException as exc:

            last_exc = exc

            logger.warning(
                "Google Lens URL request failed: %s",
                exc,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(
        f"Google Lens URL search failed after "
        f"{MAX_RETRIES} attempts: {last_exc}"
    )