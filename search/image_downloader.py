"""
search/image_downloader.py
==========================
Download candidate images from SerpAPI search results.

Prefers thumbnail URLs (small, fast) but falls back to the original page
link when a thumbnail is absent.  Each image is saved with a sanitised
filename into the ``downloads/`` directory.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "downloads"
REQUEST_TIMEOUT = 15  # seconds per image
MAX_RETRIES = 2
MIN_FILE_SIZE = 1_024  # bytes — skip obviously corrupt/empty responses
SUPPORTED_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitise_filename(url: str) -> str:
    """Derive a safe, unique filename from *url*.

    Uses a short SHA-256 prefix to avoid collisions and strips unsafe chars.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    parsed = urlparse(url)
    basename = Path(parsed.path).name or "image"
    # Strip query strings and non-alphanumeric characters (keep dots, dashes)
    basename = re.sub(r"[^A-Za-z0-9._-]", "_", basename)[:40]
    return f"{url_hash}_{basename}"


def _resolve_extension(content_type: str, filename: str) -> str:
    """Ensure *filename* has an image-appropriate extension."""
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return filename

    # Guess from content-type
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
    if guessed:
        return filename.rstrip(".") + guessed.replace(".jpe", ".jpg")

    return filename + ".jpg"  # safe default


def _download_single(url: str, dest: Path) -> bool:
    """Download *url* to *dest*.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any non-fatal failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True
            )

            if resp.status_code == 404:
                logger.debug("404 Not Found: %s", url)
                return False

            resp.raise_for_status()

            # --------------------------------------- MIME type validation
            content_type = resp.headers.get("Content-Type", "")
            mime = content_type.split(";")[0].strip().lower()
            if mime and not any(mime.startswith(m.split("/")[0]) for m in SUPPORTED_MIMES):
                # Allow image/* broadly
                if not mime.startswith("image/"):
                    logger.debug(
                        "Skipping non-image content-type '%s' for %s", mime, url
                    )
                    return False

            # ---------------------------------------------- write to disk
            dest_str = _resolve_extension(content_type, str(dest))
            dest = Path(dest_str)

            total = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    total += len(chunk)

            if total < MIN_FILE_SIZE:
                logger.debug(
                    "File too small (%d bytes), discarding: %s", total, url
                )
                dest.unlink(missing_ok=True)
                return False

            logger.debug("Downloaded %d bytes → %s", total, dest.name)
            return True

        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d for %s", attempt, url)
        except requests.exceptions.RequestException as exc:
            logger.warning("Download error (attempt %d): %s", attempt, exc)

        if attempt < MAX_RETRIES:
            time.sleep(1.5 * attempt)

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_candidates(
    search_results: list[dict[str, Any]],
    download_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Download thumbnail (or fallback) images for each search result.

    For each result in *search_results*:
    1. Attempts to download the ``thumbnail`` URL.
    2. Falls back to the ``link`` URL if the thumbnail fails or is absent.

    Parameters
    ----------
    search_results:
        List of dicts as returned by :func:`~search.serpapi_search.search_by_image`.
        Each dict must contain ``thumbnail``, ``link``, and ``title`` keys.
    download_dir:
        Directory to save images in.  Defaults to the project ``downloads/``.

    Returns
    -------
    list[dict]
        A subset of *search_results* that were successfully downloaded, each
        augmented with:
        - ``"image_path"`` (str) — local filesystem path of the downloaded file.
        - ``"downloaded_url"`` (str) — the URL that was actually downloaded.
    """
    save_dir = Path(download_dir) if download_dir else DOWNLOAD_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []

    for idx, result in enumerate(search_results):
        thumbnail_url: str = result.get("thumbnail", "")
        page_url: str = result.get("link", "")

        # Determine URLs to try in order
        urls_to_try: list[str] = []
        if thumbnail_url:
            urls_to_try.append(thumbnail_url)
        if page_url and page_url not in urls_to_try:
            urls_to_try.append(page_url)

        if not urls_to_try:
            logger.debug("No downloadable URL for result #%d. Skipping.", idx)
            continue

        success = False
        for url in urls_to_try:
            filename = _sanitise_filename(url)
            dest = save_dir / filename

            if dest.exists() and dest.stat().st_size >= MIN_FILE_SIZE:
                logger.info("Cache hit: %s", dest.name)
                downloaded.append(
                    {**result, "image_path": str(dest), "downloaded_url": url}
                )
                success = True
                break

            if _download_single(url, dest):
                # Resolve the actual saved path (extension may have changed)
                # Find the file with the hash prefix
                hash_prefix = hashlib.sha256(url.encode()).hexdigest()[:12]
                matching = list(save_dir.glob(f"{hash_prefix}_*"))
                actual_path = str(matching[0]) if matching else str(dest)

                downloaded.append(
                    {**result, "image_path": actual_path, "downloaded_url": url}
                )
                success = True
                break

        if not success:
            logger.warning(
                "Could not download any image for result #%d: %s",
                idx,
                page_url[:80],
            )

    logger.info(
        "Downloaded %d/%d candidate images into '%s'.",
        len(downloaded),
        len(search_results),
        save_dir,
    )

    return downloaded
