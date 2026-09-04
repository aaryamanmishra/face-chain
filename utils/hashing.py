"""
utils/hashing.py
================
Cryptographic hashing utilities for the face-chain pipeline.

Provides deterministic SHA-256 digests over arbitrary Python objects by
first serialising them to a canonically-sorted JSON string, ensuring that
the same logical evidence record always produces the same hash regardless
of key insertion order.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_hash(record: dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 hexdigest for *record*.

    The record is first serialised to a UTF-8 JSON string with keys sorted
    alphabetically.  This makes the digest stable across different Python
    interpreter runs and dict ordering choices.

    Parameters
    ----------
    record:
        An arbitrary JSON-serialisable mapping.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    TypeError
        If *record* contains values that cannot be serialised to JSON.
    ValueError
        If *record* is empty.
    """
    if not record:
        raise ValueError("Cannot hash an empty record.")

    try:
        canonical_json: str = json.dumps(record, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialise record to JSON: %s", exc)
        raise TypeError(f"Record is not JSON-serialisable: {exc}") from exc

    raw_bytes: bytes = canonical_json.encode("utf-8")
    digest: str = hashlib.sha256(raw_bytes).hexdigest()

    logger.debug(
        "SHA-256 digest generated | record_keys=%s | digest=%s…",
        list(record.keys()),
        digest[:16],
    )
    return digest


def verify_hash(record: dict[str, Any], expected_hash: str) -> bool:
    """Verify that *record* produces *expected_hash*.

    Parameters
    ----------
    record:
        The evidence record to verify.
    expected_hash:
        Previously stored SHA-256 hexdigest.

    Returns
    -------
    bool
        ``True`` if the computed digest matches *expected_hash*.
    """
    computed: str = generate_hash(record)
    match: bool = computed == expected_hash.lower()

    if match:
        logger.info("Hash verification PASSED for record.")
    else:
        logger.warning(
            "Hash verification FAILED | expected=%s | computed=%s",
            expected_hash,
            computed,
        )

    return match


def hash_file(file_path: str) -> str:
    """Compute the SHA-256 digest of a file's raw bytes.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the file.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    IOError
        If the file cannot be read.
    """
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except FileNotFoundError:
        logger.error("File not found for hashing: %s", file_path)
        raise
    except IOError as exc:
        logger.error("IO error while hashing file %s: %s", file_path, exc)
        raise

    digest = h.hexdigest()
    logger.debug("File hash | path=%s | digest=%s…", file_path, digest[:16])
    return digest
