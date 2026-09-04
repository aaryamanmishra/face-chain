"""
blockchain/blockchain.py
========================
Lightweight local blockchain for tamper-evident evidence storage.

Each block is a Python dict containing:

    {
        "index"         : int,
        "timestamp"     : str (ISO 8601 UTC),
        "previous_hash" : str (hex SHA-256),
        "data"          : Any (JSON-serialisable),
        "hash"          : str (hex SHA-256),
    }

The chain is persisted as a JSON file in the ``chain/`` directory and can be
reloaded and re-verified at any time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAIN_DIR = Path(__file__).resolve().parent.parent / "chain"
CHAIN_FILE = CHAIN_DIR / "blockchain.json"
GENESIS_PREV_HASH = "0" * 64  # conventional all-zero previous hash for block 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _compute_hash(block: dict[str, Any]) -> str:
    """Compute SHA-256 over the *content* fields of a block.

    The ``hash`` field itself is excluded from the digest to avoid circular
    self-reference.  Keys are sorted for determinism.

    Parameters
    ----------
    block:
        Block dict (the ``hash`` key is ignored if present).

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    content = {k: v for k, v in block.items() if k != "hash"}
    raw = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Blockchain class
# ---------------------------------------------------------------------------


class Blockchain:
    """A simple append-only blockchain backed by a local JSON file.

    Usage
    -----
    ::

        bc = Blockchain()
        bc.add_block({"evidence_hash": "…", "matched_url": "…"})
        bc.save()
        valid = bc.verify()
    """

    def __init__(self, chain_file: str | Path | None = None) -> None:
        """Initialise the blockchain.

        If *chain_file* already exists on disk, the persisted chain is loaded.
        Otherwise a fresh chain with a genesis block is created.

        Parameters
        ----------
        chain_file:
            Path to the JSON file used for persistence.
            Defaults to ``chain/blockchain.json``.
        """
        self._chain_path = Path(chain_file) if chain_file else CHAIN_FILE
        self._chain_path.parent.mkdir(parents=True, exist_ok=True)

        if self._chain_path.exists():
            logger.info("Loading existing chain from %s", self._chain_path)
            self._chain: list[dict[str, Any]] = self._load()
        else:
            logger.info("No existing chain found. Creating genesis block.")
            self._chain = []
            self._create_genesis()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chain(self) -> list[dict[str, Any]]:
        """Read-only view of the chain."""
        return list(self._chain)

    @property
    def length(self) -> int:
        """Number of blocks in the chain."""
        return len(self._chain)

    @property
    def latest_block(self) -> dict[str, Any]:
        """The most recently added block."""
        return self._chain[-1]

    # ------------------------------------------------------------------
    # Genesis
    # ------------------------------------------------------------------

    def _create_genesis(self) -> None:
        """Create and append the genesis (index=0) block."""
        genesis: dict[str, Any] = {
            "index": 0,
            "timestamp": _utc_now(),
            "previous_hash": GENESIS_PREV_HASH,
            "data": {
                "message": "Genesis block — face-chain evidence ledger",
                "version": "1.0.0",
            },
        }
        genesis["hash"] = _compute_hash(genesis)
        self._chain.append(genesis)
        logger.debug("Genesis block created | hash=%s…", genesis["hash"][:16])

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def add_block(self, data: Any) -> dict[str, Any]:
        """Append a new block containing *data* to the chain.

        Parameters
        ----------
        data:
            Arbitrary JSON-serialisable payload (e.g. an evidence record).

        Returns
        -------
        dict
            The newly created block.

        Raises
        ------
        TypeError
            If *data* is not JSON-serialisable.
        """
        # Validate JSON-serialisability up-front for a clear error message.
        try:
            json.dumps(data)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Block data is not JSON-serialisable: {exc}") from exc

        new_block: dict[str, Any] = {
            "index": self.length,
            "timestamp": _utc_now(),
            "previous_hash": self.latest_block["hash"],
            "data": data,
        }
        new_block["hash"] = _compute_hash(new_block)
        self._chain.append(new_block)

        logger.info(
            "Block #%d added | hash=%s… | prev=%s…",
            new_block["index"],
            new_block["hash"][:16],
            new_block["previous_hash"][:16],
        )

        return new_block

    def verify(self) -> bool:
        """Verify the integrity of the entire chain.

        Checks:
        1. Each block's stored hash matches the recomputed hash of its fields.
        2. Each block's ``previous_hash`` matches the preceding block's hash.
        3. The genesis block's ``previous_hash`` is the canonical all-zero string.

        Returns
        -------
        bool
            ``True`` if the chain is intact and untampered; ``False`` otherwise.
        """
        if not self._chain:
            logger.error("Chain is empty — cannot verify.")
            return False

        # ---------------------------------- verify genesis previous_hash
        if self._chain[0]["previous_hash"] != GENESIS_PREV_HASH:
            logger.error("Genesis block previous_hash is invalid.")
            return False

        for i, block in enumerate(self._chain):
            # ----------------------------------- recompute and compare hash
            expected_hash = _compute_hash(block)
            if block["hash"] != expected_hash:
                logger.error(
                    "Hash mismatch at block #%d: stored=%s… expected=%s…",
                    i,
                    block["hash"][:16],
                    expected_hash[:16],
                )
                return False

            # --------------------------------- verify linkage (skip genesis)
            if i > 0:
                prev_block = self._chain[i - 1]
                if block["previous_hash"] != prev_block["hash"]:
                    logger.error(
                        "Broken link between block #%d and #%d.", i - 1, i
                    )
                    return False

        logger.info("Blockchain verification PASSED (%d blocks).", len(self._chain))
        return True

    def find_block_by_evidence_hash(
        self, evidence_hash: str
    ) -> dict[str, Any] | None:
        """Search all blocks for one whose data contains *evidence_hash*.

        Parameters
        ----------
        evidence_hash:
            The SHA-256 hex digest of an evidence record to look up.

        Returns
        -------
        dict or None
            The matching block, or ``None`` if not found.
        """
        for block in self._chain:
            data = block.get("data", {})
            if isinstance(data, dict) and data.get("evidence_hash") == evidence_hash:
                return block
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, chain_file: str | Path | None = None) -> Path:
        """Serialise the chain to a JSON file.

        Parameters
        ----------
        chain_file:
            Override the default save path.

        Returns
        -------
        Path
            The path where the chain was saved.
        """
        dest = Path(chain_file) if chain_file else self._chain_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(
                {"chain": self._chain, "length": self.length},
                fh,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Chain saved to %s (%d blocks).", dest, self.length)
        return dest

    def _load(self) -> list[dict[str, Any]]:
        """Load and return the chain from the JSON file on disk.

        Raises
        ------
        ValueError
            If the file cannot be parsed or the chain structure is invalid.
        """
        try:
            with open(self._chain_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Chain file is corrupted (invalid JSON): {exc}"
            ) from exc

        chain = payload.get("chain")
        if not isinstance(chain, list):
            raise ValueError("Chain file missing 'chain' list.")

        logger.debug("Loaded %d block(s) from disk.", len(chain))
        return chain

    def reload(self) -> None:
        """Reload the chain from disk, discarding any in-memory changes."""
        self._chain = self._load()
        logger.info("Chain reloaded from disk (%d blocks).", len(self._chain))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Blockchain(length={self.length}, "
            f"latest_hash={self.latest_block['hash'][:16]}…)"
        )
