"""
blockchain/verify.py
====================
Stand-alone verification utilities for the face-chain evidence ledger.

Can be imported programmatically or invoked directly as a CLI script:

    python -m blockchain.verify [--chain chain/blockchain.json] [--hash <hex>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from blockchain.blockchain import Blockchain, CHAIN_FILE
from utils.hashing import verify_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification result type
# ---------------------------------------------------------------------------


class VerificationResult:
    """Immutable container for a verification outcome."""

    def __init__(
        self,
        chain_valid: bool,
        hash_valid: bool | None,
        message: str,
        block: dict[str, Any] | None = None,
    ) -> None:
        self.chain_valid = chain_valid
        self.hash_valid = hash_valid  # None when not checked
        self.message = message
        self.block = block

    @property
    def is_valid(self) -> bool:
        """Overall validity: chain must be intact and, if hash was given, it must match."""
        if not self.chain_valid:
            return False
        if self.hash_valid is not None and not self.hash_valid:
            return False
        return True

    @property
    def status_string(self) -> str:
        return "VALID" if self.is_valid else "INVALID"

    def __repr__(self) -> str:
        return f"VerificationResult(status={self.status_string}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_chain(chain_file: str | Path | None = None) -> VerificationResult:
    """Load and verify the blockchain stored at *chain_file*.

    Parameters
    ----------
    chain_file:
        Path to the persisted chain JSON.  Defaults to ``chain/blockchain.json``.

    Returns
    -------
    VerificationResult
    """
    path = Path(chain_file) if chain_file else CHAIN_FILE

    if not path.exists():
        logger.error("Chain file not found: %s", path)
        return VerificationResult(
            chain_valid=False,
            hash_valid=None,
            message=f"Chain file not found: {path}",
        )

    try:
        bc = Blockchain(chain_file=path)
    except ValueError as exc:
        logger.error("Failed to load chain: %s", exc)
        return VerificationResult(
            chain_valid=False,
            hash_valid=None,
            message=f"Chain load error: {exc}",
        )

    chain_ok = bc.verify()
    msg = (
        f"Chain integrity {'PASSED' if chain_ok else 'FAILED'} "
        f"({bc.length} blocks)."
    )
    logger.info(msg)

    return VerificationResult(
        chain_valid=chain_ok,
        hash_valid=None,
        message=msg,
    )


def verify_evidence(
    evidence_record: dict[str, Any],
    expected_hash: str,
    chain_file: str | Path | None = None,
) -> VerificationResult:
    """Verify both the blockchain and the evidence record's hash integrity.

    Steps:
    1. Recompute the SHA-256 hash of *evidence_record* and compare to
       *expected_hash*.
    2. Verify the blockchain chain integrity.
    3. Confirm *expected_hash* appears in the chain.

    Parameters
    ----------
    evidence_record:
        The original evidence dict to re-hash.
    expected_hash:
        The stored SHA-256 hex digest to validate against.
    chain_file:
        Optional override for the chain file path.

    Returns
    -------
    VerificationResult
    """
    # --------------------------------------------------- hash re-verification
    hash_ok = verify_hash(evidence_record, expected_hash)

    if not hash_ok:
        return VerificationResult(
            chain_valid=False,
            hash_valid=False,
            message="Evidence hash MISMATCH — record may have been tampered with.",
        )

    # --------------------------------------------------- chain verification
    chain_result = verify_chain(chain_file)
    if not chain_result.chain_valid:
        return VerificationResult(
            chain_valid=False,
            hash_valid=True,
            message=f"Hash OK but chain INVALID: {chain_result.message}",
        )

    # ------------------------------------------- confirm hash is in the chain
    path = Path(chain_file) if chain_file else CHAIN_FILE
    bc = Blockchain(chain_file=path)
    block = bc.find_block_by_evidence_hash(expected_hash)

    if block is None:
        return VerificationResult(
            chain_valid=True,
            hash_valid=True,
            message=(
                f"Hash OK and chain intact, but evidence hash {expected_hash[:16]}… "
                "was NOT found in any block."
            ),
        )

    return VerificationResult(
        chain_valid=True,
        hash_valid=True,
        message=(
            f"FULLY VERIFIED — evidence hash found in block #{block['index']} "
            f"(timestamp: {block['timestamp']})."
        ),
        block=block,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m blockchain.verify",
        description="Verify the face-chain evidence blockchain.",
    )
    parser.add_argument(
        "--chain",
        default=str(CHAIN_FILE),
        help="Path to the blockchain JSON file (default: chain/blockchain.json).",
    )
    parser.add_argument(
        "--hash",
        dest="evidence_hash",
        default=None,
        help="Evidence SHA-256 hash to verify against the chain.",
    )
    parser.add_argument(
        "--record",
        dest="record_file",
        default=None,
        help="Path to the JSON evidence record file (required with --hash).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """CLI entrypoint for blockchain verification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.evidence_hash and args.record_file:
        try:
            with open(args.record_file, "r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"ERROR: Cannot read record file: {exc}", file=sys.stderr)
            sys.exit(1)

        result = verify_evidence(record, args.evidence_hash, chain_file=args.chain)
    else:
        result = verify_chain(chain_file=args.chain)

    print("\n" + "=" * 50)
    print(f"  STATUS  : {result.status_string}")
    print(f"  MESSAGE : {result.message}")
    print("=" * 50 + "\n")

    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
