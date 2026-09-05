"""
blockchain/testnet_anchor.py
============================
Optional, self-contained module for anchoring evidence hashes to the
Ethereum Sepolia testnet.

This module is purely ADDITIVE to the face-chain pipeline.  It is imported
lazily in app.py so that a missing ``web3`` installation or absent .env vars
never break the local-only pipeline.

Public API
----------
anchor_hash(evidence_hash)   -> str   (tx_hash, or "" on any failure)
verify_onchain(tx_hash, expected_hash) -> bool

CLI
---
    python -m blockchain.testnet_anchor verify <tx_hash> <evidence_hash>

Environment variables (read from .env via python-dotenv)
---------------------------------------------------------
PRIVATE_KEY   0x-prefixed private key of the signing wallet
RPC_URL       HTTPS endpoint for an Ethereum Sepolia JSON-RPC node
              e.g. https://ethereum-sepolia-rpc.publicnode.com
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEPOLIA_CHAIN_ID: int = 11155111
ETHERSCAN_BASE: str = "https://sepolia.etherscan.io/tx"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load .env from the project root (two levels above this file)."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        logger.debug("python-dotenv not installed; relying on OS environment vars.")


def _get_web3():
    """Return a connected Web3 instance or raise ImportError / ConnectionError."""
    from web3 import Web3  # noqa: PLC0415

    rpc_url = os.getenv("RPC_URL", "").strip()
    if not rpc_url:
        raise ValueError(
            "RPC_URL is not set.  Add it to your .env file.\n"
            "  Example: RPC_URL=https://ethereum-sepolia-rpc.publicnode.com"
        )

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise ConnectionError(
            f"Cannot reach Sepolia RPC at {rpc_url!r}.\n"
            "  Check your internet connection and that RPC_URL is correct."
        )
    return w3


def _get_credentials() -> tuple[str, str]:
    """Return (checksum_address, private_key) or raise ValueError."""
    from web3 import Web3  # noqa: PLC0415

    private_key = os.getenv("PRIVATE_KEY", "").strip()
    if not private_key:
        raise ValueError(
            "PRIVATE_KEY is not set.  Add it to your .env file.\n"
            "  Example: PRIVATE_KEY=0xYourPrivateKeyHere"
        )

    try:
        account = Web3().eth.account.from_key(private_key)
    except Exception as exc:
        raise ValueError(f"PRIVATE_KEY is invalid: {exc}") from exc

    return account.address, private_key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def anchor_hash(evidence_hash: str) -> str:
    """Anchor *evidence_hash* on Ethereum Sepolia by embedding it in a
    0-value self-transaction's ``data`` field.

    The transaction uses EIP-1559 gas fields (``maxFeePerGas`` /
    ``maxPriorityFeePerGas``) and fetches the nonce via
    ``get_transaction_count(address, 'pending')`` to be mempool-safe.

    Parameters
    ----------
    evidence_hash:
        The SHA-256 hex digest string to anchor (e.g. from
        ``utils.hashing.generate_hash``).

    Returns
    -------
    str
        The transaction hash hex string (``0x...``) on success, or an empty
        string ``""`` on any failure (missing config, RPC error, etc.).
        Failures are logged as WARNINGs so the caller's pipeline continues.
    """
    _load_env()

    # ── pre-flight: check env vars ────────────────────────────────────────
    try:
        w3 = _get_web3()
        address, private_key = _get_credentials()
    except (ImportError, ValueError, ConnectionError) as exc:
        logger.warning(
            "[testnet_anchor] Skipping Sepolia anchoring — %s", exc
        )
        return ""

    # ── build EIP-1559 transaction ────────────────────────────────────────
    try:
        # Encode the evidence hash as raw UTF-8 bytes in the data field.
        data_bytes: bytes = evidence_hash.encode("utf-8")

        # Nonce: use 'pending' to handle in-flight transactions safely.
        nonce: int = w3.eth.get_transaction_count(address, "pending")

        # EIP-1559 gas estimation from the latest block's baseFeePerGas.
        latest_block = w3.eth.get_block("latest")
        base_fee: int = latest_block.get("baseFeePerGas", 0)  # type: ignore[arg-type]
        max_priority_fee: int = w3.to_wei(1, "gwei")           # 1 gwei tip
        max_fee: int = base_fee * 2 + max_priority_fee          # headroom for 1 block

        tx: dict = {
            "chainId": SEPOLIA_CHAIN_ID,
            "from": address,
            "to": address,          # self-transaction
            "value": 0,
            "data": data_bytes,
            "nonce": nonce,
            "maxPriorityFeePerGas": max_priority_fee,
            "maxFeePerGas": max_fee,
            "gas": 21_000 + 68 * len(data_bytes),  # base + per-byte cost
        }

        signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash_bytes = w3.eth.send_raw_transaction(signed.raw_transaction)
        tx_hash: str = tx_hash_bytes.hex()

        logger.info(
            "[testnet_anchor] Transaction sent | tx=%s | chain=Sepolia (%d)",
            tx_hash,
            SEPOLIA_CHAIN_ID,
        )
        return tx_hash

    except Exception as exc:  # noqa: BLE001
        _handle_tx_error(exc)
        return ""


def verify_onchain(tx_hash: str, expected_hash: str) -> bool:
    """Fetch *tx_hash* from Sepolia and verify it embeds *expected_hash*.

    Both sides of the comparison are normalised with ``.strip().lower()``
    before comparison so hex-case or whitespace differences never cause a
    false mismatch.

    Parameters
    ----------
    tx_hash:
        The ``0x``-prefixed transaction hash returned by :func:`anchor_hash`.
    expected_hash:
        The SHA-256 evidence hash to verify.

    Returns
    -------
    bool
        ``True`` if the on-chain data field matches *expected_hash*,
        ``False`` otherwise.
    """
    _load_env()

    etherscan_link = f"{ETHERSCAN_BASE}/{tx_hash}"

    try:
        w3 = _get_web3()
    except (ImportError, ValueError, ConnectionError) as exc:
        logger.warning("[testnet_anchor] Cannot verify — %s", exc)
        return False

    try:
        tx = w3.eth.get_transaction(tx_hash)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [testnet_anchor] ERROR fetching transaction: {exc}")
        print(f"  Etherscan : {etherscan_link}")
        return False

    if tx is None:
        print(f"\n  [testnet_anchor] Transaction {tx_hash} not found on Sepolia.")
        print(f"  Etherscan : {etherscan_link}")
        return False

    # Decode the data field.  The `input` attribute is bytes in web3.py ≥ 6.
    raw_input: bytes = tx.get("input", b"") or b""
    try:
        decoded: str = raw_input.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        decoded = raw_input.hex() if isinstance(raw_input, (bytes, bytearray)) else str(raw_input)

    # Normalise both sides before comparison.
    onchain_norm: str = decoded.strip().lower()
    expected_norm: str = expected_hash.strip().lower()

    matched: bool = onchain_norm == expected_norm

    # ── human-readable output ─────────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    if matched:
        print("  │  ✓  MATCH  — on-chain data matches evidence hash   │")
    else:
        print("  │  ✗  MISMATCH — on-chain data does NOT match hash   │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Expected : {expected_norm[:48]}")
    print(f"  │  On-chain : {onchain_norm[:48]}")
    print(f"  │  Etherscan: {etherscan_link}")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    logger.info(
        "[testnet_anchor] verify_onchain | match=%s | tx=%s", matched, tx_hash
    )
    return matched


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _handle_tx_error(exc: Exception) -> None:
    """Log a clear, actionable message for known transaction errors."""
    msg = str(exc)

    if "insufficient funds" in msg.lower():
        print(
            "\n  [testnet_anchor] INSUFFICIENT FUNDS — your Sepolia wallet has no ETH.\n"
            "  Get free Sepolia ETH from a faucet:\n"
            "    • https://sepoliafaucet.com/\n"
            "    • https://faucets.chain.link/sepolia\n"
            "  Then re-run the pipeline."
        )
        logger.warning("[testnet_anchor] Skipping — insufficient funds on Sepolia.")

    elif any(k in msg.lower() for k in ("timeout", "timed out", "connection", "refused")):
        print(
            "\n  [testnet_anchor] RPC TIMEOUT / CONNECTION ERROR.\n"
            "  Check that RPC_URL in .env is reachable and the node is live.\n"
            f"  RPC_URL = {os.getenv('RPC_URL', '<not set>')}"
        )
        logger.warning("[testnet_anchor] Skipping — RPC unreachable: %s", exc)

    elif "nonce" in msg.lower():
        print(
            "\n  [testnet_anchor] NONCE ERROR — a previous transaction may still be pending.\n"
            "  Wait a moment and re-run, or check your wallet on Etherscan."
        )
        logger.warning("[testnet_anchor] Skipping — nonce error: %s", exc)

    else:
        print(f"\n  [testnet_anchor] Transaction failed: {exc}")
        logger.warning("[testnet_anchor] Unexpected error: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m blockchain.testnet_anchor",
        description=(
            "Testnet anchoring utilities for face-chain evidence hashes.\n\n"
            "Sub-commands:\n"
            "  anchor  <evidence_hash>              Send anchor tx to Sepolia\n"
            "  verify  <tx_hash> <evidence_hash>    Verify an anchored tx on Sepolia"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # anchor sub-command
    anchor_p = sub.add_parser("anchor", help="Anchor an evidence hash to Sepolia.")
    anchor_p.add_argument("evidence_hash", help="SHA-256 evidence hex digest to anchor.")

    # verify sub-command
    verify_p = sub.add_parser(
        "verify", help="Verify an anchored tx matches an evidence hash."
    )
    verify_p.add_argument("tx_hash", help="0x-prefixed Sepolia transaction hash.")
    verify_p.add_argument("evidence_hash", help="SHA-256 evidence hex digest to compare.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for testnet anchoring utilities."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "anchor":
        tx_hash = anchor_hash(args.evidence_hash)
        if tx_hash:
            print(f"\n  TX Hash   : {tx_hash}")
            print(f"  Etherscan : {ETHERSCAN_BASE}/{tx_hash}\n")
            sys.exit(0)
        else:
            print("\n  Anchoring failed — see warnings above.\n")
            sys.exit(1)

    elif args.command == "verify":
        matched = verify_onchain(args.tx_hash, args.evidence_hash)
        sys.exit(0 if matched else 1)


if __name__ == "__main__":
    main()
