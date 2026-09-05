"""
app.py
======
face-chain — End-to-end facial match + blockchain evidence pipeline.

Usage
-----
Command-line (pipeline mode):
    python app.py --image path/to/face.jpg [--threshold 0.40] [--top-k 10]

Flask API mode:
    python app.py --serve [--port 5000]

Environment Variables
---------------------
SERPAPI_KEY     (required) Your SerpAPI key.
LOG_LEVEL       (optional) Python logging level, default INFO.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging setup — must happen before any local imports that use logging.
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger with coloured output when ``colorlog`` is available."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    try:
        import colorlog  # noqa: PLC0415

        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s "
                "%(cyan)s%(name)s%(reset)s: %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "white",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logging.root.setLevel(numeric_level)
    logging.root.handlers = [handler]


_setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from face.detector import extract_embedding
from face.matcher import compare_embeddings, rank_candidates
from search.serpapi_search import search_by_image
from search.image_downloader import download_candidates
from blockchain.blockchain import Blockchain
from blockchain.verify import verify_evidence, VerificationResult
from utils.hashing import generate_hash, hash_file
from utils.social_media import annotate_platform, filter_social_media, is_specific_post

# Testnet anchoring — optional; gracefully absent when web3 is not installed.
try:
    from blockchain.testnet_anchor import anchor_hash as _anchor_hash  # noqa: PLC0415
    _TESTNET_ANCHOR_AVAILABLE = True
except ImportError:
    _TESTNET_ANCHOR_AVAILABLE = False
    logger.debug("web3 not installed — Sepolia anchoring will be skipped.")

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
UPLOADS_DIR = ROOT / "uploads"
CHAIN_DIR = ROOT / "chain"

for _d in (RESULTS_DIR, UPLOADS_DIR, CHAIN_DIR, ROOT / "downloads"):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _build_evidence_record(
    source_image: str,
    matched_url: str,
    similarity: float,
    candidate_image: str,
    source_embedding: np.ndarray,
    platform: str | None = None,
    source_image_hash: str | None = None,
) -> dict[str, Any]:
    """Construct a JSON-serialisable evidence record.

    Parameters
    ----------
    source_image:
        Path to the query face image.
    matched_url:
        Web URL of the best-matching result.
    similarity:
        Cosine similarity score (0-1).
    candidate_image:
        Local path of the downloaded candidate image.
    source_embedding:
        The ArcFace embedding vector of the source face.
    platform:
        Canonical social media platform name, or ``None`` for general web.
    source_image_hash:
        SHA-256 hex digest of the raw source image file.

    Returns
    -------
    dict
        Tamper-evident evidence record ready for hashing and blockchain storage.
    """
    record: dict[str, Any] = {
        "source_image": str(source_image),
        "matched_url": matched_url,
        "similarity": round(float(similarity), 6),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "candidate_image": str(candidate_image),
        "source_embedding_norm": round(float(np.linalg.norm(source_embedding)), 6),
    }
    if platform is not None:
        record["platform"] = platform
    if source_image_hash is not None:
        record["source_image_hash"] = source_image_hash
    return record


def _print_banner(title: str, width: int = 50) -> None:
    """Print a centred title banner."""
    bar = "-" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    image_path: str | Path,
    threshold: float = 0.40,
    top_k: int = 10,
    chain_file: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the full face-chain pipeline.

    Steps
    -----
    1.  Validate and load the source image.
    2.  Extract ArcFace embedding from the source face.
    3.  Reverse-image search via SerpAPI Google Lens.
    4.  Download candidate images (thumbnails preferred).
    5.  Extract embeddings from all candidates.
    6.  Rank by cosine similarity.
    7.  Build evidence record for the best match.
    8.  SHA-256 hash the evidence record.
    9.  Append to the local blockchain.
    10. Verify blockchain integrity.
    11. Persist results to ``results/result.json``.
    12. Print formatted summary.

    Parameters
    ----------
    image_path:
        Path to the input face image.
    threshold:
        Cosine similarity threshold for declaring a match.
    top_k:
        Maximum number of search results to process.
    chain_file:
        Override the default blockchain JSON path.

    Returns
    -------
    dict
        Final pipeline result containing all relevant fields.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    RuntimeError
        On unrecoverable pipeline errors (e.g. SerpAPI unreachable).
    """
    image_path = Path(image_path)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("face-chain pipeline starting | image=%s", image_path.name)
    logger.info("=" * 60)

    # ─────────────────────────────────────────── Step 1: validate image
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    # ─────────────────────────────────────── Step 2: extract source embedding
    logger.info("[1/9] Extracting face embedding from source image…")
    try:
        source_result = extract_embedding(image_path)
    except ValueError as exc:
        raise RuntimeError(f"Face detection failed: {exc}") from exc

    source_embedding: np.ndarray = source_result["embedding"]
    logger.info(
        "     Source face detected | bbox=%s | det_score=%.3f | faces=%d",
        source_result["bbox"],
        source_result["det_score"],
        source_result["num_faces"],
    )

    # ────────────────────────────────────── Step 3: Google Lens search
    logger.info("[2/9] Searching Google Lens via SerpAPI…")
    search_results = search_by_image(image_path)

    if not search_results:
        raise RuntimeError(
            "SerpAPI returned no results. "
            "Check your SERPAPI_KEY and that the image contains a clear face."
        )

    logger.info("     %d result(s) received.", len(search_results))

    # ──────────────────────────────────────── Step 4: download candidates
    logger.info("[3/9] Downloading candidate images…")
    downloaded = download_candidates(search_results[:top_k])

    if not downloaded:
        raise RuntimeError("No candidate images could be downloaded.")

    logger.info("     %d image(s) downloaded.", len(downloaded))

    # ─────────────────────────────────── Step 5: embed candidate images
    logger.info("[4/9] Generating candidate face embeddings…")
    candidates_with_embeddings: list[dict[str, Any]] = []

    for cand in downloaded:
        img_path = cand.get("image_path", "")
        try:
            emb_result = extract_embedding(img_path)
            candidates_with_embeddings.append(
                {
                    **cand,
                    "embedding": emb_result["embedding"],
                    "cand_det_score": emb_result["det_score"],
                }
            )
            logger.debug(
                "     Embedded: %s (score=%.3f)",
                Path(img_path).name,
                emb_result["det_score"],
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.debug("     Skipping %s — %s", Path(img_path).name, exc)

    if not candidates_with_embeddings:
        raise RuntimeError(
            "No faces were detected in any of the candidate images. "
            "Try a clearer source image or broaden the search."
        )

    logger.info(
        "     %d/%d candidates had detectable faces.",
        len(candidates_with_embeddings),
        len(downloaded),
    )

    # ─────────────────────────────────────── Step 6: annotate platforms + rank
    logger.info("[5/9] Ranking candidates by cosine similarity…")

    # Annotate every candidate with its social media platform (None if not social).
    candidates_annotated = annotate_platform(
        candidates_with_embeddings, url_key="link"
    )

    # Global rank across ALL candidates (unchanged behaviour).
    ranked = rank_candidates(source_embedding, candidates_annotated, threshold)

    best = ranked[0]
    logger.info(
        "     Overall best | similarity=%.4f | match=%s | url=%s",
        best["similarity"],
        best["match"],
        best.get("link", "")[:80],
    )

    # ───────────────────────── Step 6b: social media + post-specificity ranking
    logger.info("[5b] Identifying specific social-media post candidates…")

    # Tier-1: social candidates whose URL is a verified specific post
    # (e.g. /p/<id>, /status/<id>, /posts/<slug>, watch?v=…)
    specific_post_candidates: list[dict[str, Any]] = [
        c for c in ranked
        if c.get("platform") is not None and c.get("is_specific_post", False)
    ]

    # Tier-2 fallback: any social-media URL (generic pages included)
    any_social_candidates: list[dict[str, Any]] = [
        c for c in ranked if c.get("platform") is not None
    ]

    # best_social is the highest-similarity SPECIFIC post;
    # best_social_generic is the Tier-2 fallback (not used for evidence).
    best_social: dict[str, Any] | None = (
        specific_post_candidates[0] if specific_post_candidates else None
    )
    best_social_generic: dict[str, Any] | None = (
        any_social_candidates[0] if any_social_candidates else None
    )

    if best_social:
        logger.info(
            "     Best specific social post | platform=%s | similarity=%.4f | url=%s",
            best_social.get("platform"),
            best_social["similarity"],
            best_social.get("link", "")[:80],
        )
    elif best_social_generic:
        logger.warning(
            "     Social media URLs found but NONE are specific posts. "
            "Best generic social URL: %s (%s)",
            best_social_generic.get("link", "")[:80],
            best_social_generic.get("post_rejection", "unknown reason"),
        )
    else:
        logger.warning(
            "     No social media candidates found among the search results."
        )

    # ─────────────────────────────────── Step 7: SHA-256 hash of source image
    logger.info("[6/9] Building evidence record…")
    try:
        source_image_hash = hash_file(str(image_path))
    except (FileNotFoundError, IOError) as exc:
        logger.warning("Could not hash source image file: %s", exc)
        source_image_hash = None

    # The primary evidence record anchors to the best SPECIFIC SOCIAL POST
    # when one exists; falls back to overall best otherwise.
    # Generic social pages (popular/, explore, hashtag…) are never used.
    primary = best_social if best_social is not None else best
    evidence_is_specific_post: bool = best_social is not None

    evidence_record = _build_evidence_record(
        source_image=str(image_path),
        matched_url=primary.get("link", primary.get("downloaded_url", "")),
        similarity=primary["similarity"],
        candidate_image=primary.get("image_path", ""),
        source_embedding=source_embedding,
        platform=primary.get("platform"),
        source_image_hash=source_image_hash,
    )

    # ───────────────────────────────────── Step 8: hash evidence record
    logger.info("[7/9] Hashing evidence record (SHA-256)…")
    evidence_hash = generate_hash(evidence_record)
    logger.info("     Evidence hash: %s…", evidence_hash[:32])

    # ────────────────────────────────────── Step 9: store in blockchain
    logger.info("[8/9] Writing evidence to blockchain…")
    bc = Blockchain(chain_file=chain_file)

    block_data: dict[str, Any] = {
        "evidence_hash": evidence_hash,
        "matched_url": evidence_record["matched_url"],
        "similarity": evidence_record["similarity"],
        "timestamp": evidence_record["timestamp"],
        "source_image": evidence_record["source_image"],
        "source_image_hash": source_image_hash,
        "evidence_is_specific_post": evidence_is_specific_post,
    }
    if evidence_record.get("platform"):
        block_data["platform"] = evidence_record["platform"]

    # Audit fields: overall best + best generic social (if any)
    block_data["overall_best_url"] = best.get("link", best.get("downloaded_url", ""))
    block_data["overall_best_similarity"] = round(float(best["similarity"]), 6)
    if best_social_generic and best_social is None:
        block_data["best_generic_social_url"] = best_social_generic.get("link", "")
        block_data["best_generic_social_rejection"] = best_social_generic.get("post_rejection", "")

    new_block = bc.add_block(block_data)
    bc.save()

    logger.info(
        "     Block #%d written | chain_length=%d",
        new_block["index"],
        bc.length,
    )

    # ─────────────────────────────── Step 10: verify blockchain
    logger.info("[9/9] Verifying blockchain integrity…")
    verification: VerificationResult = verify_evidence(
        evidence_record, evidence_hash, chain_file=chain_file or bc._chain_path
    )

    blockchain_status = verification.status_string
    logger.info("     Blockchain: %s | %s", blockchain_status, verification.message)

    # ─────────────────────── Step 11: anchor to Ethereum Sepolia (optional)
    sepolia_tx_hash: str | None = None
    if _TESTNET_ANCHOR_AVAILABLE:
        logger.info("[10/10] Anchoring evidence hash to Ethereum Sepolia testnet…")
        _result = _anchor_hash(evidence_hash)
        sepolia_tx_hash = _result if _result else None
        if sepolia_tx_hash:
            print(f"\n  Sepolia TX : {sepolia_tx_hash}")
            print(f"  Etherscan  : https://sepolia.etherscan.io/tx/{sepolia_tx_hash}\n")
        else:
            logger.warning(
                "     Sepolia anchoring skipped (see warnings above). "
                "Local pipeline result is unaffected."
            )

    # ─────────────────────────────────────── Compile final result
    elapsed = round(time.perf_counter() - t0, 2)

    final_result: dict[str, Any] = {
        "pipeline_version": "1.1.0",
        "elapsed_seconds": elapsed,
        "source_image": str(image_path),
        "source_image_hash": source_image_hash,
        "face_detected": True,
        "source_det_score": source_result["det_score"],
        "num_search_results": len(search_results),
        "num_candidates_downloaded": len(downloaded),
        "num_candidates_with_face": len(candidates_with_embeddings),
        # ── overall best (may be any site)
        "best_match": {
            "url": best.get("link", best.get("downloaded_url", "")),
            "similarity": round(float(best["similarity"]), 6),
            "match": best["match"],
            "image_path": best.get("image_path", ""),
            "title": best.get("title", ""),
            "source": best.get("source", ""),
            "platform": best.get("platform"),
            "is_specific_post": best.get("is_specific_post", False),
        },
        # ── best verified SPECIFIC social-media post
        "best_social_match": (
            {
                "url": best_social.get("link", best_social.get("downloaded_url", "")),
                "platform": best_social.get("platform"),
                "similarity": round(float(best_social["similarity"]), 6),
                "match": best_social["match"],
                "image_path": best_social.get("image_path", ""),
                "title": best_social.get("title", ""),
                "is_specific_post": True,
            }
            if best_social
            else None
        ),
        # ── best generic social URL (rejected from evidence; informational only)
        "best_generic_social": (
            {
                "url": best_social_generic.get("link", best_social_generic.get("downloaded_url", "")),
                "platform": best_social_generic.get("platform"),
                "similarity": round(float(best_social_generic["similarity"]), 6),
                "is_specific_post": False,
                "rejection_reason": best_social_generic.get("post_rejection"),
            }
            if best_social_generic and best_social is None
            else None
        ),
        "all_ranked_matches": [
            {
                "rank": i + 1,
                "url": r.get("link", r.get("downloaded_url", "")),
                "similarity": round(float(r["similarity"]), 6),
                "match": r["match"],
                "title": r.get("title", ""),
                "platform": r.get("platform"),
                "is_specific_post": r.get("is_specific_post", False),
                "post_rejection": r.get("post_rejection"),
            }
            for i, r in enumerate(ranked)
        ],
        "evidence_record": evidence_record,
        "evidence_hash": evidence_hash,
        "blockchain": {
            "block_index": new_block["index"],
            "block_hash": new_block["hash"],
            "status": blockchain_status,
            "chain_length": bc.length,
            "chain_file": str(bc._chain_path),
            # Sepolia anchoring (None when skipped or web3 unavailable)
            "sepolia_tx_hash": sepolia_tx_hash,
            "sepolia_etherscan": (
                f"https://sepolia.etherscan.io/tx/{sepolia_tx_hash}"
                if sepolia_tx_hash else None
            ),
        },
    }

    # ──────────────────────────────────────────────── save to disk
    result_path = RESULTS_DIR / "result.json"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(final_result, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to: %s", result_path)

    # ──────────────────────────────────────────────── print summary
    _print_summary(final_result)

    return final_result


def _print_summary(result: dict[str, Any]) -> None:
    """Print a formatted pipeline summary to stdout."""
    best_info = result["best_match"]
    social_info = result.get("best_social_match")
    bc_info = result["blockchain"]

    W = 58  # banner width

    # ─── Header banner ──────────────────────────────────────────
    if best_info["match"] or (social_info and social_info["match"]):
        _print_banner("FACE MATCH FOUND", W)
    else:
        _print_banner("PIPELINE COMPLETE — LOW CONFIDENCE", W)

    # ─── Overall best match ──────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  OVERALL BEST MATCH")
    print(f"{'─'*W}")
    best_platform = best_info.get("platform") or "General Web"
    flag_o = "✓" if best_info["match"] else "✗"
    print(f"  Platform   : {best_platform}")
    print(f"  URL        : {best_info['url']}")
    print(f"  Similarity : {best_info['similarity']:.4f}  [{flag_o}]")
    if best_info.get("title"):
        print(f"  Title      : {best_info['title']}")

    # ─── Best verified specific social-media post ───────────────
    print(f"\n{'─'*W}")
    print("  BEST VERIFIED SPECIFIC SOCIAL-MEDIA POST")
    print(f"{'─'*W}")
    if social_info:
        flag_s = "✓" if social_info["match"] else "✗"
        print(f"  Platform   : {social_info['platform']}")
        print(f"  URL        : {social_info['url']}")
        print(f"  Similarity : {social_info['similarity']:.4f}  [{flag_s}]")
        if social_info.get("title"):
            print(f"  Title      : {social_info['title']}")
    else:
        # Check if there was a generic social URL that was rejected
        generic = result.get("best_generic_social")
        if generic:
            print("  No verified specific social-media post found.")
            print(f"  (Rejected generic page: {generic['platform']} — {generic['url'][:70]}")
            print(f"   Reason: {generic.get('rejection_reason', 'unknown')}")
            print( "   Use a clearer face image for better social media results.)")
        else:
            print("  No verified specific social-media post found.")
            print("  (No social media URLs appeared in the search results.)")

    # ─── Evidence & blockchain ───────────────────────────────────
    print(f"\n{'─'*W}")
    print("  EVIDENCE & BLOCKCHAIN")
    print(f"{'─'*W}")
    ev = result["evidence_record"]
    print(f"  Evidence URL       : {ev.get('matched_url', 'N/A')}")
    print(f"  Evidence Platform  : {ev.get('platform', 'General Web')}")
    print(f"  Evidence Similarity: {ev.get('similarity', 0):.4f}")
    if result.get("source_image_hash"):
        print(f"  Source Image SHA256: {result['source_image_hash'][:32]}…")
    print(f"  Evidence Hash      : {result['evidence_hash']}")
    print(f"  Blockchain Status  : {bc_info['status']}")
    print(f"  Block Index        : #{bc_info['block_index']}")
    print(f"  Chain Length       : {bc_info['chain_length']} block(s)")

    # ─── All ranked candidates ───────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ALL RANKED CANDIDATES ({len(result['all_ranked_matches'])})")
    print(f"{'─'*W}")
    for m in result["all_ranked_matches"]:
        face_flag = "✓" if m["match"] else "✗"
        post_flag = "📌" if m.get("is_specific_post") else "  "
        plat = f"[{m['platform']}]" if m.get("platform") else "[web]"
        url_short = (m["url"] or "N/A")[:52]
        print(
            f"  [{face_flag}]{post_flag} #{m['rank']:2d}  sim={m['similarity']:.4f}  "
            f"{plat:<18}  {url_short}"
        )
    print(f"  (✓=face match threshold met  📌=specific post URL)")

    print(f"\n{'─'*W}")
    print(f"  Elapsed : {result['elapsed_seconds']}s")
    print(f"  Saved   : results/result.json")
    print(f"{'─'*W}\n")


# ---------------------------------------------------------------------------
# Flask API server
# ---------------------------------------------------------------------------


def create_flask_app() -> Any:
    """Create and return the Flask application.

    Routes
    ------
    POST /analyse
        Accepts a multipart/form-data request with ``file`` (image) and
        optional ``threshold`` (float, default 0.40) form fields.
        Returns the pipeline result as JSON.

    GET /health
        Returns ``{"status": "ok"}``.

    GET /chain
        Returns the current blockchain as JSON.

    GET /verify/<evidence_hash>
        Verifies a specific evidence hash against the chain.
    """
    try:
        from flask import Flask, request, jsonify  # noqa: PLC0415
        from flask_cors import CORS  # noqa: PLC0415
    except ImportError as exc:
        logger.critical("Flask not installed: %s", exc)
        raise

    app = Flask("face-chain")
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok", "service": "face-chain"})

    @app.route("/analyse", methods=["POST"])
    def analyse() -> Any:
        """Analyse an uploaded face image through the full pipeline."""
        if "file" not in request.files:
            return jsonify({"error": "No file field in request."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename."}), 400

        threshold = float(request.form.get("threshold", 0.40))
        top_k = int(request.form.get("top_k", 10))

        # Save upload
        upload_path = UPLOADS_DIR / file.filename
        file.save(str(upload_path))
        logger.info("Received upload: %s", upload_path.name)

        try:
            result = run_pipeline(
                image_path=upload_path,
                threshold=threshold,
                top_k=top_k,
            )
            # numpy arrays are not JSON-serialisable; already excluded by pipeline
            return jsonify(result)
        except RuntimeError as exc:
            logger.error("Pipeline error: %s", exc)
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error: %s", exc)
            return jsonify({"error": "Internal server error."}), 500

    @app.route("/chain", methods=["GET"])
    def get_chain() -> Any:
        """Return the full blockchain as JSON."""
        from blockchain.blockchain import Blockchain, CHAIN_FILE  # noqa: PLC0415

        bc = Blockchain(chain_file=CHAIN_FILE)
        return jsonify({"chain": bc.chain, "length": bc.length})

    @app.route("/verify/<evidence_hash>", methods=["GET"])
    def verify(evidence_hash: str) -> Any:
        """Verify a specific evidence hash against the chain."""
        # We can only verify chain integrity without the original record.
        from blockchain.verify import verify_chain  # noqa: PLC0415

        result = verify_chain()
        block = Blockchain()._chain  # noqa: SLF001

        matching_blocks = [
            b
            for b in block
            if isinstance(b.get("data"), dict)
            and b["data"].get("evidence_hash") == evidence_hash
        ]

        return jsonify(
            {
                "evidence_hash": evidence_hash,
                "chain_valid": result.chain_valid,
                "found_in_chain": len(matching_blocks) > 0,
                "blocks": matching_blocks,
                "status": "VALID" if (result.chain_valid and matching_blocks) else "INVALID",
            }
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-chain",
        description="Face identity pipeline with blockchain evidence storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --image uploads/face.jpg
  python app.py --image uploads/face.jpg --threshold 0.35 --top-k 5
  python app.py --serve --port 5000
        """,
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to the input face image (required unless --serve).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.40,
        help="Cosine similarity threshold for a positive match (default: 0.40).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Maximum number of SerpAPI results to process (default: 10).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the Flask HTTP API server instead of running the pipeline.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the Flask server (default: 5000).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry-point for the face-chain CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Re-configure logging with the requested level
    _setup_logging(args.log_level)

    if args.serve:
        # ------------------------------------------ Flask server mode
        flask_app = create_flask_app()
        logger.info("Starting Flask server on port %d…", args.port)
        flask_app.run(host="0.0.0.0", port=args.port, debug=False)
        return

    # ------------------------------------------------- Pipeline mode
    if not args.image:
        parser.error("--image is required when not using --serve.")

    try:
        run_pipeline(
            image_path=args.image,
            threshold=args.threshold,
            top_k=args.top_k,
        )
    except FileNotFoundError as exc:
        logger.critical("File not found: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.critical("Pipeline failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
