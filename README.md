# face-chain

**Reverse-image face identification with independently verified social-media matches and tamper-evident blockchain storage.**

---

## Overview

`face-chain` is an end-to-end Python pipeline that takes an image containing a face, performs a genuine reverse-image search, verifies candidate images using face embeddings, identifies a specific social-media post when available, and records the resulting evidence in a local blockchain.

The pipeline combines:

- **InsightFace / ArcFace** for face detection and embeddings
- **SerpAPI Google Lens** for genuine reverse-image search
- **Cosine similarity** for independent face verification
- **SHA-256** for deterministic evidence hashing
- A **local linked blockchain** for tamper-evident storage and verification

The project was built for **Hacker House Goa 2026 – Shortlisting Task 3: Face Identification & Blockchain Verification**.

---

## What It Does

Given an input image:

```text
Input Image
     |
     v
Face Detection
     |
     v
ArcFace Embedding
     |
     v
SerpAPI Image Upload
     |
     v
Google Lens Reverse-Image Search
     |
     v
Visual Search Results
     |
     v
Download Candidate Images
     |
     v
Candidate Face Embeddings
     |
     v
Cosine Similarity Ranking
     |
     v
Specific Social-Media URL Detection
     |
     v
Best Verified Social-Media Post
     |
     v
Evidence Record
     |
     v
SHA-256 Hash
     |
     v
Local Blockchain
     |
     v
Blockchain Integrity Verification
     |
     v
Ethereum Sepolia Anchoring
```

## Ethereum Sepolia Anchoring

The evidence hash computed during the pipeline can be optionally anchored to the **Ethereum Sepolia testnet** — a public, permissionless EVM testnet — providing immutable, externally-verifiable proof that the evidence record existed at a specific point in time.

This step is **purely additive and non-blocking**: if configuration is missing or the RPC is unreachable, the local pipeline completes normally and a warning is logged.

### Chain

| Property | Value |
|---|---|
| Network | Ethereum **Sepolia** testnet |
| Chain ID | `11155111` |
| Explorer | https://sepolia.etherscan.io |
| Gas model | EIP-1559 (`maxFeePerGas` / `maxPriorityFeePerGas`) |

### Required `.env` Variables

```env
PRIVATE_KEY=0xYourPrivateKey   # Signing wallet — test wallet only, never mainnet
RPC_URL=https://ethereum-sepolia-rpc.publicnode.com  # Free public RPC
```

> **Security**: `.env` is listed in `.gitignore` and must never be committed. Use a dedicated throwaway test wallet funded exclusively with free Sepolia ETH.

### Getting Sepolia ETH (free)

- https://sepoliafaucet.com/
- https://faucets.chain.link/sepolia

### How It Works

When `PRIVATE_KEY` and `RPC_URL` are set, `anchor_hash()` sends a **0-value self-transaction** from your wallet to itself on Sepolia. The SHA-256 evidence hash is encoded as UTF-8 bytes and embedded in the transaction's `data` field. The returned transaction hash is printed to stdout and stored in `results/result.json` under `blockchain.sepolia_tx_hash`.

### Verifying an Anchored Hash

```bash
# After a pipeline run, copy the tx hash from stdout or result.json, then:
python -m blockchain.testnet_anchor verify \
    0xYourTxHash \
    <evidence_hash_from_result.json>
```

This fetches the transaction from Sepolia via RPC, decodes the `data` field, compares it to the expected hash (case/whitespace-insensitive), and prints a MATCH ✓ / MISMATCH ✗ result along with the Etherscan link.

### Anchoring Manually

```bash
python -m blockchain.testnet_anchor anchor <evidence_hash>
```