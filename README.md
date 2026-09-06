# face-chain 🔍⛓️

> **Task 3: Face Identification & Blockchain Verification**  
> An autonomous end-to-end OSINT & cryptographic pipeline: **Face Scan Input ➔ Genuine Reverse-Image Web Search ➔ Deepfake & Social Post Discovery ➔ Dual-Layer Blockchain Verification (Local Ledger + Ethereum Sepolia Testnet)**.

---

## 📌 Executive Summary

**face-chain** is an end-to-end investigative pipeline engineered to address digital identity verification, OSINT attribution, and tamper-evident evidence preservation. 

Given an arbitrary portrait or facial scan, the system:
1. Detects the face and computes an invariant **512-dimensional facial embedding** using InsightFace (ArcFace).
2. Performs a **live, non-hardcoded reverse-image search** across the web via Google Lens (SerpAPI).
3. Downloads discovered candidate images and **independently re-verifies them** using cosine similarity to eliminate false positives.
4. Identifies and isolates **specific social media post URLs** (e.g., Instagram Reels, X/Twitter statuses, YouTube videos) rather than generic portal links.
5. Evaluates the candidate against an **experimental deepfake detection classifier** (ViT).
6. Packages the forensic metadata into a deterministic **SHA-256 evidence fingerprint**.
7. Permanently logs the evidence into a **dual-layer blockchain architecture**:
   - **Layer 1 (Local):** An instant, zero-cost, hash-linked cryptographic ledger (`chain/blockchain.json`).
   - **Layer 2 (Public Testnet):** Immutable anchoring to the **Ethereum Sepolia Testnet** with public Etherscan verifiability.

---

## 🏛️ System Architecture & Data Flow

```text
               ┌────────────────────────┐
               │    Input Face Image    │
               └───────────┬────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │  InsightFace / ArcFace Detector  │  ──> 512-D Face Embedding Vector
          └────────────────┬─────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │  Live SerpAPI Google Lens Search │  ──> Real-world Visual Matches (Web & Social)
          └────────────────┬─────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │   Image Downloader & Embedding   │  ──> Extracts embeddings of candidates
          └────────────────┬─────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │    Cosine Similarity Ranking     │  ──> Independent identity confirmation
          └────────────────┬─────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │   Social Media Post Extractor    │  ──> Isolates specific URLs (/reel/, /status/)
          └────────────────┬─────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │    ViT Deepfake Risk Analysis    │  ──> Probabilistic synthetic media signal
          └────────────────┬─────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │   SHA-256 Evidence Fingerprint   │  ──> Deterministic record hashing
          └────────────────┬─────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   [ Layer 1: Local Chain ]    [ Layer 2: Public Chain ]
    `chain/blockchain.json`      Ethereum Sepolia Testnet
   (Linked-block integrity)     (0-value TX + Calldata payload)
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │  Automated Verification Script   │  ──> Etherscan & local hash match: MATCH ✓
          └──────────────────────────────────┘
```

---

## ⛓️ Which Blockchain We Used (And Why)

To satisfy the hackathon prompt with production-grade rigor, `face-chain` implements a **hybrid dual-layer blockchain strategy**:

### 1. Public Blockchain: Ethereum Sepolia Testnet (EVM)
* **Chain Name:** Ethereum Sepolia Testnet
* **Chain ID:** `11155111`
* **Explorer:** [https://sepolia.etherscan.io](https://sepolia.etherscan.io)
* **Transaction Model:** EIP-1559 (`maxFeePerGas` / `maxPriorityFeePerGas`)
* **How Evidence Is Stored:**  
  The SHA-256 evidence fingerprint is embedded directly into the `input` (calldata) field of an on-chain self-transaction (`0-value ETH`). Because the Ethereum Virtual Machine (EVM) immutably logs transaction calldata and timestamps in mined blocks, this creates a **globally verifiable, decentralized Proof of Existence (PoE)** that cannot be altered, forged, or censored by any single party or server administrator.

### 2. Local Blockchain: Cryptographically Linked Hash Ledger
* **Implementation:** `blockchain/blockchain.py` (`chain/blockchain.json`)
* **Structure:** Cryptographic linked list where each block contains `index`, `timestamp`, `evidence_record`, `previous_hash`, and current block `hash` (SHA-256).
* **Why Both?**  
  - The local blockchain ensures high-throughput, offline-capable, zero-gas local auditability.
  - The public Ethereum Sepolia anchor guarantees global consensus and decentralized proof of publication.

---

## ✨ Key Features & Technical Highlights

* **100% Genuine, Non-Hardcoded Search:** No pre-baked results. The pipeline uploads the source image directly to SerpAPI (Google Lens visual search) and streams live web results.
* **Independent Biometric Re-Verification:** Rather than blindly trusting Google Lens search rankings, the pipeline downloads each candidate image, extracts candidate face embeddings using ArcFace, and computes exact cosine similarity against the original image.
* **Specific Social-Media Post Filtering:** Differentiates between dead/generic aggregator links (e.g. `instagram.com/explore`) and high-value, actionable evidence posts (e.g., `instagram.com/reel/<id>`, `twitter.com/<user>/status/<id>`).
* **Synthetic Media & Deepfake Signal:** Evaluates candidate media through a Vision Transformer (ViT) deepfake classification model to alert investigators to manipulated or synthetic faces.
* **Automated Cryptographic Verification:** Comes with `check.py` and `blockchain.verify` CLI tools that pull transaction calldata from Sepolia via Web3 RPC, decode the UTF-8 payload, and verify byte-for-byte equality against the local run.
* **Dual Execution Modes:** Runs as a standalone CLI script for automated forensic runs or as a high-performance **Flask REST API**.

---

## 🚀 Quickstart & How to Run

### 1. Prerequisites

* **Python 3.10+ or 3.11** recommended.
* Virtual environment (`venv` or `conda`).

Clone the repository:
```bash
git clone https://github.com/aaryamanmishra/face-chain.git
cd face-chain
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** The deepfake classifier uses `torch` and `transformers`. The first execution downloads the pre-trained weights (`prithivMLmods/Deep-Fake-Detector-v2-Model`).

---

### 2. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:
```env
# Required for live Google Lens reverse search
SERPAPI_KEY=your_serpapi_key_here

# Optional: For Public Ethereum Sepolia testnet anchoring
PRIVATE_KEY=0x_your_testnet_private_key_here
RPC_URL=https://ethereum-sepolia-rpc.publicnode.com

# Optional logging
LOG_LEVEL=INFO
```

* Free SerpAPI Key: [serpapi.com](https://serpapi.com/)
* Free Sepolia ETH Faucets: [sepoliafaucet.com](https://sepoliafaucet.com/) or [faucets.chain.link/sepolia](https://faucets.chain.link/sepolia)

> *Tip: If Ethereum variables are omitted, the pipeline still fully executes, logging evidence to the local blockchain ledger.*

---

### 3. Running the End-to-End Pipeline

Run the pipeline on any face image:
```bash
python app.py --image path/to/your/image.jpg
```

**Common Flags:**
* `--threshold 0.35`: Minimum cosine similarity score required for face match (default: `0.40`).
* `--top-k 5`: Number of reverse-search candidate images to retrieve and inspect (default: `10`).
* `--log-level DEBUG`: Enable verbose output.

**Sample CLI Output:**
```text
[1/9] Extracting face embedding from source image…
      Face detected (det_score=0.821, norm=20.65)
[2/9] Searching web via SerpAPI Google Lens…
      1 visual matches returned.
[3/9] Downloading candidate images…
      1 candidates downloaded.
[4/9] Extracting face embeddings from candidates…
      1/1 candidates had detectable faces.
[5/9] Ranking candidates by cosine similarity…
      Overall best | similarity=1.0000 | match=True | url=https://www.instagram.com/reel/DXGaS1lDMC1/
[5b] Identifying specific social-media post candidates…
      Best specific social post | platform=Instagram | similarity=1.0000 | url=https://www.instagram.com/reel/DXGaS1lDMC1/
[5c] Annotating trust signals (video detection & content corroboration)…
[5d] Analyzing deepfake risk on primary candidate image…
[6/9] Building evidence record…
[7/9] Hashing evidence record (SHA-256)…
      Evidence hash: cbfd4687a965dfabbb143fa3b481c786…
[8/9] Writing evidence to blockchain…
      Block #2 written | chain_length=3
[9/9] Verifying blockchain integrity…
      Blockchain: VALID | Evidence hash found in block 2. Chain integrity valid.
[10/10] Anchoring evidence hash to Ethereum Sepolia testnet…

  Sepolia TX : 866a0e6987418ccb7cd9fb013694487e4dc3e7cd8dac70c68b91116bbdff42ac
  Etherscan  : https://sepolia.etherscan.io/tx/866a0e6987418ccb7cd9fb013694487e4dc3e7cd8dac70c68b91116bbdff42ac
```

---

### 4. Independent Verification on Blockchain

To verify that the evidence hash stored on Sepolia matches the generated local output:

Run the automated verifier:
```bash
python check.py
```

Or verify an arbitrary transaction directly via CLI:
```bash
python -m blockchain.testnet_anchor verify 0x866a0e6987418ccb7cd9fb013694487e4dc3e7cd8dac70c68b91116bbdff42ac cbfd4687a965dfabbb143fa3b481c7862303abdb5dfc336bbc3966a47a8a4b41
```

**Verification Output:**
```text
TX Hash       : 0x866a0e6987418ccb7cd9fb013694487e4dc3e7cd8dac70c68b91116bbdff42ac
On-chain hash : cbfd4687a965dfabbb143fa3b481c7862303abdb5dfc336bbc3966a47a8a4b41
Expected hash : cbfd4687a965dfabbb143fa3b481c7862303abdb5dfc336bbc3966a47a8a4b41
Etherscan     : https://sepolia.etherscan.io/tx/0x866a0e6987418ccb7cd9fb013694487e4dc3e7cd8dac70c68b91116bbdff42ac
Result        : MATCH ✓
```

---

### 5. Running the REST API Mode

You can run `face-chain` as a microservice:
```bash
python app.py --serve --port 5000
```

#### API Endpoints:
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Healthcheck and readiness probe |
| `POST` | `/analyse` | Upload image file (`file`), optional `threshold` & `top_k` |
| `GET` | `/chain` | Fetch local blockchain ledger and validation status |
| `GET` | `/verify/<hash>` | Verify if an evidence hash exists in a valid block |

**cURL Example:**
```bash
curl -X POST -F "file=@test-image.jpg" http://localhost:5000/analyse
```

---

## 🔎 Live Verification Proof (Real Demonstration)

An actual run produced the following tamper-evident artifact recorded in `results/result.json`:

* **Source Image Hash (SHA-256):** `708d6852d0bc089b85cf8bd721be6822e5469a6c0509f7b762b6e52ba47ab56c`
* **Discovered Social Post:** [Instagram Reel `DXGaS1lDMC1`](https://www.instagram.com/reel/DXGaS1lDMC1/)
* **Face Match Cosine Similarity:** `1.0000` (Direct identity match)
* **Evidence Hash:** `cbfd4687a965dfabbb143fa3b481c7862303abdb5dfc336bbc3966a47a8a4b41`
* **Ethereum Sepolia TX Hash:** [`0x866a0e6987418ccb7cd9fb013694487e4dc3e7cd8dac70c68b91116bbdff42ac`](https://sepolia.etherscan.io/tx/866a0e6987418ccb7cd9fb013694487e4dc3e7cd8dac70c68b91116bbdff42ac)

> Anyone can open the Etherscan link, click **"Click to show more"**, view the **Input Data**, select **"UTF-8"**, and directly read the exact Evidence Hash anchored into Ethereum.

---

## ⚠️ Known Limitations & Edge Cases

In compliance with the challenge requirements, here are the documented limitations:
1. **Search Provider Dependency:** Reverse image lookups depend on external search engine indexing (Google Lens via SerpAPI). Newly published posts or private profiles (e.g. private Instagram/Facebook accounts) cannot be scraped or indexed.
2. **Single-Frame Deepfake Detection:** The deepfake classifier is based on a single-frame Vision Transformer (ViT). While highly effective at spotting full-face synthetic swaps, it does not inspect temporal facial inconsistencies across full video files (e.g. subtle audio-lip desynchronization).
3. **Public RPC Latency & Gas:** Sepolia testnet confirmation depends on public RPC node availability and testnet block production times (~12-15 seconds per block).
4. **Local Chain Scalability:** The local JSON ledger provides immediate, lightweight verification, but for multi-node enterprise environments, a distributed database or decentralized smart contract event indexing would replace local file operations.

---

## 📁 Repository Structure

```
face-chain/
├── app.py                      # Core CLI pipeline & Flask REST API server
├── check.py                    # Independent on-chain verification script
├── requirements.txt            # Project dependencies
├── .env.example                # Configuration template
├── blockchain/
│   ├── blockchain.py           # Local hash-linked blockchain implementation
│   ├── testnet_anchor.py       # Web3 Ethereum Sepolia anchoring & verification
│   └── verify.py               # Proof and chain integrity validation
├── face/
│   ├── detector.py             # InsightFace ArcFace face detection & embeddings
│   └── matcher.py              # Cosine similarity calculation & candidate ranking
├── search/
│   ├── serpapi_search.py       # Live Google Lens reverse search via SerpAPI
│   └── image_downloader.py     # Resilient asynchronous candidate image fetcher
├── utils/
│   ├── deepfake_detector.py    # ViT synthetic media classifier
│   ├── hashing.py              # Cryptographic SHA-256 deterministic serialization
│   ├── social_media.py         # Social platform heuristic & specific post parser
│   └── trust_signals.py        # Video frame extraction & corroboration scoring
├── chain/
│   └── blockchain.json         # Local blockchain ledger
└── results/
    └── result.json             # Serialized execution result & on-chain proof
```

---

## 👥 Hackathon Submission Info

* **Event:** Hacker House Goa 2026
* **Task:** Task #3 — Face Identification & Blockchain Verification
* **Team:** Code Cortex
* **Repository:** [https://github.com/aaryamanmishra/face-chain](https://github.com/aaryamanmishra/face-chain)
* **Demo Video:** [Working Screen Recording Link]
