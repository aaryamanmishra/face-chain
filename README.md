# face-chain

**Reverse-image face-identity verification with tamper-evident blockchain storage.**

---

## Overview

`face-chain` is a production-quality Python pipeline that:

1. Accepts a user image containing a face.
2. Extracts a 512-d ArcFace embedding via **InsightFace**.
3. Runs a **Google Lens** reverse-image search through **SerpAPI**.
4. Downloads candidate images from the top results.
5. Generates embeddings for each candidate face.
6. Ranks candidates by **cosine similarity**.
7. Builds a tamper-evident **evidence record**.
8. Hashes the record with **SHA-256**.
9. Stores the evidence hash in a **local blockchain**.
10. Verifies blockchain integrity.
11. Prints and saves all results.

---

## Project Structure

```
face-chain/
│
├── app.py                  ← Main pipeline + Flask API
├── requirements.txt
├── .env.example
│
├── face/
│   ├── detector.py         ← InsightFace embedding extraction
│   └── matcher.py          ← Cosine similarity matching
│
├── search/
│   ├── serpapi_search.py   ← SerpAPI Google Lens integration
│   └── image_downloader.py ← Thumbnail / image downloader
│
├── blockchain/
│   ├── blockchain.py       ← Local SHA-256 linked blockchain
│   └── verify.py           ← Chain + evidence hash verification
│
├── utils/
│   └── hashing.py          ← SHA-256 hashing utilities
│
├── uploads/                ← Input images
├── downloads/              ← Downloaded candidate images
├── results/                ← Pipeline output (result.json)
└── chain/                  ← Persisted blockchain (blockchain.json)
```

---

## Quick Start

### 1. Install dependencies

```bash
cd face-chain
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set SERPAPI_KEY=your_key_here
```

> Get a free SerpAPI key at [serpapi.com](https://serpapi.com/).

### 3. Run the pipeline

```bash
python app.py --image uploads/your_face.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `0.40` | Minimum cosine similarity to declare a match |
| `--top-k` | `10` | Number of SerpAPI results to process |
| `--log-level` | `INFO` | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Flask API Mode

Start the HTTP server:

```bash
python app.py --serve --port 5000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyse` | POST | Run pipeline on uploaded image |
| `/chain` | GET | Return the full blockchain |
| `/verify/<hash>` | GET | Verify an evidence hash |

**Example with curl:**
```bash
curl -X POST http://localhost:5000/analyse \
     -F "file=@uploads/face.jpg" \
     -F "threshold=0.40"
```

---

## Output

### Console

```
--------------------------------------------------
  FACE MATCH FOUND
--------------------------------------------------

Matched URL:
  https://example.com/profile/john-doe

Similarity:
  0.8712

Evidence Hash:
  3a9f7c2e1b…

Blockchain Status:
  VALID

Block Index:
  #1
```

### results/result.json

```json
{
  "pipeline_version": "1.0.0",
  "elapsed_seconds": 14.3,
  "best_match": {
    "url": "https://…",
    "similarity": 0.8712,
    "match": true
  },
  "evidence_hash": "3a9f7c2e…",
  "blockchain": {
    "block_index": 1,
    "status": "VALID"
  }
}
```

---

## Blockchain Verification (standalone)

```bash
# Verify chain integrity only
python -m blockchain.verify --chain chain/blockchain.json

# Verify a specific evidence record + hash
python -m blockchain.verify \
    --chain chain/blockchain.json \
    --hash <sha256_hex> \
    --record results/result.json
```

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `SERPAPI_KEY` | ✅ | SerpAPI authentication key |
| `LOG_LEVEL` | ❌ | Logging verbosity (default: `INFO`) |

---

## Notes

- InsightFace downloads the `buffalo_l` model pack (~600 MB) on first run into `~/.insightface/`.
- The default cosine similarity threshold of **0.40** is tuned for ArcFace embeddings; higher values (e.g., 0.60) increase precision at the cost of recall.
- All blockchain data is stored locally in `chain/blockchain.json`; no external nodes or network calls are needed.
