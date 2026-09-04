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