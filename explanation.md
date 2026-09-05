# Face-Chain: Pipeline Explanation

This document explains the end-to-end data flow of the `face-chain` project. It breaks down what happens from the moment an image is uploaded to the moment the final evidence is anchored to the blockchain, detailing the significance of each step and the exact data passed between layers.

---

## 1. Face Detection & Embedding

**Item Passed:** The local file path of the uploaded source image.
**Technology:** InsightFace (ArcFace model)

**What happens:** 
The system loads the image and runs it through a facial recognition model. It detects the face bounding box and generates a mathematical vector (typically 512 dimensions) called an **embedding**. 

**Significance:** 
This embedding is the "fingerprint" of the face. By extracting this up front, the system isn't forced to trust the black-box results of a search engine. We retain this mathematical fingerprint to independently verify all future search results.

---

## 2. Reverse Image Search

**Item Passed:** The local source image (uploaded via API).
**Technology:** SerpAPI (Google Lens API)

**What happens:** 
The image is uploaded to Google Lens via SerpAPI. Google scans the internet for visually similar images, returning a list of search results containing image URLs, page titles, and source webpage links.

**Significance:** 
This is the "discovery" phase. It casts a wide net across the internet to find where this face, or identical photos, might exist, especially focusing on public social media profiles or news articles.

---

## 3. Candidate Download & Independent Verification

**Item Passed:** The URLs of the candidate images returned by SerpAPI.
**Technology:** HTTP Requests & InsightFace

**What happens:** 
The pipeline downloads the candidate images locally. It then passes each downloaded image back through the InsightFace model to extract a *new* face embedding for every candidate. Finally, it calculates the **Cosine Similarity** (a mathematical measure of distance) between the original face embedding (from Step 1) and the candidate embeddings.

**Significance:** 
Search engines often return false positives (e.g., people wearing similar clothes but with different faces). This step acts as an independent cryptographic filter. By comparing embeddings, the system rigorously proves that the face found on the internet is actually the same person as the uploaded photo.

---

## 4. Social Media Identification & Ranking

**Item Passed:** The ranked list of verified candidate matches and their source webpage URLs.
**Technology:** Python String/Regex matching

**What happens:** 
The system filters the highly-ranked matches to prioritize specific social media platforms (e.g., Instagram, Twitter, LinkedIn). It specifically looks for URLs that point to a *specific post* or *profile* rather than a generic explore page.

**Significance:** 
The goal of the investigation is usually to find a person's identity. A match on a specific social media post is highly valuable evidence, whereas a match on a generic search aggregator is not. This step ensures the best possible actionable intelligence is selected.

---

## 5. Evidence Hashing

**Item Passed:** The metadata of the best match (Source Image Hash, Matched URL, Similarity Score, Platform, Timestamp).
**Technology:** SHA-256 Hashing

**What happens:** 
All the crucial details of the match are compiled into a JSON object (the "Evidence Record"). This object is run through a SHA-256 algorithm to generate a unique, fixed-size string of characters (the Evidence Hash).

**Significance:** 
Hashing is a one-way function. This hash perfectly represents the exact state of the evidence. If even one character in the URL or similarity score is changed later, the hash will change completely, making the record tamper-evident.

---

## 6. Local Blockchain Storage

**Item Passed:** The Evidence Hash and the Evidence Record metadata.
**Technology:** Local Linked-List JSON (`chain/blockchain.json`)

**What happens:** 
The system creates a new "block" containing the evidence. Crucially, this block also contains the hash of the *previous* block in the chain. The new block is appended to the local JSON ledger.

**Significance:** 
This creates a cryptographically linked history of investigations. If a malicious actor tries to alter a past investigation in the JSON file, the hash of that block changes, which breaks the link to the next block, immediately flagging the database as tampered.

---

## 7. Ethereum Sepolia Anchoring

**Item Passed:** The SHA-256 Evidence Hash (from Step 5).
**Technology:** Web3.py & Ethereum Sepolia Testnet

**What happens:** 
The system connects to the Sepolia blockchain using your provided RPC URL and Private Key. It creates a 0-value "self-transaction" (sending 0 ETH from your wallet to your wallet). It embeds the Evidence Hash into the `data` (input) field of this transaction and broadcasts it to the network.

**Significance:** 
While the local blockchain is tamper-evident, it can still be entirely deleted or lost. Anchoring to a public Ethereum network provides **immutable, public proof of existence**. Once the transaction is mined on Sepolia, you can prove to anyone that you possessed this specific evidence hash at that exact timestamp, and no one (not even the server admin) can alter or delete that record.
