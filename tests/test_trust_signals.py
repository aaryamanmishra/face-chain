"""
tests/test_trust_signals.py
===========================
Tests for trust signals: video content detection, content corroboration,
candidate annotation, and evidence hash propagation.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.trust_signals import (
    FRAME_EXTRACTION_NOTE,
    detect_video_content,
    evaluate_content_corroboration,
    annotate_trust_signals,
)
from utils.hashing import generate_hash, verify_hash


class TestTrustSignals(unittest.TestCase):
    def test_detect_video_content_instagram_reel(self):
        url = "https://www.instagram.com/reel/DXGaS1lDMC1/"
        is_video, note = detect_video_content(url, platform="Instagram")
        self.assertTrue(is_video)
        self.assertEqual(note, FRAME_EXTRACTION_NOTE)

    def test_detect_video_content_instagram_post(self):
        url = "https://www.instagram.com/p/C-abc123/"
        is_video, note = detect_video_content(url, platform="Instagram")
        self.assertFalse(is_video)
        self.assertIsNone(note)

    def test_detect_video_content_youtube_shorts(self):
        url = "https://www.youtube.com/shorts/abcd1234ef"
        is_video, note = detect_video_content(url, platform="YouTube")
        self.assertTrue(is_video)
        self.assertEqual(note, FRAME_EXTRACTION_NOTE)

    def test_detect_video_content_tiktok(self):
        url = "https://www.tiktok.com/@creator/video/7123456789"
        is_video, note = detect_video_content(url, platform="TikTok")
        self.assertTrue(is_video)
        self.assertEqual(note, FRAME_EXTRACTION_NOTE)

    def test_detect_video_content_static_web(self):
        url = "https://example.com/articles/profile-photo.jpg"
        is_video, note = detect_video_content(url, platform=None)
        self.assertFalse(is_video)
        self.assertIsNone(note)

    def test_evaluate_content_corroboration_no_context_normal(self):
        title = "Road to Math Olympiad Q24"
        corrob = evaluate_content_corroboration(title, identity_context=None)
        self.assertEqual(corrob, "UNKNOWN")

    def test_evaluate_content_corroboration_no_context_clickbait(self):
        cases = [
            "She put her house up for sale after this",
            "You won't believe what happened next!",
            "Day in the life of a college student",
            "Wait till the end shocking twist",
            "Top 10 shocking celebrity moments gone wrong",
        ]
        for title in cases:
            with self.subTest(title=title):
                corrob = evaluate_content_corroboration(title, identity_context=None)
                self.assertEqual(corrob, "LOW")

    def test_evaluate_content_corroboration_with_context(self):
        title_aligned = "Jane Doe Announced as Keynote Speaker"
        corrob_aligned = evaluate_content_corroboration(title_aligned, identity_context="Jane Doe")
        self.assertEqual(corrob_aligned, "HIGH")

        title_unaligned = "Day in the life of a generic vlogger"
        corrob_unaligned = evaluate_content_corroboration(title_unaligned, identity_context="Jane Doe")
        self.assertEqual(corrob_unaligned, "LOW")

    def test_annotate_trust_signals(self):
        cand = {
            "link": "https://www.instagram.com/reel/DXGaS1lDMC1/",
            "platform": "Instagram",
            "title": "You won't believe this move",
        }
        annotated = annotate_trust_signals(cand)
        self.assertEqual(annotated["content_corroboration"], "LOW")
        self.assertTrue(annotated["is_video_content"])
        self.assertEqual(annotated["frame_extraction_note"], FRAME_EXTRACTION_NOTE)

    def test_evidence_hash_covers_trust_signals(self):
        base_record = {
            "source_image": "uploads/source.jpg",
            "matched_url": "https://www.instagram.com/reel/DXGaS1lDMC1/",
            "similarity": 0.78,
            "timestamp": "2026-09-05T12:00:00.000000+00:00",
            "candidate_image": "downloads/test.jpg",
            "source_embedding_norm": 22.0,
            "platform": "Instagram",
            "content_corroboration": "LOW",
            "is_video_content": True,
            "frame_extraction_note": FRAME_EXTRACTION_NOTE,
        }
        hash_1 = generate_hash(base_record)
        self.assertTrue(verify_hash(base_record, hash_1))

        tampered = dict(base_record)
        tampered["content_corroboration"] = "HIGH"
        hash_2 = generate_hash(tampered)
        self.assertNotEqual(hash_1, hash_2)


if __name__ == "__main__":
    unittest.main()
