"""
tests/test_deepfake_detector.py
===============================
Unit tests for utils/deepfake_detector.py.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.deepfake_detector import (
    DEFAULT_SUSPICIOUS_CONFIDENCE_THRESHOLD,
    FIXED_DEEPFAKE_CAVEAT_NOTE,
    analyze_deepfake_risk,
    is_suspicious_deepfake,
)
from utils.hashing import generate_hash, verify_hash


class TestDeepfakeDetector(unittest.TestCase):
    def test_return_keys_always_present_on_missing_file(self):
        result = analyze_deepfake_risk("nonexistent_path_12345.jpg")
        self.assertIn("deepfake_analysis_available", result)
        self.assertIn("predicted_label", result)
        self.assertIn("confidence", result)
        self.assertIn("note", result)
        self.assertFalse(result["deepfake_analysis_available"])
        self.assertIsNone(result["predicted_label"])
        self.assertIsNone(result["confidence"])
        self.assertIn(FIXED_DEEPFAKE_CAVEAT_NOTE, result["note"])

    def test_is_suspicious_deepfake(self):
        fake_high = {
            "deepfake_analysis_available": True,
            "predicted_label": "Deepfake",
            "confidence": 0.85,
            "note": FIXED_DEEPFAKE_CAVEAT_NOTE,
        }
        self.assertTrue(is_suspicious_deepfake(fake_high))

        fake_low_conf = {
            "deepfake_analysis_available": True,
            "predicted_label": "Deepfake",
            "confidence": 0.35,
            "note": FIXED_DEEPFAKE_CAVEAT_NOTE,
        }
        self.assertFalse(is_suspicious_deepfake(fake_low_conf, threshold=0.50))
        self.assertTrue(is_suspicious_deepfake(fake_low_conf, threshold=0.30))

        real_high = {
            "deepfake_analysis_available": True,
            "predicted_label": "Realism",
            "confidence": 0.95,
            "note": FIXED_DEEPFAKE_CAVEAT_NOTE,
        }
        self.assertFalse(is_suspicious_deepfake(real_high))

        unavailable = {
            "deepfake_analysis_available": False,
            "predicted_label": None,
            "confidence": None,
            "note": FIXED_DEEPFAKE_CAVEAT_NOTE,
        }
        self.assertFalse(is_suspicious_deepfake(unavailable))

    def test_mocked_successful_inference(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "Deepfake", "score": 0.89123},
            {"label": "Realism", "score": 0.10877},
        ]

        # Use an existing test file from the repo or a temp file
        img_path = Path(__file__).resolve()

        with patch("utils.deepfake_detector._get_classifier", return_value=mock_pipeline):
            with patch("PIL.Image.open") as mock_img_open:
                mock_img = MagicMock()
                mock_img.convert.return_value = mock_img
                mock_img_open.return_value.__enter__.return_value = mock_img

                res = analyze_deepfake_risk(img_path)

        self.assertTrue(res["deepfake_analysis_available"])
        self.assertEqual(res["predicted_label"], "Deepfake")
        self.assertEqual(res["confidence"], 0.8912)
        self.assertEqual(res["note"], FIXED_DEEPFAKE_CAVEAT_NOTE)

    def test_evidence_hash_changes_with_deepfake_analysis(self):
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
            "deepfake_analysis": {
                "deepfake_analysis_available": False,
                "predicted_label": None,
                "confidence": None,
                "note": FIXED_DEEPFAKE_CAVEAT_NOTE,
            },
        }
        hash_1 = generate_hash(base_record)
        self.assertTrue(verify_hash(base_record, hash_1))

        # Changing deepfake analysis changes the evidence hash
        analyzed_record = dict(base_record)
        analyzed_record["deepfake_analysis"] = {
            "deepfake_analysis_available": True,
            "predicted_label": "Deepfake",
            "confidence": 0.875,
            "note": FIXED_DEEPFAKE_CAVEAT_NOTE,
        }
        hash_2 = generate_hash(analyzed_record)
        self.assertNotEqual(hash_1, hash_2)


if __name__ == "__main__":
    unittest.main()
