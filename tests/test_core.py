import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.phrase_mapper import get_phrase, load_phrases
from backend.sign_recognition import FEATURE_NAMES, normalize_landmarks, predict
from backend.validation import classify_confidence
from backend.voice_response import select_response


class CoreTests(unittest.TestCase):
    def test_normalization_is_translation_and_scale_invariant(self):
        points = [[float(i), float(i % 5), 0.1 * i] for i in range(21)]
        shifted = [[4 * x + 30, 4 * y - 10, 4 * z + 7] for x, y, z in points]
        import numpy as np

        np.testing.assert_allclose(normalize_landmarks(points), normalize_landmarks(shifted), atol=1e-5)
        self.assertEqual(len(normalize_landmarks(points)), len(FEATURE_NAMES))

    def test_invalid_landmarks_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_landmarks([[0, 0, 0]] * 20)
        with self.assertRaises(ValueError):
            normalize_landmarks([[0, 0, 0]] * 21)

    def test_confidence_gates_audio_output(self):
        self.assertEqual(classify_confidence("HELP", 0.9, "I need help").status, "high")
        self.assertEqual(classify_confidence("HELP", 0.7, "I need help").status, "confirm")
        low = classify_confidence("HELP", 0.2, "I need help")
        self.assertEqual(low.status, "retry")
        self.assertIsNone(low.phrase)

    def test_phrase_mapping_and_scripted_response(self):
        phrases = load_phrases()
        self.assertEqual(get_phrase("WATER", phrases), "I need water")
        self.assertIn("water", select_response("water please").lower())
        self.assertIn("I heard:", select_response("Good morning"))

    def test_model_prediction_does_not_invent_missing_phrase(self):
        import numpy as np

        class FakeModel:
            classes_ = np.array(["HELP", "UNKNOWN"])

            def predict_proba(self, x):
                return np.array([[0.95, 0.05]])

        result = predict(np.zeros(63), FakeModel(), {"HELP": "I need help"})
        self.assertEqual(result.phrase, "I need help")


if __name__ == "__main__":
    unittest.main()
