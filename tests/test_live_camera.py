import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from backend.live_camera import LiveEngine
from backend.validation import classify_confidence


class LiveCameraTests(unittest.TestCase):
    def test_preview_needs_no_model(self):
        engine = LiveEngine(None, {})
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24")
        with patch("backend.live_camera.HandExtractor") as extractor:
            self.assertIs(engine.recv(frame), frame)
            extractor.assert_not_called()

    def test_stability_loss_error_and_cleanup(self):
        engine = LiveEngine(Mock(), {"HELP": "Help"})
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24")
        extractor = Mock()
        extractor.extract.return_value = (np.zeros(63), "Right", [(0.5, 0.5)])
        result = classify_confidence("HELP", 0.95, "Help")
        with patch("backend.live_camera.HandExtractor", return_value=extractor), patch("backend.live_camera.predict", return_value=result):
            for _ in range(3):
                engine.recv(frame)
                self.assertIsNone(engine.snapshot()[0])
            engine.recv(frame)
            self.assertEqual(engine.snapshot()[0], result)
            extractor.extract.side_effect = RuntimeError("test failure")
            engine.recv(frame)
            self.assertIsNone(engine.snapshot()[0])
            extractor.extract.side_effect = None
            engine.recv(frame)
            self.assertIsNone(engine.snapshot()[0])
            extractor.extract.return_value = (None, "no_hand", None)
            engine.recv(frame)
            self.assertEqual(engine.snapshot(), (None, "Move one hand into view."))
            engine.on_ended()
            engine.on_ended()
            extractor.close.assert_called_once()
            self.assertIsNone(engine.extractor)
