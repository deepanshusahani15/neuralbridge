"""Per-stream video processing, independent of Streamlit reruns."""
from collections import deque
import threading

from backend.sign_recognition import HandExtractor, predict


class LiveEngine:
    """The video callback owns the landmarker; the UI reads a small thread-safe result."""

    def __init__(self, model, phrases):
        self.model = model
        self.phrases = phrases
        self.lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.extractor = None
        self.votes = deque(maxlen=5)
        self.latest = None
        self.message = "Waiting for one hand in the camera frame."

    def recv(self, frame):
        with self.processing_lock:
            return self.process(frame)

    def process(self, frame):
        import av
        import cv2

        if self.model is None:
            return frame
        bgr = frame.to_ndarray(format="bgr24")
        try:
            if self.extractor is None:
                self.extractor = HandExtractor()
            features, state, points = self.extractor.extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if features is None:
                self.votes.clear()
                with self.lock:
                    self.latest = None
                    self.message = "Move one hand into view." if state == "no_hand" else "Show only one hand."
            else:
                result = predict(features, self.model, self.phrases)
                self.votes.append(result.label if result.status == "high" else None)
                stable = len(self.votes) >= 4 and list(self.votes)[-4:] == [result.label] * 4
                with self.lock:
                    self.latest = result if stable and result.status == "high" else None
                    self.message = "Gesture held steady." if self.latest else "Hold a supported gesture steady for a moment."
                height, width = bgr.shape[:2]
                for x, y in points:
                    cv2.circle(bgr, (int(x * width), int(y * height)), 3, (62, 220, 112), -1)
        except Exception as exc:
            self.votes.clear()
            with self.lock:
                self.latest = None
                self.message = f"Camera processing unavailable: {exc}"
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    def snapshot(self):
        with self.lock:
            return self.latest, self.message


    def on_ended(self):
        with self.processing_lock:
            if self.extractor is not None:
                self.extractor.close()
                self.extractor = None
            self.votes.clear()
            with self.lock:
                self.latest = None
                self.message = "Camera stopped."
