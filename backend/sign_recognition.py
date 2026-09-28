"""Image-mode MediaPipe hand features and a separately trained gesture classifier."""
from pathlib import Path

import numpy as np

from backend.validation import classify_confidence

ROOT = Path(__file__).resolve().parents[1]
LANDMARK_MODEL = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER = ROOT / "models" / "sign_classifier.joblib"
FEATURE_NAMES = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def normalize_landmarks(points, handedness="Right"):
    """Wrist origin, hand-size scale, and a common orientation for left/right hands."""
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError("Expected 21 finite (x, y, z) hand landmarks")
    p = p - p[0]
    scale = np.max(np.linalg.norm(p[:, :2], axis=1))
    if scale < 1e-6:
        raise ValueError("Hand landmarks have no usable size")
    p = p / scale
    if handedness.lower() == "left":
        p[:, 0] *= -1
    return p.flatten()


class HandExtractor:
    def __init__(self, model_path=LANDMARK_MODEL):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Hand model missing: {model_path}. Run python scripts/download_hand_model.py")
        import mediapipe as mp

        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract(self, rgb):
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected RGB image with three channels")
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not result.hand_landmarks:
            return None, "no_hand", None
        if len(result.hand_landmarks) != 1:
            return None, "multiple_hands", None
        hand = result.hand_landmarks[0]
        side = result.handedness[0][0].category_name
        features = normalize_landmarks([(p.x, p.y, p.z) for p in hand], side)
        return features, side, [(p.x, p.y) for p in hand]

    def close(self):
        self.detector.close()


def load_classifier(path=CLASSIFIER):
    import joblib

    if not Path(path).is_file():
        raise FileNotFoundError("Gesture classifier missing. Collect labeled samples and run python scripts/train.py")
    bundle = joblib.load(path)  # Load only a model trained by your own team; joblib uses pickle.
    if bundle.get("schema_version") != 1 or bundle.get("features") != FEATURE_NAMES:
        raise ValueError("Gesture classifier feature schema does not match this app")
    return bundle["model"]


def predict(features, model, phrases):
    probabilities = model.predict_proba(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
    index = int(np.argmax(probabilities))
    label = str(model.classes_[index])
    phrase = phrases.get(label)
    if phrase is None:
        raise ValueError(f"No phrase configured for model label {label}")
    return classify_confidence(label, float(probabilities[index]), phrase)
