"""Capture only landmarks, with explicit participant and session identifiers."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.phrase_mapper import load_phrases
from backend.sign_recognition import FEATURE_NAMES, HandExtractor


def main():
    parser = argparse.ArgumentParser(description="Collect one gesture label from the local camera")
    parser.add_argument("--label", required=True)
    parser.add_argument("--participant", required=True, help="Stable anonymized participant ID")
    parser.add_argument("--session", default=None, help="Unique recording session ID; default current UTC time")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.35)
    args = parser.parse_args()
    label = args.label.upper().strip()
    if label not in load_phrases():
        parser.error(f"Unsupported label {label}; edit data/phrase_map.json first")
    if not args.participant.strip() or args.count < 1 or args.interval <= 0:
        parser.error("Provide a participant ID, a positive count, and a positive interval")
    session = args.session or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import cv2

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open local camera. Check permissions and --camera index")
    output = ROOT / "data" / "processed" / "landmarks.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "label", "participant", "session", "handedness", *FEATURE_NAMES]
    extractor = None
    saved = 0
    last = 0.0
    try:
        extractor = HandExtractor()
        with output.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            if file.tell() == 0:
                writer.writeheader()
            while saved < args.count:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                # Flip only for a natural camera preview. The same image orientation is used for features.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features, side, _ = extractor.extract(rgb)
                now = time.monotonic()
                if features is not None and now - last >= args.interval:
                    row = dict(zip(FEATURE_NAMES, map(float, features)))
                    row.update(sample_id=uuid4().hex, label=label, participant=args.participant,
                               session=session, handedness=side)
                    writer.writerow(row)
                    file.flush()
                    saved += 1
                    last = now
                status = f"{label}: {saved}/{args.count} | {'one hand' if features is not None else side} | q to quit"
                cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 220, 65), 2)
                cv2.imshow("Neuralbridge data collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if extractor:
            extractor.close()
    print(f"Saved {saved} samples to {output}. Do not label a gesture without verifying its sign reference.")


if __name__ == "__main__":
    main()
