"""Download the official MediaPipe hand landmarker asset into models/."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
TARGET = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file() and TARGET.stat().st_size > 100_000:
        print(f"Already present: {TARGET}")
        return
    with urlopen(URL, timeout=60) as response:
        data = response.read()
    if len(data) < 100_000:
        raise RuntimeError("Downloaded hand model appears incomplete")
    temporary = TARGET.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(TARGET)
    print(f"Saved {len(data):,} bytes to {TARGET}")


if __name__ == "__main__":
    main()
