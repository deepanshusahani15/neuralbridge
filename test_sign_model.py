"""
Test a trained sign language model.

    python test_sign_model.py --image dataset/hello/hello_0001.jpg
    python test_sign_model.py --folder test_data      # test_data/<sign>/*.jpg -> accuracy
    python test_sign_model.py --live                  # live webcam prediction

Live mode keys: Q or ESC = quit
"""
import argparse
import json
import os
from collections import deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from tensorflow import keras

IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_args():
    p = argparse.ArgumentParser(description="Test sign language model")
    p.add_argument("--model", default="sign_model.keras")
    p.add_argument("--labels", default="labels.json")
    p.add_argument("--image", nargs="+", help="One or more image files")
    p.add_argument("--folder", help="Folder of images (flat, or one subfolder per sign)")
    p.add_argument("--live", action="store_true", help="Use the webcam")
    p.add_argument("--camera", type=int, default=1, help="Camera index (1 on your Mac)")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Live mode: below this confidence show '...'")
    return p.parse_args()


def prep(bgr, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size)).astype("float32")   # model rescales internally


def run_images(model, labels, size, paths):
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: could not read image")
            continue
        probs = model(prep(img, size)[None], training=False).numpy()[0]
        best = int(probs.argmax())
        print(f"{path} -> {labels[best]} ({probs[best] * 100:.1f}%)")


def run_folder(model, labels, size, folder):
    items = []   # (path, true_label or None)
    subdirs = [d for d in sorted(os.listdir(folder))
               if os.path.isdir(os.path.join(folder, d))]
    if subdirs:
        for d in subdirs:
            if d not in labels:
                print(f"Skipping folder '{d}' (not a sign the model knows)")
                continue
            for f in sorted(os.listdir(os.path.join(folder, d))):
                if f.lower().endswith(IMG_EXT):
                    items.append((os.path.join(folder, d, f), d))
    else:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXT):
                items.append((os.path.join(folder, f), None))
    if not items:
        raise SystemExit("No images found.")

    results = []
    for start in range(0, len(items), 32):
        chunk, imgs = [], []
        for path, true in items[start:start + 32]:
            img = cv2.imread(path)
            if img is not None:
                imgs.append(prep(img, size))
                chunk.append((path, true))
        if not imgs:
            continue
        probs = model(np.stack(imgs), training=False).numpy()
        for (path, true), p in zip(chunk, probs):
            results.append((path, true, labels[int(p.argmax())], float(p.max())))

    if subdirs:   # we know the true sign -> report accuracy
        correct = sum(t == pred for _, t, pred, _ in results)
        print(f"\nOverall accuracy: {correct / len(results) * 100:.1f}%  "
              f"({correct}/{len(results)})\n")
        for name in labels:
            rows = [r for r in results if r[1] == name]
            if rows:
                ok = sum(r[2] == name for r in rows)
                print(f"{name:>14}: {ok / len(rows) * 100:5.1f}%  ({ok}/{len(rows)})")
        wrong = [r for r in results if r[1] != r[2]]
        if wrong:
            print("\nFirst mistakes:")
            for path, true, pred, conf in wrong[:10]:
                print(f"  {path}: true={true}  predicted={pred} ({conf * 100:.0f}%)")
    else:
        for path, _, pred, conf in results:
            print(f"{path} -> {pred} ({conf * 100:.1f}%)")


def run_live(model, labels, size, camera, threshold):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera}")
    history = deque(maxlen=5)    # average a few frames so the label doesn't flicker

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)   # same mirroring as the capture script
        probs = model(prep(frame, size)[None], training=False).numpy()[0]
        history.append(probs)
        avg = np.mean(history, axis=0)
        best = int(avg.argmax())
        conf = float(avg[best])

        text = f"{labels[best]}  {conf * 100:.0f}%" if conf >= threshold else "..."
        color = (0, 255, 0) if conf >= threshold else (0, 165, 255)
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, "Q/ESC = quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Sign prediction", frame)

        raw = cv2.waitKey(1) & 0xFF
        if raw == 27 or (raw < 128 and chr(raw).lower() == "q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    if not (args.image or args.folder or args.live):
        raise SystemExit("Choose one of: --image FILE [FILE ...] | --folder DIR | --live")

    model = keras.models.load_model(args.model)
    with open(args.labels) as f:
        labels = json.load(f)
    size = model.input_shape[1]

    if args.image:
        run_images(model, labels, size, args.image)
    if args.folder:
        run_folder(model, labels, size, args.folder)
    if args.live:
        run_live(model, labels, size, args.camera, args.threshold)


if __name__ == "__main__":
    main()