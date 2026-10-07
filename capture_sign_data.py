import cv2
import os
import time

CAMERA = 1        # camera 1 is the working webcam on this Mac
TARGET = 100      # minimum images per sign

cap = cv2.VideoCapture(CAMERA, cv2.CAP_AVFOUNDATION)   # AVFoundation = macOS camera backend
if not cap.isOpened():
    raise SystemExit("Could not open camera. Check System Settings > Privacy & Security > Camera")

# macOS can take a moment before the first frame arrives, so wait for it
frame = None
for _ in range(50):
    ok, frame = cap.read()
    if ok:
        break
    time.sleep(0.1)
if frame is None:
    cap.release()
    raise SystemExit("Camera opened but gives no frames. Allow camera access for your "
                     "terminal app, fully quit it (Cmd+Q), reopen and try again.")

quit_all = False

while not quit_all:
    label = input("\nEnter sign name (leave blank to finish): ").strip()
    if not label or label.lower() in ("quit", "exit"):
        break

    folder = os.path.join("dataset", label)
    count = len(os.listdir(folder)) if os.path.isdir(folder) else 0   # folder is NOT created yet

    print(f"Capturing '{label}'.  C = capture   E = finish this sign   Q = close camera")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        show = frame.copy()
        color = (0, 255, 0) if count >= TARGET else (0, 255, 255)
        cv2.putText(show, f"{label}: {count}/{TARGET}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(show, "C=capture  E=next sign  Q/ESC=quit", (10, show.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Capture", show)

        raw = cv2.waitKey(1) & 0xFF
        key = chr(raw).lower() if raw < 128 else ""   # works with Caps Lock on too
        if key == "c":
            os.makedirs(folder, exist_ok=True)   # folder is created on the first captured image
            cv2.imwrite(os.path.join(folder, f"{label}_{count:04d}.jpg"), frame)
            count += 1
            if count == TARGET:        # reached target: automatically ask for the next sign
                print(f"\nDone! {TARGET} images saved for '{label}'")
                break
        elif key == "e":               # end this sign, go back to the name prompt
            print(f"Saved {count} images for '{label}'")
            break
        elif key == "q" or raw == 27:  # Q or ESC: close camera and end the program
            print(f"Saved {count} images for '{label}'")
            quit_all = True
            break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")