"""Local speech synthesis with a text-only fallback in the UI."""
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


def synthesize(text):
    if not text or not text.strip():
        raise ValueError("Enter some text to speak")
    with tempfile.TemporaryDirectory(prefix="neuralbridge_tts_") as tmp:
        output = Path(tmp) / "speech.wav"
        if platform.system() == "Darwin" and shutil.which("say") and shutil.which("afconvert"):
            aiff = Path(tmp) / "speech.aiff"
            subprocess.run(["say", "-o", str(aiff), text], check=True, capture_output=True, timeout=30)
            command = ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff), str(output)]
            mime = "audio/wav"
        elif shutil.which("espeak-ng") or shutil.which("espeak"):
            command = [shutil.which("espeak-ng") or shutil.which("espeak"), "-w", str(output), text]
            mime = "audio/wav"
        else:
            raise RuntimeError("Install espeak-ng on Linux or use macOS 'say' for local audio output")
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        return output.read_bytes(), mime
