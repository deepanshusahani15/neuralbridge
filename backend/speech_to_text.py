"""Offline speech transcription; model weights download on first use."""
from functools import lru_cache
from pathlib import Path
import tempfile


@lru_cache(maxsize=2)
def _model(name):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(audio_bytes, model_name="tiny", language=None):
    if not audio_bytes:
        raise ValueError("Record a short audio message first")
    # The browser sends a WAV stream; use a temporary file because the decoder accepts paths.
    with tempfile.TemporaryDirectory(prefix="neuralbridge_audio_") as tmp:
        path = Path(tmp) / "recording.wav"
        path.write_bytes(audio_bytes)
        segments, _ = _model(model_name).transcribe(
            str(path), language=language, beam_size=3,
            vad_filter=True, condition_on_previous_text=False,
        )
        return " ".join(part.text.strip() for part in segments).strip()
