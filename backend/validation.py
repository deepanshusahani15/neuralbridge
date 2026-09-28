from dataclasses import dataclass


@dataclass(frozen=True)
class Recognition:
    label: str | None
    confidence: float
    status: str
    phrase: str | None = None


def classify_confidence(label, confidence, phrase=None, high=0.80, medium=0.55):
    if not label or not 0 <= confidence <= 1:
        return Recognition(None, 0.0, "retry")
    if confidence >= high:
        return Recognition(label, confidence, "high", phrase)
    if confidence >= medium:
        return Recognition(label, confidence, "confirm", phrase)
    return Recognition(None, confidence, "retry")
