import json
from pathlib import Path

DEFAULT_MAP = Path(__file__).resolve().parents[1] / "data" / "phrase_map.json"


def load_phrases(path=DEFAULT_MAP):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Phrase map must be a nonempty JSON object")
    if any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in data.items()):
        raise ValueError("Phrase map keys and phrases must be nonempty strings")
    return data


def get_phrase(label, phrases):
    return phrases.get(label)
