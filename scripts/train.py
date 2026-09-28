"""Train and evaluate a gesture baseline without silently mixing participants."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.phrase_mapper import load_phrases
from backend.sign_recognition import FEATURE_NAMES


def prepare_split(frame):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    labels = frame["label"]
    people = frame["participant"].nunique()
    if people >= 3:
        groups = frame["participant"]
        # Reject a group split that leaves a class missing from either side.
        for seed in range(42, 142):
            train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(frame, labels, groups))
            if set(labels.iloc[train_idx]) == set(labels) == set(labels.iloc[test_idx]):
                return train_idx, test_idx, "held-out participants"
        raise ValueError("Cannot leave out a participant while retaining all labels. Collect every class from more people.")
    if frame["session"].nunique() >= 4:
        groups = frame["participant"].astype(str) + "/" + frame["session"].astype(str)
        for seed in range(42, 142):
            train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(frame, labels, groups))
            if set(labels.iloc[train_idx]) == set(labels) == set(labels.iloc[test_idx]):
                return train_idx, test_idx, "held-out sessions (same participants; weaker evidence)"
    raise ValueError("Need at least 3 participants covering every label, or 4 sessions that allow a class-complete group split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "landmarks.csv")
    args = parser.parse_args()
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    frame = pd.read_csv(args.data)
    required = {"sample_id", "label", "participant", "session", *FEATURE_NAMES}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    frame = frame.drop_duplicates("sample_id")
    frame[FEATURE_NAMES] = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["label", "participant", "session", *FEATURE_NAMES])
    frame = frame[np.isfinite(frame[FEATURE_NAMES].to_numpy()).all(axis=1)]
    labels = sorted(frame["label"].unique().tolist())
    if len(labels) < 2 or min(frame["label"].value_counts()) < 10:
        raise ValueError("Need at least two classes and 10 valid samples per class")
    unknown = set(labels) - set(load_phrases())
    if unknown:
        raise ValueError(f"Labels absent from phrase_map.json: {sorted(unknown)}")
    train_idx, test_idx, split = prepare_split(frame)
    x_train, y_train = frame.iloc[train_idx][FEATURE_NAMES], frame.iloc[train_idx]["label"]
    x_test, y_test = frame.iloc[test_idx][FEATURE_NAMES], frame.iloc[test_idx]["label"]
    model = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    results = {
        "split": split, "train_samples": len(train_idx), "test_samples": len(test_idx),
        "classes": labels, "accuracy": report["accuracy"], "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "note": "Random Forest probabilities are heuristic; confidence thresholds are not calibrated OOD guarantees.",
    }
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    joblib.dump({"schema_version": 1, "features": FEATURE_NAMES, "model": model}, models / "sign_classifier.joblib")
    (models / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
