"""Data, labels, splits and metrics shared by the baseline and the fine-tuned model."""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

URL = "https://huggingface.co/api/datasets/community-datasets/gnad10/parquet/default/{split}/0.parquet"
LABELS = ["Web", "Panorama", "International", "Wirtschaft", "Sport", "Inland", "Etat", "Wissenschaft", "Kultur"]
SEED = 42
RESULTS = Path("results")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test. The official test set is only used for the final numbers."""
    frames = {}
    for split in ["train", "test"]:
        path = Path(f"data/{split}.parquet")
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(URL.format(split=split), path)
        frames[split] = pd.read_parquet(path)
    # a placeholder text ("Der Volltext dieses ... Artikels steht aus rechtlichen Gründen nicht mehr zur
    # Verfügung") appears several times with different labels, in train and test: not an article, so dropped
    counts = pd.concat(frames.values())["text"].value_counts()
    repeated = set(counts[counts > 1].index)
    frames = {split: df[~df["text"].isin(repeated)].reset_index(drop=True) for split, df in frames.items()}
    train, validation = train_test_split(frames["train"], test_size=0.1, stratify=frames["train"]["label"], random_state=SEED)
    return train.reset_index(drop=True), validation.reset_index(drop=True), frames["test"]


def scores(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, labels=range(len(LABELS)), target_names=LABELS,
                                   output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "per_class_f1": {name: report[name]["f1-score"] for name in LABELS}}


def per_document_ms(predict, texts: list[str], repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        predict(texts)
    return (time.perf_counter() - start) * 1000 / (len(texts) * repeats)


def save(name: str, result: dict, predictions: np.ndarray) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(RESULTS / f"{name}_test_predictions.npy", predictions)
