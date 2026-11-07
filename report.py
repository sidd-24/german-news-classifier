"""
Compares the models and analyses the fine-tuned model's errors: charts in docs/, numbers in
results/comparison.json.

    python report.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

from common import LABELS, RESULTS, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = {"majority": "Majority class", "tfidf": "TF-IDF + logistic regression", "gbert": "German BERT, fine-tuned"}


def plot_per_class(results, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(LABELS))
    for offset, (name, color) in zip([-0.2, 0.2], [("tfidf", "#9ca3af"), ("gbert", "#2563eb")]):
        values = [results[name]["per_class_f1"][label] for label in LABELS]
        ax.bar(x + offset, values, 0.4, color=color, label=MODELS[name])
    ax.set_xticks(x, LABELS, rotation=20)
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("F1 on the test set")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)), normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if matrix[i, j] >= 0.01:
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted section")
    ax.set_ylabel("Actual section")
    ax.set_title("German BERT on the test set (share of each actual section)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusions(y_true, y_pred, top=5) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    pairs = [(matrix[i, j], i, j) for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j and matrix[i, j]]
    return [{"actual": LABELS[i], "predicted": LABELS[j], "articles": int(n)} for n, i, j in sorted(pairs, reverse=True)[:top]]


def confidence_table(y_true, probabilities) -> list[dict]:
    """If only confident predictions are automated and the rest go to an editor: how many, how accurate?"""
    confidence, predicted = probabilities.max(1), probabilities.argmax(1)
    rows = []
    for threshold in [0.0, 0.5, 0.7, 0.9, 0.95]:
        kept = confidence >= threshold
        rows.append({"threshold": threshold, "automated_share": float(kept.mean()),
                     "accuracy_automated": float((predicted[kept] == y_true[kept]).mean())})
    return rows


def main():
    _, _, test = load()
    y_true = test["label"].to_numpy()
    results = {name: json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8")) for name in MODELS}
    y_pred = np.load(RESULTS / "gbert_test_predictions.npy")
    probabilities = np.load(RESULTS / "gbert_test_probabilities.npy")
    Path("docs").mkdir(exist_ok=True)
    plot_per_class(results, "docs/per_class_f1.png")
    plot_confusion(y_true, y_pred, "docs/confusion.png")

    top = confusions(y_true, y_pred)
    first = top[0]
    mask = (y_true == LABELS.index(first["actual"])) & (y_pred == LABELS.index(first["predicted"]))
    examples = [text[:160] for text in test["text"][mask].head(3)]
    comparison = {
        "models": {MODELS[n]: {k: r.get(k) for k in ["accuracy", "macro_f1", "ms_per_document_cpu", "ms_per_document_gpu",
                                                      "model_size_mb", "train_minutes"]} for n, r in results.items()},
        "top_confusions": top, "examples_of_top_confusion": examples,
        "confidence": confidence_table(y_true, probabilities),
        "tfidf_and_gbert_both_wrong": int(((np.load(RESULTS / "tfidf_test_predictions.npy") != y_true) & (y_pred != y_true)).sum()),
        "gbert_wrong": int((y_pred != y_true).sum()),
    }
    (RESULTS / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
