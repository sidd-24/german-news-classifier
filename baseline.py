"""
Baselines: always predicting the largest section, and TF-IDF with logistic regression
(regularisation chosen on the validation split).

    python baseline.py
"""
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from common import LABELS, load, per_document_ms, save, scores


def tfidf(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
        ("model", LogisticRegression(C=C, max_iter=2000)),
    ])


def main():
    train, validation, test = load()

    majority = int(train["label"].mode()[0])
    save("majority", {"model": f"always '{LABELS[majority]}'", **scores(test["label"], [majority] * len(test))},
         np.full(len(test), majority))

    search = {}
    for C in [4, 16, 64, 256]:
        model = tfidf(C).fit(train["text"], train["label"])
        search[C] = f1_score(validation["label"], model.predict(validation["text"]), average="macro")
    best = max(search, key=search.get)
    print("validation macro F1 by C:", {c: round(v, 4) for c, v in search.items()}, "-> C =", best)

    model = tfidf(best).fit(train["text"], train["label"])
    predictions = model.predict(test["text"])
    result = {"model": "TF-IDF + logistic regression", "C": best, "validation_macro_f1": search[best],
              **scores(test["label"], predictions),
              "ms_per_document_cpu": per_document_ms(model.predict, list(test["text"]), repeats=3),
              "model_size_mb": len(pickle.dumps(model)) / 1e6}
    save("tfidf", result, predictions)
    print({k: v for k, v in result.items() if k != "per_class_f1"})


if __name__ == "__main__":
    main()
