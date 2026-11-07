import numpy as np
import pytest

import baseline
import common


@pytest.fixture(scope="module")
def splits():
    return common.load()


def test_validation_split_is_stratified_and_separate(splits):
    train, validation, test = splits
    assert len(train) + len(validation) == 9245 and len(test) == 1028
    shares = lambda df: df["label"].value_counts(normalize=True).sort_index()
    assert np.allclose(shares(train), shares(validation), atol=0.01)
    assert not set(validation["text"]) & set(train["text"])


def test_scores():
    result = common.scores([0, 0, 1, 2], [0, 1, 1, 2])
    assert result["accuracy"] == 0.75
    assert result["per_class_f1"]["Web"] == pytest.approx(2 / 3)
    assert set(result["per_class_f1"]) == set(common.LABELS)


def test_per_document_ms_counts_documents_and_repeats():
    calls = []
    ms = common.per_document_ms(lambda texts: calls.append(len(texts)), ["a"] * 10, repeats=3)
    assert calls == [10, 10, 10] and ms >= 0


def test_tfidf_baseline_learns_the_sections(splits):
    train, validation, _ = splits
    sample = train.groupby("label").head(150)
    model = baseline.tfidf(C=4).fit(sample["text"], sample["label"])
    accuracy = (model.predict(validation["text"]) == validation["label"]).mean()
    assert accuracy > 0.6                     # 9 classes, so chance is about 0.11
