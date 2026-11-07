"""
Fine-tunes German BERT (deepset/gbert-base) on the news sections. The epoch with the best
validation macro F1 is kept; the test set is used once, at the end.

    python finetune.py            # about 15 minutes on an 8 GB laptop GPU
"""
import time

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from common import LABELS, SEED, load, per_document_ms, save, scores

MODEL = "deepset/gbert-base"
MAX_LENGTH = 512        # articles average ~2,600 characters; BERT reads at most 512 tokens, the start carries the topic


class Articles(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.encodings.items()}, "labels": self.labels[i]}


def metrics(prediction):
    result = scores(prediction.label_ids, prediction.predictions.argmax(-1))
    return {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}


def main():
    set_seed(SEED)
    train, validation, test = load()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encode = lambda df: Articles(tokenizer(list(df["text"]), truncation=True, max_length=MAX_LENGTH), df["label"])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

    args = TrainingArguments(
        output_dir="models/checkpoints", num_train_epochs=3, learning_rate=3e-5, warmup_ratio=0.1, weight_decay=0.01,
        per_device_train_batch_size=8, gradient_accumulation_steps=2, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        save_total_limit=1, fp16=torch.cuda.is_available(), logging_steps=100, report_to=[], seed=SEED)
    trainer = Trainer(model=model, args=args, train_dataset=encode(train), eval_dataset=encode(validation),
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - start) / 60
    history = [{"epoch": h["epoch"], "macro_f1": h["eval_macro_f1"], "accuracy": h["eval_accuracy"]}
               for h in trainer.state.log_history if "eval_macro_f1" in h]

    output = trainer.predict(encode(test))
    probabilities = torch.softmax(torch.tensor(output.predictions, dtype=torch.float32), -1).numpy()
    predictions = probabilities.argmax(-1)
    trainer.save_model("models/gbert-news")
    tokenizer.save_pretrained("models/gbert-news")

    model = trainer.model.eval()
    texts = list(test["text"])

    def predict_on(device, batch=32):
        model.to(device)

        def run(chunk):
            with torch.no_grad():
                for i in range(0, len(chunk), batch):
                    inputs = tokenizer(chunk[i:i + batch], truncation=True, max_length=MAX_LENGTH, padding=True,
                                       return_tensors="pt").to(device)
                    model(**inputs)
        return run

    gpu_ms = per_document_ms(predict_on("cuda"), texts) if torch.cuda.is_available() else None
    cpu_ms = per_document_ms(predict_on("cpu", batch=8), texts[:96])
    result = {"model": f"{MODEL}, fine-tuned", "max_length": MAX_LENGTH, "epochs_validation": history,
              **scores(test["label"], predictions), "train_minutes": train_minutes,
              "ms_per_document_gpu": gpu_ms, "ms_per_document_cpu": cpu_ms,
              "model_size_mb": sum(p.numel() for p in model.parameters()) * 4 / 1e6}
    save("gbert", result, predictions)
    np.save("results/gbert_test_probabilities.npy", probabilities)
    print({k: v for k, v in result.items() if k != "per_class_f1"})


if __name__ == "__main__":
    main()
