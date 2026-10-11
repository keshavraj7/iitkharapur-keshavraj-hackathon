import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BENCHMARK_DIR = DATA_DIR / "raw" / "benchmarks"
PHRASEBANK_PATH = (
    DATA_DIR / "cache" / "external" / "financial_phrasebank_75agree.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "sentiment"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

TWITTER_VALIDATION = (
    BENCHMARK_DIR / "twitter_sentiment_validation.csv"
)

FINBERT = "ProsusAI/finbert"
FINTWITBERT = "StephanAkkerman/FinTwitBERT-sentiment"

LABELS = ["negative", "neutral", "positive"]


def canonical_label(value: str) -> str:
    value = str(value).strip().lower()

    if "bear" in value or "negative" in value:
        return "negative"

    if "bull" in value or "positive" in value:
        return "positive"

    if "neutral" in value:
        return "neutral"

    raise ValueError(
        f"Unrecognized sentiment label {value!r}. "
        "Inspect the dataset/model label mapping before continuing."
    )


def load_labeled_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)

    if not {"text", "label_name"}.issubset(frame.columns):
        raise ValueError(
            f"{path.name} must contain text and label_name columns; "
            f"found {list(frame.columns)}"
        )

    frame = frame[["text", "label_name"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame[frame["text"].str.strip().ne("")].reset_index(drop=True)
    frame["true_label"] = frame["label_name"].map(canonical_label)

    return frame[["text", "true_label"]]


def load_model(model_id: str):
    print(f"\nLoading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    label_map = {
        int(index): canonical_label(label)
        for index, label in model.config.id2label.items()
    }

    if set(label_map.values()) != set(LABELS):
        raise ValueError(
            f"{model_id} does not expose the expected three sentiment labels: "
            f"{label_map}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model labels: {label_map}")

    return tokenizer, model, label_map, device


def predict_dataset(
    model_id: str,
    tokenizer,
    model,
    label_map: dict[int, str],
    device,
    frame: pd.DataFrame,
    dataset_name: str,
    batch_size: int,
    note: str,
) -> dict:
    texts = frame["text"].tolist()
    y_true = frame["true_label"].tolist()

    prediction_labels = []
    probabilities = []
    started = time.perf_counter()

    print(
        f"Running {model_id} on {dataset_name}: "
        f"{len(texts):,} examples"
    )

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        probabilities.extend(batch_probs.tolist())
        prediction_labels.extend(
            label_map[int(np.argmax(row))]
            for row in batch_probs
        )

        completed = min(start + len(batch_texts), len(texts))
        print(
            f"\r  Completed {completed:,}/{len(texts):,}",
            end="",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    probs = np.asarray(probabilities)

    if len(prediction_labels) != len(y_true):
        raise RuntimeError("Prediction count does not match label count.")

    metrics = {
        "model_id": model_id,
        "dataset": dataset_name,
        "note": note,
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, prediction_labels)),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                prediction_labels,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                prediction_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                prediction_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "class_labels": LABELS,
        "confusion_matrix": confusion_matrix(
            y_true,
            prediction_labels,
            labels=LABELS,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            prediction_labels,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "inference_seconds": round(elapsed, 3),
        "documents_per_second": round(len(texts) / elapsed, 2) if elapsed else None,
    }

    prediction_frame = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "predicted_label": prediction_labels,
        "confidence": probs.max(axis=1),
    })

    for index, label in label_map.items():
        prediction_frame[f"prob_{label}"] = probs[:, index]

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PREDICTION_DIR / f"{dataset_name}_{model_id.split('/')[-1]}.csv"
    prediction_frame.to_csv(output_file, index=False, encoding="utf-8")

    metrics["predictions_file"] = str(output_file.relative_to(ROOT))

    print()
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:  {metrics['weighted_f1']:.4f}")
    print(f"Docs/second:  {metrics['documents_per_second']}")
    print(f"Predictions:  {metrics['predictions_file']}")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-phrasebank",
        action="store_true",
        help="Skip the additional Financial PhraseBank sanity check.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    twitter = load_labeled_csv(TWITTER_VALIDATION)
    results = []

    # FinBERT: evaluate on the same held-out Twitter set for comparison.
    tokenizer, model, label_map, device = load_model(FINBERT)

    results.append(
        predict_dataset(
            FINBERT,
            tokenizer,
            model,
            label_map,
            device,
            twitter,
            "twitter_validation",
            args.batch_size,
            (
                "Cross-domain evaluation on the Twitter benchmark validation "
                "split; check dataset provenance for potential overlap."
            ),
        )
    )

    if not args.skip_phrasebank:
        phrasebank = load_labeled_csv(PHRASEBANK_PATH)

        results.append(
            predict_dataset(
                FINBERT,
                tokenizer,
                model,
                label_map,
                device,
                phrasebank,
                "phrasebank_75agree",
                args.batch_size,
                (
                    "Sanity check only. ProsusAI/finbert was fine-tuned on "
                    "Financial PhraseBank, so this is not an independent "
                    "unseen test result."
                ),
            )
        )

    del tokenizer, model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FinTwitBERT: evaluate on the same held-out Twitter validation set.
    tokenizer, model, label_map, device = load_model(FINTWITBERT)

    results.append(
        predict_dataset(
            FINTWITBERT,
            tokenizer,
            model,
            label_map,
            device,
            twitter,
            "twitter_validation",
            args.batch_size,
            (
                "Financial-tweet sentiment model evaluated on the supplied "
                "validation split; verify source overlap before claiming "
                "fully independent generalization."
            ),
        )
    )

    del tokenizer, model
    gc.collect()

    report = {
        "task": "Financial sentiment model comparison",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "label_order": LABELS,
        "dataset_sizes": {
            "twitter_validation": len(twitter),
        },
        "results": results,
        "interpretation": (
            "Compare FinBERT and FinTwitBERT on the same Twitter validation "
            "examples. Financial PhraseBank results are a sanity check, not "
            "independent evaluation because FinBERT was fine-tuned on that "
            "dataset. Do not infer market-prediction ability from sentiment "
            "classification metrics."
        ),
    }

    report_path = OUTPUT_DIR / "sentiment_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nEvaluation report saved to: {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
