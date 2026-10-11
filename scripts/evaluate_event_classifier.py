import json
import re
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
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/raw/benchmarks/twitter_topic_validation.csv"
MAPPING_PATH = ROOT / "data/config/event_topic_mapping.json"
OUTPUT_DIR = ROOT / "outputs/events"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

MODEL_ID = (
    "leonas5555/finnews-topic-single-classify"
)

TOPICS = [
    "Analyst Update",
    "Fed | Central Banks",
    "Company | Product News",
    "Treasuries | Corporate Debt",
    "Dividend",
    "Earnings",
    "Energy | Oil",
    "Financials",
    "Currencies",
    "General News | Opinion",
    "Gold | Metals | Materials",
    "IPO",
    "Legal | Regulation",
    "M&A | Investments",
    "Macro",
    "Markets",
    "Politics",
    "Personnel Change",
    "Stock Commentary",
    "Stock Movement",
]

EVENT_TYPES = [
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "M&A",
    "Earnings",
    "Regulatory/Legal",
    "Product/Tech",
    "Cyber/Operational",
    "Supply Chain/Commodity",
    "Other",
]


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def resolve_topic_label(value, id_to_label=None) -> str:
    """Resolve a topic label from either LABEL_n or its human-readable name."""
    text = str(value).strip()
    match = re.fullmatch(r"LABEL_(\d+)", text, flags=re.IGNORECASE)

    if match:
        index = int(match.group(1))

        if index < 0 or index >= len(TOPICS):
            raise ValueError(f"Invalid topic label ID: {value}")

        return TOPICS[index]

    normalized = normalize_label(text)
    lookup = {normalize_label(topic): topic for topic in TOPICS}

    if normalized in lookup:
        return lookup[normalized]

    if id_to_label is not None:
        raise ValueError(f"Model emitted an unknown topic label: {value}")

    raise ValueError(f"Unknown dataset topic label: {value}")


def main() -> None:
    for path in (DATASET_PATH, MAPPING_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with DATASET_PATH.open("r", encoding="utf-8-sig") as file:
        frame = pd.read_csv(file)

    with MAPPING_PATH.open("r", encoding="utf-8-sig") as file:
        topic_mapping = json.load(file)

    if set(topic_mapping) != set(TOPICS):
        missing = sorted(set(TOPICS) - set(topic_mapping))
        extra = sorted(set(topic_mapping) - set(TOPICS))
        raise ValueError(
            f"Topic mapping must cover all 20 labels. Missing={missing}, extra={extra}"
        )

    invalid_events = sorted(set(topic_mapping.values()) - set(EVENT_TYPES))
    if invalid_events:
        raise ValueError(f"Invalid mapped event types: {invalid_events}")

    if not {"text", "label_name"}.issubset(frame.columns):
        raise ValueError(
            f"Unexpected dataset columns: {list(frame.columns)}"
        )

    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame[frame["text"].str.strip().ne("")].reset_index(drop=True)
    frame["true_topic"] = frame["label_name"].map(resolve_topic_label)
    frame["true_event"] = frame["true_topic"].map(topic_mapping)

    print(f"Validation samples: {len(frame):,}")
    print(f"Loading {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    id_to_label = {
        int(index): str(label)
        for index, label in model.config.id2label.items()
    }

    if model.config.num_labels != len(TOPICS):
        raise ValueError(
            f"Expected 20 model labels, got {model.config.num_labels}"
        )

    model_label_topics = {
        index: resolve_topic_label(label, id_to_label)
        for index, label in id_to_label.items()
    }

    if set(model_label_topics) != set(range(len(TOPICS))):
        raise ValueError(f"Unexpected model label indices: {model_label_topics}")

    print("Model label mapping verified.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Inference device: {device}")

    texts = frame["text"].tolist()
    predicted_topics = []
    confidences = []
    batch_size = 16
    started = time.perf_counter()

    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.inference_mode():
            probabilities = torch.softmax(model(**batch).logits, dim=-1)

        probs = probabilities.cpu().numpy()

        for row in probs:
            predicted_id = int(np.argmax(row))
            predicted_topics.append(model_label_topics[predicted_id])
            confidences.append(float(row[predicted_id]))

        print(
            f"\rProcessed {min(start + len(probs), len(texts)):,}/{len(texts):,}",
            end="",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    print()

    frame["predicted_topic"] = predicted_topics
    frame["predicted_event"] = frame["predicted_topic"].map(topic_mapping)
    frame["confidence"] = confidences

    topic_report = classification_report(
        frame["true_topic"],
        frame["predicted_topic"],
        labels=TOPICS,
        output_dict=True,
        zero_division=0,
    )

    # Only evaluate mapped classes represented by at least one true validation
    # example. The uncovered EventType classes are reported separately.
    supported_events = [
        event for event in EVENT_TYPES
        if event in set(frame["true_event"])
    ]

    event_report = classification_report(
        frame["true_event"],
        frame["predicted_event"],
        labels=supported_events,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model_id": MODEL_ID,
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "sample_count": len(frame),
        "device": str(device),
        "batch_size": batch_size,
        "max_length": 128,
        "inference_seconds": round(elapsed, 3),
        "documents_per_second": round(len(frame) / elapsed, 2) if elapsed else None,
        "topic_classification": {
            "accuracy": float(
                accuracy_score(frame["true_topic"], frame["predicted_topic"])
            ),
            "macro_f1": float(
                f1_score(
                    frame["true_topic"],
                    frame["predicted_topic"],
                    labels=TOPICS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    frame["true_topic"],
                    frame["predicted_topic"],
                    labels=TOPICS,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "labels": TOPICS,
            "confusion_matrix": confusion_matrix(
                frame["true_topic"],
                frame["predicted_topic"],
                labels=TOPICS,
            ).tolist(),
            "classification_report": topic_report,
        },
        "mapped_event_classification": {
            "accuracy": float(
                accuracy_score(frame["true_event"], frame["predicted_event"])
            ),
            "macro_f1": float(
                f1_score(
                    frame["true_event"],
                    frame["predicted_event"],
                    labels=supported_events,
                    average="macro",
                    zero_division=0,
                )
            ),
            "labels_evaluated": supported_events,
            "uncovered_event_types": [
                event for event in EVENT_TYPES if event not in supported_events
            ],
            "confusion_matrix": confusion_matrix(
                frame["true_event"],
                frame["predicted_event"],
                labels=supported_events,
            ).tolist(),
            "classification_report": event_report,
            "mapping_file": str(MAPPING_PATH.relative_to(ROOT)),
            "warning": (
                "Mapped event metrics depend on manually chosen many-to-one "
                "topic mapping; they are not gold labels for real-world events."
            ),
        },
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = PREDICTIONS_DIR / "twitter_topic_validation.csv"
    frame["confidence"] = frame["confidence"].round(6)
    frame.to_csv(prediction_path, index=False, encoding="utf-8")

    results["predictions_file"] = str(prediction_path.relative_to(ROOT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "event_evaluation.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Topic accuracy: {results['topic_classification']['accuracy']:.4f}")
    print(f"Topic macro-F1: {results['topic_classification']['macro_f1']:.4f}")
    print(
        "Mapped-event accuracy: "
        f"{results['mapped_event_classification']['accuracy']:.4f}"
    )
    print(
        "Mapped-event macro-F1: "
        f"{results['mapped_event_classification']['macro_f1']:.4f}"
    )
    print(f"Documents/sec: {results['documents_per_second']}")
    print(f"Report: {report_path.relative_to(ROOT)}")
    print(f"Predictions: {prediction_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

