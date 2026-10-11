import json
from collections import Counter
from pathlib import Path

from app.nlp.classifier import analyze_sentiment, classify_event


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "synthetic" / "risk_eval.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation_report.json"


def sentiment_label(score: float) -> str:
    if score < -0.15:
        return "negative"
    if score > 0.15:
        return "positive"
    return "neutral"


def calculate_metrics(actual: list[str], predicted: list[str]) -> dict:
    if not actual:
        return {"accuracy": 0.0, "macro_f1": 0.0, "count": 0}

    labels = sorted(set(actual))
    accuracy = sum(
        a == p for a, p in zip(actual, predicted)
    ) / len(actual)

    f1_scores = []

    for label in labels:
        tp = sum(
            a == label and p == label
            for a, p in zip(actual, predicted)
        )
        fp = sum(
            a != label and p == label
            for a, p in zip(actual, predicted)
        )
        fn = sum(
            a == label and p != label
            for a, p in zip(actual, predicted)
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        f1_scores.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "count": len(actual),
    }


def main() -> None:
    with DATASET_PATH.open("r", encoding="utf-8-sig") as file:
        dataset = json.load(file)

    items = dataset["items"]
    event_actual, event_predicted = [], []
    sentiment_actual, sentiment_predicted = [], []
    errors = []

    for item in items:
        text = item["text"]

        predicted_event, event_confidence, event_matches = classify_event(text)
        score, sentiment_confidence, positive, negative = analyze_sentiment(text)
        predicted_sentiment = sentiment_label(score)

        expected_event = item["expected_event"]
        expected_sentiment = item["expected_sentiment"]

        event_actual.append(expected_event)
        event_predicted.append(predicted_event)
        sentiment_actual.append(expected_sentiment)
        sentiment_predicted.append(predicted_sentiment)

        if (
            predicted_event != expected_event
            or predicted_sentiment != expected_sentiment
        ):
            errors.append({
                "id": item["id"],
                "text": text,
                "expected_event": expected_event,
                "predicted_event": predicted_event,
                "event_confidence_heuristic": event_confidence,
                "event_matches": event_matches,
                "expected_sentiment": expected_sentiment,
                "predicted_sentiment": predicted_sentiment,
                "sentiment_score": score,
                "positive_matches": positive,
                "negative_matches": negative,
            })

    report = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "source": dataset["source"],
        "evaluation_warning": (
            "Synthetic development set; examples are designed around the "
            "baseline rules. Results are not evidence of real-world "
            "generalization or calibrated confidence."
        ),
        "sample_count": len(items),
        "event_classification": calculate_metrics(
            event_actual, event_predicted
        ),
        "sentiment_classification": calculate_metrics(
            sentiment_actual, sentiment_predicted
        ),
        "event_label_counts": dict(Counter(event_actual)),
        "sentiment_label_counts": dict(Counter(sentiment_actual)),
        "errors": errors,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Samples: {len(items)}")
    print("Event classification:", report["event_classification"])
    print("Sentiment classification:", report["sentiment_classification"])
    print(f"Misclassified samples: {len(errors)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(
                f"- {error['id']}: "
                f"event {error['expected_event']} -> {error['predicted_event']}; "
                f"sentiment {error['expected_sentiment']} -> "
                f"{error['predicted_sentiment']}"
            )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
