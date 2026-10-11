import json
import re
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FINBERT_MODEL = "ProsusAI/finbert"
FINTWITBERT_MODEL = "StephanAkkerman/FinTwitBERT-sentiment"
EVENT_MODEL = "leonas5555/finnews-topic-single-classify"

ROOT = Path(__file__).resolve().parents[3]
EVENT_MAPPING_PATH = ROOT / "data" / "config" / "event_topic_mapping.json"

TOPIC_LABELS = [
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


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _sentiment_label(value: str) -> str:
    label = str(value).strip().casefold()

    if "bear" in label or "negative" in label:
        return "negative"
    if "bull" in label or "positive" in label:
        return "positive"
    if "neutral" in label:
        return "neutral"

    raise ValueError(f"Unsupported sentiment model label: {value!r}")


def _topic_label(value: str) -> str:
    label = str(value).strip()
    match = re.fullmatch(r"LABEL_(\d+)", label, flags=re.IGNORECASE)

    if match:
        index = int(match.group(1))
        if 0 <= index < len(TOPIC_LABELS):
            return TOPIC_LABELS[index]
        raise ValueError(f"Invalid topic label ID: {label}")

    topic_by_key = {_normalize_label(topic): topic for topic in TOPIC_LABELS}
    key = _normalize_label(label)

    if key not in topic_by_key:
        raise ValueError(f"Unknown event topic label: {label!r}")

    return topic_by_key[key]


@lru_cache(maxsize=3)
def _load_model(model_id: str):
    """Load each model once per Python process."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def _probabilities(text: str, model_id: str) -> tuple[dict[int, float], dict[int, str]]:
    tokenizer, model, device = _load_model(model_id)

    encoded = tokenizer(
        str(text)[:12000],
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    raw_labels = model.config.id2label
    labels = {
        int(index): str(label)
        for index, label in raw_labels.items()
    }

    if set(labels) != set(range(len(probs))):
        raise ValueError(f"Unexpected model label indices for {model_id}: {labels}")

    return {i: float(prob) for i, prob in enumerate(probs)}, labels


def predict_sentiment(text: str, source_type: str) -> dict:
    """Return sentiment score P(positive)-P(negative) and confidence."""
    social_source = source_type == "social"
    model_id = FINTWITBERT_MODEL if social_source else FINBERT_MODEL

    probs, raw_labels = _probabilities(text, model_id)

    canonical_probs: dict[str, float] = {}

    for index, prob in probs.items():
        label = _sentiment_label(raw_labels[index])
        canonical_probs[label] = canonical_probs.get(label, 0.0) + prob

    for label in ("negative", "neutral", "positive"):
        canonical_probs.setdefault(label, 0.0)

    score = canonical_probs["positive"] - canonical_probs["negative"]
    predicted = max(canonical_probs, key=canonical_probs.get)

    return {
        "score": max(-1.0, min(1.0, score)),
        "confidence": max(canonical_probs.values()),
        "label": predicted,
        "probabilities": canonical_probs,
        "model_id": model_id,
    }


@lru_cache(maxsize=1)
def _load_event_mapping() -> dict[str, str]:
    with EVENT_MAPPING_PATH.open("r", encoding="utf-8-sig") as file:
        mapping = json.load(file)

    return {
        _normalize_label(topic): str(event_type)
        for topic, event_type in mapping.items()
    }


def predict_event(text: str) -> dict:
    """Predict the 20-topic label and aggregate probability into the 10 event types."""
    probs, raw_labels = _probabilities(text, EVENT_MODEL)
    mapping = _load_event_mapping()

    topic_probs: dict[str, float] = {}
    event_probs: dict[str, float] = {}

    for index, prob in probs.items():
        topic = _topic_label(raw_labels[index])
        topic_probs[topic] = topic_probs.get(topic, 0.0) + prob

        key = _normalize_label(topic)

        if key not in mapping:
            raise ValueError(f"Topic {topic!r} is missing from the event mapping.")

        event_type = mapping[key]
        event_probs[event_type] = event_probs.get(event_type, 0.0) + prob

    topic = max(topic_probs, key=topic_probs.get)
    event_type = max(event_probs, key=event_probs.get)

    return {
        "type": event_type,
        "confidence": max(0.0, min(1.0, event_probs[event_type])),
        "topic": topic,
        "topic_confidence": topic_probs[topic],
        "probabilities": event_probs,
        "model_id": EVENT_MODEL,
    }
