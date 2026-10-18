import hashlib
import logging
import os
from urllib.parse import urlsplit

from app.nlp.classifier import (
    analyze_sentiment,
    classify_event,
    extract_entities,
)
from app.schemas.document import RawDocument
from app.schemas.signal import (
    EventResult,
    ImpactResult,
    RiskSignal,
    SentimentResult,
    SourceInfo,
)


logger = logging.getLogger(__name__)

EVENT_IMPACT = {
    "Geopolitical": 6,
    "Macroeconomic": 5,
    "Credit Event": 7,
    "M&A": 4,
    "Earnings": 5,
    "Regulatory/Legal": 6,
    "Product/Tech": 3,
    "Cyber/Operational": 7,
    "Supply Chain/Commodity": 5,
    "Other": 3,
}

SEVERE_TERMS = (
    "bankruptcy", "default", "insolvency", "fraud",
    "cyberattack", "data breach", "ransomware",
    "war", "bank run", "liquidity crisis",
)


def generate_risk_signal(
    document: RawDocument,
    mode: str | None = None,
) -> RiskSignal:
    """Generate one signal using pretrained models or the rules baseline."""
    selected_mode = (
        mode or os.getenv("FINRISK_NLP_MODE", "transformer")
    ).strip().lower()

    if selected_mode not in {"transformer", "rules"}:
        raise ValueError("FINRISK_NLP_MODE must be 'transformer' or 'rules'.")

    text = f"{document.title}. {document.text}".strip()
    sentiment_result = None
    event_result = None
    predicted_topic = None
    used_mode = selected_mode

    if selected_mode == "transformer":
        try:
            from app.nlp.model_inference import predict_event, predict_sentiment

            content_type = str(
                document.metadata.get("content_type", "")
            ).lower()

            sentiment_source = (
                "social"
                if document.source_type == "social"
                or content_type == "social"
                else document.source_type
            )

            sentiment_result = predict_sentiment(text, sentiment_source)
            event_result = predict_event(text)
            sentiment_score = float(sentiment_result["score"])
            sentiment_confidence = float(sentiment_result["confidence"])
            event_type = event_result["type"]
            event_confidence = float(event_result["confidence"])
            predicted_topic = event_result["topic"]

        except Exception:
            if os.getenv("FINRISK_NLP_FALLBACK", "rules").lower() != "rules":
                raise

            logger.exception(
                "Transformer inference failed; using the rules fallback."
            )
            used_mode = "rules-fallback"
            sentiment_score, sentiment_confidence, positive, negative = (
                analyze_sentiment(text)
            )
            event_type, event_confidence, event_matches = classify_event(text)

    else:
        sentiment_score, sentiment_confidence, positive, negative = (
            analyze_sentiment(text)
        )
        event_type, event_confidence, event_matches = classify_event(text)

    entities = extract_entities(text)

    impact_score = EVENT_IMPACT[event_type]

    if sentiment_score <= -0.50:
        impact_score += 1

    lowered_text = text.casefold()
    severe_matches = [
        term for term in SEVERE_TERMS if term.casefold() in lowered_text
    ]

    if severe_matches:
        impact_score += 1

    impact_score = min(10, max(1, impact_score))

    if event_type in ("Credit Event", "Cyber/Operational"):
        horizon = "1h" if severe_matches else "1d"
    elif event_type in ("Macroeconomic", "Supply Chain/Commodity"):
        horizon = "1w"
    else:
        horizon = "1d"

    source_url = None
    if document.url:
        try:
            parsed = urlsplit(document.url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                source_url = document.url
        except ValueError:
            source_url = None

    source = SourceInfo(
        type=document.source_type,
        name=document.source_name,
        url=source_url,
    )

    identity = document.document_id or document.url or (
        f"{document.source_name}:{document.title}"
    )
    signal_hash = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    if sentiment_result is not None and event_result is not None:
        sentiment_probabilities = sentiment_result["probabilities"]
        rationale = (
            f"Pretrained-model analysis. Predicted topic: {predicted_topic}. "
            f"Mapped event: {event_type}. "
            f"Sentiment probabilities: {sentiment_probabilities}. "
            "Impact is assigned by the existing heuristic severity rules; "
            "it is not a model-calibrated market-loss prediction."
        )
        model_versions = {
            "sentiment": sentiment_result["model_id"],
            "event": event_result["model_id"],
            "impact": "heuristic-v1",
            "novelty": "neutral-placeholder-v1",
        }
    else:
        positive_terms = ", ".join(positive) or "none"
        negative_terms = ", ".join(negative) or "none"
        event_terms = ", ".join(event_matches) or "none"

        rationale = (
            f"Keyword-rule fallback. Event evidence: {event_terms}. "
            f"Positive terms: {positive_terms}. "
            f"Negative terms: {negative_terms}. "
            "Scores are heuristic and not calibrated probabilities."
        )
        model_versions = {
            "sentiment": "lexicon-baseline-v1",
            "event": "keyword-rules-v1",
            "impact": "heuristic-v1",
            "novelty": "neutral-placeholder-v1",
        }

    if used_mode == "rules-fallback":
        model_versions["pipeline"] = "transformer-fallback-to-rules"
    else:
        model_versions["pipeline"] = used_mode

    return RiskSignal(
        signal_id=f"sig_{signal_hash}",
        event_time=document.observed_at or document.ingested_at,
        source=source,
        entities=entities,
        sentiment=SentimentResult(
            score=sentiment_score,
            confidence=sentiment_confidence,
        ),
        event=EventResult(
            type=event_type,
            confidence=event_confidence,
        ),
        impact=ImpactResult(
            score=impact_score,
            horizon=horizon,
        ),
        novelty=0.50,
        headline=document.title,
        rationale=rationale,
        model_versions=model_versions,
    )
