import hashlib
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


def generate_risk_signal(document: RawDocument) -> RiskSignal:
    """Transform one normalized document into an explainable baseline signal."""
    text = f"{document.title}. {document.text}".strip()

    sentiment_score, sentiment_confidence, positive, negative = (
        analyze_sentiment(text)
    )
    event_type, event_confidence, event_matches = classify_event(text)
    entities = extract_entities(text)

    impact_score = EVENT_IMPACT[event_type]

    if sentiment_score <= -0.50:
        impact_score += 1

    severe_matches = [
        term for term in SEVERE_TERMS
        if term.casefold() in text.casefold()
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

    candidate_url = document.url
    source_url = None

    if candidate_url:
        try:
            parsed_url = urlsplit(candidate_url)
            if parsed_url.scheme in ("http", "https") and parsed_url.netloc:
                source_url = candidate_url
        except ValueError:
            source_url = None

    source = SourceInfo(
        type=document.source_type,
        name=document.source_name,
        url=source_url,
    )

    identity = (
        document.document_id
        or document.url
        or f"{document.source_name}:{document.title}"
    )
    signal_hash = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    event_evidence = ", ".join(event_matches) or "no event keywords matched"
    positive_evidence = ", ".join(positive) or "none"
    negative_evidence = ", ".join(negative) or "none"

    rationale = (
        f"Rule-based baseline. Event evidence: {event_evidence}. "
        f"Positive terms: {positive_evidence}. "
        f"Negative terms: {negative_evidence}. "
        "Novelty is set to a neutral 0.50 because historical comparison "
        "is not implemented."
    )

    return RiskSignal(
        headline=document.title,
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
        rationale=rationale,
        model_versions={
            "sentiment": "lexicon-baseline-v1",
            "event": "keyword-rules-v1",
            "entity": "company-alias-matcher-v1",
            "impact": "heuristic-v1",
            "novelty": "neutral-placeholder-v1",
        },
    )
