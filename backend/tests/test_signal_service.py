from app.schemas.document import RawDocument
from app.services.signal_service import generate_risk_signal


def test_credit_default_produces_negative_credit_signal():
    document = RawDocument(
        document_id="synthetic-credit-001",
        source_type="news",
        source_name="Synthetic Demo",
        title="JPMorgan reports losses after a debt default",
        text="JPMorgan reports increasing losses following a debt default.",
    )

    signal = generate_risk_signal(document, mode="rules")

    assert signal.event.type == "Credit Event"
    assert signal.sentiment.score < 0
    assert signal.impact.score >= 1
    assert any(entity.ticker == "JPM" for entity in signal.entities)


def test_positive_earnings_news_identifies_nvidia():
    document = RawDocument(
        document_id="synthetic-earnings-001",
        source_type="news",
        source_name="Synthetic Demo",
        title="NVIDIA reports record revenue growth",
        text="NVIDIA reports record revenue growth.",
    )

    signal = generate_risk_signal(document, mode="rules")

    assert signal.event.type == "Earnings"
    assert signal.sentiment.score > 0
    assert any(entity.ticker == "NVDA" for entity in signal.entities)
    assert signal.model_versions["sentiment"] == "lexicon-baseline-v1"

