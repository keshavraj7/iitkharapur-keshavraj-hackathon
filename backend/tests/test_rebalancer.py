import pytest

from app.schemas.rebalancing import PortfolioPosition
from app.schemas.signal import (
    EventResult,
    ImpactResult,
    RiskSignal,
    SentimentResult,
    SourceInfo,
)
from app.services.rebalancer import calculate_rebalance


def sample_positions():
    return [
        PortfolioPosition(ticker="AAPL", company="Apple", sector="Technology", base_weight=0.20, current_weight=0.20),
        PortfolioPosition(ticker="MSFT", company="Microsoft", sector="Technology", base_weight=0.20, current_weight=0.20),
        PortfolioPosition(ticker="NVDA", company="NVIDIA", sector="Technology", base_weight=0.15, current_weight=0.15),
        PortfolioPosition(ticker="JPM", company="JPMorgan Chase", sector="Financials", base_weight=0.15, current_weight=0.15),
        PortfolioPosition(ticker="XOM", company="Exxon Mobil", sector="Energy", base_weight=0.10, current_weight=0.10),
        PortfolioPosition(ticker="PFE", company="Pfizer", sector="Healthcare", base_weight=0.10, current_weight=0.10),
        PortfolioPosition(ticker="COST", company="Costco", sector="Consumer Staples", base_weight=0.05, current_weight=0.05),
        PortfolioPosition(ticker="CASH", company="Cash", sector="Cash", base_weight=0.05, current_weight=0.05),
    ]


def sample_signal(
    ticker="AAPL",
    event_type="Earnings",
    sentiment=0.8,
    impact=5,
):
    return RiskSignal(
        signal_id=f"test-{ticker}-{event_type}",
        source=SourceInfo(type="news", name="Synthetic Test"),
        entities=[
            {
                "ticker": ticker,
                "company": ticker,
                "sector": "Technology",
                "confidence": 0.95,
            }
        ],
        sentiment=SentimentResult(score=sentiment, confidence=0.9),
        event=EventResult(type=event_type, confidence=0.9),
        impact=ImpactResult(score=impact, horizon="1d"),
        novelty=0.5,
        rationale="Synthetic unit-test signal.",
        model_versions={"test": "1"},
    )


def test_weights_sum_to_one_and_respect_bounds():
    result = calculate_rebalance(
        positions=sample_positions(),
        signals=[sample_signal()],
        max_turnover=0.05,
    )

    assert sum(p.proposed_weight for p in result.positions) == pytest.approx(1.0)
    assert result.one_way_turnover <= 0.05 + 1e-6

    for position in result.positions:
        assert 0 <= position.proposed_weight <= 1

        if position.ticker != "CASH":
            assert position.proposed_weight >= 0.5 * position.base_weight - 1e-6
            assert position.proposed_weight <= 2.0 * position.base_weight + 1e-6


def test_high_impact_credit_event_cuts_exposure():
    result = calculate_rebalance(
        positions=sample_positions(),
        signals=[
            sample_signal(
                ticker="JPM",
                event_type="Credit Event",
                sentiment=-0.9,
                impact=9,
            )
        ],
        max_turnover=0.02,
    )

    jpm = next(p for p in result.positions if p.ticker == "JPM")

    assert jpm.event_override
    assert jpm.proposed_weight <= 0.5 * jpm.base_weight + 1e-6
    assert result.mandatory_turnover > 0
    assert sum(p.proposed_weight for p in result.positions) == pytest.approx(1.0)


def test_invalid_base_weights_are_rejected():
    positions = sample_positions()
    positions[0] = positions[0].model_copy(update={"base_weight": 0.4})

    with pytest.raises(ValueError, match="Base weights must sum"):
        calculate_rebalance(positions, [])


def test_missing_cash_is_rejected_when_override_requires_sales():
    positions = [
        PortfolioPosition(ticker="JPM", company="JPMorgan", base_weight=0.20, current_weight=0.20),
        PortfolioPosition(ticker="AAPL", company="Apple", base_weight=0.40, current_weight=0.40),
        PortfolioPosition(ticker="MSFT", company="Microsoft", base_weight=0.40, current_weight=0.40),
    ]

    with pytest.raises(ValueError, match="no CASH position"):
        calculate_rebalance(
            positions=positions,
            signals=[
                sample_signal(
                    ticker="JPM",
                    event_type="Credit Event",
                    sentiment=-0.9,
                    impact=9,
                )
            ],
        )
