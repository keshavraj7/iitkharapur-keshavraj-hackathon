import math
from datetime import datetime, timezone

from app.schemas.rebalancing import (
    PortfolioPosition,
    PositionRecommendation,
    RebalanceResponse,
)
from app.schemas.signal import RiskSignal


EPSILON = 1e-9
SUM_TOLERANCE = 1e-5
OVERRIDE_EVENTS = {"Credit Event", "Regulatory/Legal"}
OVERRIDE_IMPACT_THRESHOLD = 8
OVERRIDE_BASE_MULTIPLIER = 0.50
SIGNAL_HALF_LIFE_HOURS = 24.0


def _validate_portfolio(positions: list[PortfolioPosition]) -> None:
    tickers = [p.ticker.strip().upper() for p in positions]

    if len(set(tickers)) != len(tickers):
        raise ValueError("Portfolio tickers must be unique.")

    if any(not ticker for ticker in tickers):
        raise ValueError("Portfolio tickers cannot be blank.")

    if not any(p.base_weight > 0 for p in positions):
        raise ValueError("At least one position must have a positive base weight.")

    if abs(sum(p.base_weight for p in positions) - 1.0) > SUM_TOLERANCE:
        raise ValueError("Base weights must sum to 1.0.")

    if abs(sum(p.current_weight for p in positions) - 1.0) > SUM_TOLERANCE:
        raise ValueError("Current weights must sum to 1.0.")

    for position in positions:
        if position.ticker.strip().upper() != "CASH" and position.base_weight <= 0:
            raise ValueError(
                f"{position.ticker}: non-cash positions need a positive base weight."
            )


def _project_to_bounds(
    preferences: list[float],
    lower: list[float],
    upper: list[float],
    fixed: dict[int, float],
) -> list[float]:
    """Project positive preferences onto weight bounds while summing to one."""
    result = [0.0] * len(preferences)

    for i, value in fixed.items():
        if value < lower[i] - SUM_TOLERANCE or value > upper[i] + SUM_TOLERANCE:
            # Event-overridden positions intentionally use a separate, stricter cap.
            if not (lower[i] == upper[i] == value):
                raise ValueError(f"Fixed allocation for index {i} violates its bounds.")
        result[i] = value

    active = [i for i in range(len(preferences)) if i not in fixed]
    remaining = 1.0 - sum(fixed.values())

    if remaining < -SUM_TOLERANCE:
        raise ValueError("Fixed portfolio allocations exceed 100%.")

    lower_total = sum(lower[i] for i in active)
    upper_total = sum(upper[i] for i in active)

    if remaining < lower_total - SUM_TOLERANCE or remaining > upper_total + SUM_TOLERANCE:
        raise ValueError(
            "Portfolio constraints are infeasible. Check base weights, current "
            "weights, event overrides, and whether a CASH position is included."
        )

    if not active:
        if abs(remaining) > SUM_TOLERANCE:
            raise ValueError("Portfolio constraints leave an unallocated weight.")
        return result

    def allocate(scale: float) -> float:
        return sum(
            min(upper[i], max(lower[i], max(preferences[i], EPSILON) * scale))
            for i in active
        )

    low, high = 0.0, 1.0
    while allocate(high) < remaining and high < 1e18:
        high *= 2.0

    if allocate(high) < remaining - SUM_TOLERANCE:
        raise ValueError("Unable to normalize portfolio weights under these constraints.")

    for _ in range(100):
        middle = (low + high) / 2.0
        if allocate(middle) < remaining:
            low = middle
        else:
            high = middle

    scale = (low + high) / 2.0

    for i in active:
        result[i] = min(
            upper[i],
            max(lower[i], max(preferences[i], EPSILON) * scale),
        )

    # Correct floating-point residue without breaking the allocation bounds.
    residue = 1.0 - sum(result)

    if abs(residue) > 1e-10:
        for i in active:
            if residue > 0:
                adjustment = min(residue, upper[i] - result[i])
            else:
                adjustment = -min(-residue, result[i] - lower[i])

            result[i] += adjustment
            residue -= adjustment

            if abs(residue) <= 1e-10:
                break

    if abs(sum(result) - 1.0) > 1e-6:
        raise ValueError("Rebalanced portfolio weights do not sum to 1.0.")

    return result


def _signal_scores(
    signals: list[RiskSignal],
    tickers: set[str],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Compute time-decayed sentiment scores and identify severe event overrides."""
    now = datetime.now(timezone.utc)
    weighted_scores: dict[str, float] = {}
    total_weights: dict[str, float] = {}
    overrides: dict[str, list[str]] = {}

    for signal in signals:
        event_time = signal.event_time

        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)

        age_hours = max(0.0, (now - event_time).total_seconds() / 3600.0)
        decay = 0.5 ** (age_hours / SIGNAL_HALF_LIFE_HOURS)

        weight = (
            max(0.0, signal.sentiment.confidence)
            * max(0.0, signal.event.confidence)
            * decay
        )

        for entity in signal.entities:
            ticker = entity.ticker.strip().upper()

            if ticker not in tickers:
                continue

            weighted_scores[ticker] = (
                weighted_scores.get(ticker, 0.0)
                + signal.sentiment.score * weight
            )
            total_weights[ticker] = total_weights.get(ticker, 0.0) + weight

            if (
                signal.event.type in OVERRIDE_EVENTS
                and signal.impact.score >= OVERRIDE_IMPACT_THRESHOLD
            ):
                overrides.setdefault(ticker, []).append(
                    f"{signal.event.type}, impact {signal.impact.score}/10"
                )

    scores = {
        ticker: (
            weighted_scores.get(ticker, 0.0) / total_weights[ticker]
            if total_weights.get(ticker, 0.0) > 0
            else 0.0
        )
        for ticker in tickers
    }

    return scores, overrides


def calculate_rebalance(
    positions: list[PortfolioPosition],
    signals: list[RiskSignal],
    sensitivity: float = 0.75,
    no_trade_band: float = 0.005,
    max_turnover: float = 0.10,
) -> RebalanceResponse:
    """Calculate target allocations and a turnover-limited proposed allocation."""
    _validate_portfolio(positions)

    if not 0 <= sensitivity <= 2:
        raise ValueError("Sensitivity must be between 0 and 2.")
    if not 0 <= no_trade_band <= 0.05:
        raise ValueError("No-trade band must be between 0 and 0.05.")
    if not 0 <= max_turnover <= 1:
        raise ValueError("Maximum turnover must be between 0 and 1.")

    tickers = [p.ticker.strip().upper() for p in positions]
    ticker_set = set(tickers)
    cash_indices = [i for i, ticker in enumerate(tickers) if ticker == "CASH"]

    scores, overrides = _signal_scores(signals, ticker_set)
    override_indices: dict[int, float] = {}

    current = [p.current_weight for p in positions]
    base = [p.base_weight for p in positions]
    mandatory = current.copy()
    mandatory_sold = 0.0

    for i, ticker in enumerate(tickers):
        if ticker not in overrides or ticker == "CASH":
            continue

        cut_ceiling = OVERRIDE_BASE_MULTIPLIER * base[i]
        new_weight = min(current[i], cut_ceiling)
        mandatory[i] = new_weight
        mandatory_sold += current[i] - new_weight
        override_indices[i] = new_weight

    if mandatory_sold > SUM_TOLERANCE:
        if not cash_indices:
            raise ValueError(
                "A high-impact event requires a reduction, but the portfolio "
                "has no CASH position to receive the proceeds. Add a CASH position."
            )
        mandatory[cash_indices[0]] += mandatory_sold

    lower, upper, preferences = [], [], []

    for i, position in enumerate(positions):
        ticker = tickers[i]

        if ticker == "CASH":
            lower.append(0.0)
            upper.append(1.0)
        elif i in override_indices:
            lower.append(override_indices[i])
            upper.append(override_indices[i])
        else:
            lower.append(0.5 * base[i])
            upper.append(min(2.0 * base[i], 1.0))

            if current[i] < lower[i] - SUM_TOLERANCE or current[i] > upper[i] + SUM_TOLERANCE:
                raise ValueError(
                    f"{ticker}: current weight is outside the 0.5x–2x base "
                    "weight band. Bring it within bounds before rebalancing."
                )

        preferences.append(
            max(base[i], EPSILON) * math.exp(sensitivity * scores[ticker])
        )

    fixed = dict(override_indices)

    if mandatory_sold > SUM_TOLERANCE and cash_indices:
        fixed[cash_indices[0]] = mandatory[cash_indices[0]]
    first_target = _project_to_bounds(preferences, lower, upper, fixed)

    # Freeze ordinary positions whose proposed change is below the no-trade band.
    for i, ticker in enumerate(tickers):
        if i in override_indices or ticker == "CASH":
            continue

        if abs(first_target[i] - mandatory[i]) < no_trade_band:
            fixed[i] = mandatory[i]

    target = _project_to_bounds(preferences, lower, upper, fixed)

    mandatory_turnover = 0.5 * sum(
        abs(mandatory[i] - current[i]) for i in range(len(positions))
    )
    requested_discretionary_turnover = 0.5 * sum(
        abs(target[i] - mandatory[i]) for i in range(len(positions))
    )

    scale = (
        1.0
        if requested_discretionary_turnover <= max_turnover
        else max_turnover / requested_discretionary_turnover
        if requested_discretionary_turnover > 0
        else 1.0
    )

    proposed = [
        mandatory[i] + scale * (target[i] - mandatory[i])
        for i in range(len(positions))
    ]

    one_way_turnover = 0.5 * sum(
        abs(proposed[i] - current[i]) for i in range(len(positions))
    )
    discretionary_turnover = scale * requested_discretionary_turnover

    recommendations = []

    for i, position in enumerate(positions):
        ticker = tickers[i]

        if i in override_indices:
            reason = (
                "Mandatory reduction: "
                + "; ".join(overrides[ticker])
                + f". Weight capped at {OVERRIDE_BASE_MULTIPLIER:.0%} of base."
            )
        elif ticker == "CASH":
            reason = "Cash reserve absorbs event-driven cuts and portfolio reallocation."
        elif abs(proposed[i] - current[i]) < 1e-8:
            reason = "No allocation change after sentiment rules and constraints."
        else:
            reason = (
                f"Time-decayed sentiment score {scores[ticker]:+.3f}; "
                "weight adjusted subject to bounds, no-trade band and turnover cap."
            )

        recommendations.append(
            PositionRecommendation(
                ticker=ticker,
                company=position.company,
                sector=position.sector,
                base_weight=round(base[i], 8),
                current_weight=round(current[i], 8),
                target_weight=round(target[i], 8),
                proposed_weight=round(proposed[i], 8),
                change=round(proposed[i] - current[i], 8),
                signal_score=round(scores[ticker], 4),
                event_override=i in override_indices,
                rationale=reason,
            )
        )

    return RebalanceResponse(
        positions=recommendations,
        signals_used=len(signals),
        one_way_turnover=round(one_way_turnover, 6),
        mandatory_turnover=round(mandatory_turnover, 6),
        discretionary_turnover=round(discretionary_turnover, 6),
        max_turnover=max_turnover,
        turnover_cap_applied=scale < 1.0,
        summary=(
            "Allocations are experimental, constraint-based recommendations. "
            "Mandatory high-impact cuts may exceed the discretionary turnover budget."
        ),
    )
