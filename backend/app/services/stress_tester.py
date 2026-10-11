from app.schemas.stress_test import (
    PositionStressResult,
    StressPosition,
    StressTestResponse,
)
from app.schemas.signal import EventType


# Adverse scenario assumptions, not empirically calibrated shocks.
# Yield and spread changes are in basis points.
SCENARIOS: dict[str, dict] = {
    "Geopolitical": {
        "equity": -0.08, "yield_bps": 25, "spread_bps": 75, "pd_multiplier": 1.5,
        "sectors": {"Energy": 0.03, "Financials": -0.03, "Technology": -0.02},
    },
    "Macroeconomic": {
        "equity": -0.06, "yield_bps": 50, "spread_bps": 50, "pd_multiplier": 1.4,
        "sectors": {"Financials": -0.02, "Technology": -0.01, "Energy": -0.03},
    },
    "Credit Event": {
        "equity": -0.05, "yield_bps": -25, "spread_bps": 200, "pd_multiplier": 2.5,
        "sectors": {"Financials": -0.07, "Technology": -0.02},
    },
    "M&A": {
        "equity": -0.02, "yield_bps": 10, "spread_bps": 25, "pd_multiplier": 1.1,
        "sectors": {"Technology": -0.02},
    },
    "Earnings": {
        "equity": -0.03, "yield_bps": 0, "spread_bps": 25, "pd_multiplier": 1.2,
        "sectors": {"Technology": -0.03, "Consumer Discretionary": -0.03},
    },
    "Regulatory/Legal": {
        "equity": -0.04, "yield_bps": 0, "spread_bps": 75, "pd_multiplier": 1.3,
        "sectors": {"Financials": -0.04, "Technology": -0.02},
    },
    "Product/Tech": {
        "equity": -0.02, "yield_bps": 0, "spread_bps": 25, "pd_multiplier": 1.1,
        "sectors": {"Technology": -0.08},
    },
    "Cyber/Operational": {
        "equity": -0.03, "yield_bps": 0, "spread_bps": 75, "pd_multiplier": 1.3,
        "sectors": {"Technology": -0.10, "Financials": -0.07},
    },
    "Supply Chain/Commodity": {
        "equity": -0.04, "yield_bps": 25, "spread_bps": 50, "pd_multiplier": 1.25,
        "sectors": {"Energy": 0.02, "Industrials": -0.08, "Consumer Discretionary": -0.05},
    },
    "Other": {
        "equity": -0.02, "yield_bps": 0, "spread_bps": 25, "pd_multiplier": 1.1,
        "sectors": {},
    },
}


def _validate_positions(positions: list[StressPosition]) -> None:
    ids = [position.position_id.strip() for position in positions]

    if any(not position_id for position_id in ids):
        raise ValueError("Position IDs cannot be blank.")

    if len(ids) != len(set(ids)):
        raise ValueError("Position IDs must be unique.")

    for position in positions:
        if position.asset_type == "equity" and position.market_value <= 0:
            raise ValueError(f"{position.position_id}: equity market value must be positive.")

        if position.asset_type == "bond":
            if position.market_value <= 0:
                raise ValueError(f"{position.position_id}: bond market value must be positive.")
            if position.duration <= 0:
                raise ValueError(f"{position.position_id}: bond duration must be positive.")

        if position.asset_type == "loan":
            if position.ead <= 0:
                raise ValueError(f"{position.position_id}: loan EAD must be positive.")
            if position.market_value <= 0:
                raise ValueError(f"{position.position_id}: loan market value must be positive.")

        if position.asset_type == "rate_derivative" and position.dv01 < 0:
            raise ValueError(f"{position.position_id}: DV01 cannot be negative.")


def run_stress_test(
    positions: list[StressPosition],
    event_type: EventType,
    impact_score: int = 7,
) -> StressTestResponse:
    """Estimate scenario P&L using transparent, simplified valuation rules."""
    _validate_positions(positions)

    if not 1 <= impact_score <= 10:
        raise ValueError("Impact score must be between 1 and 10.")

    scenario = SCENARIOS[event_type]
    multiplier = impact_score / 5.0
    yield_shift_bps = scenario["yield_bps"] * multiplier
    spread_shift_bps = scenario["spread_bps"] * multiplier
    equity_shock = scenario["equity"] * multiplier
    stressed_pd_multiplier = 1.0 + (scenario["pd_multiplier"] - 1.0) * multiplier

    results: list[PositionStressResult] = []

    for position in positions:
        pnl = 0.0
        explanation = ""

        if position.asset_type == "equity":
            sector_shock = scenario["sectors"].get(position.sector, 0.0) * multiplier
            total_return = max(
                -1.0,
                min(1.0, (equity_shock + sector_shock) * position.equity_beta),
            )
            pnl = position.market_value * total_return
            explanation = (
                f"Equity return = market shock {equity_shock:.2%} + "
                f"sector adjustment {sector_shock:.2%}, scaled by beta "
                f"{position.equity_beta:.2f}."
            )

        elif position.asset_type == "bond":
            dy = yield_shift_bps / 10000.0
            ds = spread_shift_bps / 10000.0
            price_return = (
                -position.duration * dy
                + 0.5 * position.convexity * dy * dy
                - position.spread_duration * ds
            )
            price_return = max(-1.0, price_return)
            pnl = position.market_value * price_return
            explanation = (
                f"Duration/convexity return plus spread-duration adjustment; "
                f"yield shift {yield_shift_bps:.1f} bp, spread shift "
                f"{spread_shift_bps:.1f} bp."
            )

        elif position.asset_type == "loan":
            stressed_pd = min(
                1.0,
                position.default_pd * stressed_pd_multiplier,
            )
            incremental_el = (
                position.ead * position.lgd
                * max(0.0, stressed_pd - position.default_pd)
            )
            pnl = -incremental_el
            explanation = (
                f"Incremental expected credit loss using EAD {position.ead:,.2f}, "
                f"LGD {position.lgd:.1%}, and PD rising from "
                f"{position.default_pd:.2%} to {stressed_pd:.2%}."
            )

        elif position.asset_type == "rate_derivative":
            pnl = -position.dv01 * yield_shift_bps
            explanation = (
                f"DV01 approximation: -DV01 × yield shift "
                f"({position.dv01:,.2f} per bp × {yield_shift_bps:.1f} bp)."
            )

        elif position.asset_type == "cash":
            pnl = 0.0
            explanation = "Cash has no direct market P&L in this scenario model."

        results.append(
            PositionStressResult(
                position_id=position.position_id,
                name=position.name,
                asset_type=position.asset_type,
                market_value=position.market_value,
                pnl=round(pnl, 2),
                pnl_pct=(
                    round(pnl / position.market_value, 6)
                    if position.market_value > 0 else None
                ),
                loss_contribution=round(max(0.0, -pnl), 2),
                explanation=explanation,
            )
        )

    # Derivatives affect P&L, but their market value is excluded from NAV here.
    starting_nav = sum(
        position.market_value
        for position in positions
        if position.asset_type != "rate_derivative"
    )

    if starting_nav <= 0:
        raise ValueError("Portfolio NAV must be positive.")

    total_pnl = sum(position.pnl for position in results)
    net_loss = max(0.0, -total_pnl)
    gross_losses = sum(max(0.0, -position.pnl) for position in results)
    gross_gains = sum(max(0.0, position.pnl) for position in results)

    results.sort(key=lambda position: position.pnl)

    assumptions = {
        "base_equity_shock": scenario["equity"],
        "equity_shock_after_scaling": equity_shock,
        "government_yield_shift_bps": yield_shift_bps,
        "credit_spread_shift_bps": spread_shift_bps,
        "loan_pd_multiplier": stressed_pd_multiplier,
    }

    return StressTestResponse(
        event_type=event_type,
        impact_score=impact_score,
        shock_multiplier=round(multiplier, 4),
        starting_nav=round(starting_nav, 2),
        total_pnl=round(total_pnl, 2),
        net_loss=round(net_loss, 2),
        loss_pct=round(net_loss / starting_nav, 6),
        gross_losses=round(gross_losses, 2),
        gross_gains=round(gross_gains, 2),
        assumptions=assumptions,
        positions=results,
        summary=(
            "Scenario estimates use simplified assumptions and are not calibrated "
            "forecasts. Gross losses and gains are reported before offsetting; "
            "net loss is the negative part of total portfolio P&L."
        ),
    )
