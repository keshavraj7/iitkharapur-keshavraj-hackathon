import pytest

from app.schemas.stress_test import StressPosition
from app.services.stress_tester import run_stress_test


def sample_positions():
    return [
        StressPosition(
            position_id="EQ1",
            name="Technology equities",
            asset_type="equity",
            sector="Technology",
            market_value=1_000_000,
            equity_beta=1.0,
        ),
        StressPosition(
            position_id="BOND1",
            name="Government bonds",
            asset_type="bond",
            sector="Government",
            market_value=500_000,
            duration=5.0,
            convexity=30.0,
        ),
        StressPosition(
            position_id="LOAN1",
            name="Corporate loan",
            asset_type="loan",
            sector="Financials",
            market_value=400_000,
            ead=500_000,
            default_pd=0.02,
            lgd=0.45,
        ),
        StressPosition(
            position_id="RATE1",
            name="Rate swap",
            asset_type="rate_derivative",
            market_value=0,
            dv01=100,
        ),
        StressPosition(
            position_id="CASH1",
            name="Cash",
            asset_type="cash",
            market_value=100_000,
        ),
    ]


def test_credit_stress_returns_position_breakdown():
    result = run_stress_test(
        sample_positions(),
        event_type="Credit Event",
        impact_score=9,
    )

    assert result.starting_nav == pytest.approx(2_000_000)
    assert result.total_pnl < 0
    assert result.net_loss > 0
    assert result.loss_pct > 0
    assert len(result.positions) == 5
    assert sum(p.pnl for p in result.positions) == pytest.approx(result.total_pnl)


def test_higher_impact_scales_the_equity_shock():
    positions = [
        StressPosition(
            position_id="EQ1",
            name="Broad equity portfolio",
            asset_type="equity",
            sector="Technology",
            market_value=1_000_000,
        )
    ]

    moderate = run_stress_test(positions, "Macroeconomic", impact_score=5)
    severe = run_stress_test(positions, "Macroeconomic", impact_score=10)

    assert severe.positions[0].pnl < moderate.positions[0].pnl
    assert severe.shock_multiplier == 2.0


def test_cash_has_no_direct_market_loss():
    positions = [
        StressPosition(
            position_id="CASH1",
            name="Cash",
            asset_type="cash",
            market_value=100_000,
        )
    ]

    result = run_stress_test(positions, "Geopolitical", impact_score=8)

    assert result.positions[0].pnl == 0
    assert result.starting_nav == 100_000


def test_duplicate_position_ids_are_rejected():
    positions = sample_positions()
    positions[1] = positions[1].model_copy(
        update={"position_id": positions[0].position_id}
    )

    with pytest.raises(ValueError, match="Position IDs must be unique"):
        run_stress_test(positions, "Credit Event", impact_score=9)
