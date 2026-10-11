from fastapi import APIRouter, HTTPException, Query

from app.schemas.signal import EventType
from app.schemas.stress_test import (
    StressPosition,
    StressTestRequest,
    StressTestResponse,
)
from app.services.stress_tester import run_stress_test


router = APIRouter(prefix="/stress-test", tags=["Portfolio Stress Testing"])


def _run(request: StressTestRequest) -> StressTestResponse:
    try:
        return run_stress_test(
            positions=request.positions,
            event_type=request.event_type,
            impact_score=request.impact_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/run", response_model=StressTestResponse)
def stress_test_portfolio(request: StressTestRequest):
    """Run a stress scenario against a supplied portfolio."""
    return _run(request)


@router.get("/demo", response_model=StressTestResponse)
def stress_test_demo(
    event_type: EventType = "Credit Event",
    impact_score: int = Query(default=9, ge=1, le=10),
) -> StressTestResponse:
    """Run a reproducible scenario on a synthetic multi-asset portfolio."""
    positions = [
        StressPosition(
            position_id="EQ_TECH_01",
            name="Technology equities",
            asset_type="equity",
            sector="Technology",
            market_value=2_500_000,
            equity_beta=1.2,
        ),
        StressPosition(
            position_id="EQ_BANK_01",
            name="Bank equities",
            asset_type="equity",
            sector="Financials",
            market_value=1_500_000,
            equity_beta=1.1,
        ),
        StressPosition(
            position_id="EQ_ENERGY_01",
            name="Energy equities",
            asset_type="equity",
            sector="Energy",
            market_value=800_000,
            equity_beta=0.9,
        ),
        StressPosition(
            position_id="BOND_GOV_01",
            name="Government bond portfolio",
            asset_type="bond",
            sector="Government",
            market_value=2_000_000,
            duration=5.5,
            convexity=35,
        ),
        StressPosition(
            position_id="BOND_CORP_01",
            name="Investment-grade corporate bonds",
            asset_type="bond",
            sector="Financials",
            market_value=1_200_000,
            duration=4.2,
            convexity=25,
            spread_duration=3.8,
        ),
        StressPosition(
            position_id="LOAN_01",
            name="Commercial credit book",
            asset_type="loan",
            sector="Financials",
            market_value=1_000_000,
            ead=1_200_000,
            default_pd=0.025,
            lgd=0.45,
        ),
        StressPosition(
            position_id="RATE_DERIV_01",
            name="Interest-rate derivative",
            asset_type="rate_derivative",
            sector="Rates",
            market_value=0,
            dv01=350,
        ),
        StressPosition(
            position_id="CASH_01",
            name="Cash reserves",
            asset_type="cash",
            sector="Cash",
            market_value=1_000_000,
        ),
    ]

    request = StressTestRequest(
        positions=positions,
        event_type=event_type,
        impact_score=impact_score,
    )
    return _run(request)
