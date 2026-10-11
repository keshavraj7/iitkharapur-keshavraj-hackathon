from fastapi import APIRouter, HTTPException

from app.schemas.rebalancing import (
    PortfolioPosition,
    RebalanceRequest,
    RebalanceResponse,
)
from app.services.rebalancer import calculate_rebalance
from app.storage.sqlite_store import list_signals


router = APIRouter(prefix="/rebalancer", tags=["Portfolio Rebalancer"])


def _execute(payload: RebalanceRequest) -> RebalanceResponse:
    # Prefer explicitly provided signals; otherwise use stored risk signals.
    signals = payload.signals or list_signals(limit=200)

    try:
        return calculate_rebalance(
            positions=payload.positions,
            signals=signals,
            sensitivity=payload.sensitivity,
            no_trade_band=payload.no_trade_band,
            max_turnover=payload.max_turnover,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/run", response_model=RebalanceResponse)
def run_rebalancer(payload: RebalanceRequest):
    """Calculate constrained portfolio weights from available risk signals."""
    return _execute(payload)


@router.get("/demo", response_model=RebalanceResponse)
def demo_rebalancer() -> RebalanceResponse:
    """Run the rebalancer on a deterministic synthetic portfolio."""
    positions = [
        PortfolioPosition(ticker="AAPL", company="Apple", sector="Technology", base_weight=0.20, current_weight=0.20),
        PortfolioPosition(ticker="MSFT", company="Microsoft", sector="Technology", base_weight=0.20, current_weight=0.20),
        PortfolioPosition(ticker="NVDA", company="NVIDIA", sector="Technology", base_weight=0.15, current_weight=0.15),
        PortfolioPosition(ticker="JPM", company="JPMorgan Chase", sector="Financials", base_weight=0.15, current_weight=0.15),
        PortfolioPosition(ticker="XOM", company="Exxon Mobil", sector="Energy", base_weight=0.10, current_weight=0.10),
        PortfolioPosition(ticker="PFE", company="Pfizer", sector="Healthcare", base_weight=0.10, current_weight=0.10),
        PortfolioPosition(ticker="COST", company="Costco", sector="Consumer Staples", base_weight=0.05, current_weight=0.05),
        PortfolioPosition(ticker="CASH", company="Cash", sector="Cash", base_weight=0.05, current_weight=0.05),
    ]

    return _execute(RebalanceRequest(positions=positions))
