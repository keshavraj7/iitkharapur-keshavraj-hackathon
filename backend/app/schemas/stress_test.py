from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.signal import EventType


AssetType = Literal["equity", "bond", "loan", "rate_derivative", "cash"]


class StressPosition(BaseModel):
    position_id: str
    name: str
    asset_type: AssetType
    sector: str = "Other"

    # Market value is denominated in the portfolio's base currency.
    # A rate derivative may use market_value=0 when only DV01 exposure is modelled.
    market_value: float = Field(ge=0)

    equity_beta: float = Field(default=1.0, ge=-3.0, le=3.0)

    # Bond risk measures
    duration: float = Field(default=0.0, ge=0, le=50)
    convexity: float = Field(default=0.0, ge=0, le=200)
    spread_duration: float = Field(default=0.0, ge=0, le=50)

    # Loan credit-risk measures
    ead: float = Field(default=0.0, ge=0)
    default_pd: float = Field(default=0.0, ge=0, le=1)
    lgd: float = Field(default=0.0, ge=0, le=1)

    # Dollar P&L sensitivity per 1 basis-point rate move
    dv01: float = Field(default=0.0, ge=0)


class StressTestRequest(BaseModel):
    positions: list[StressPosition] = Field(min_length=1, max_length=1000)
    event_type: EventType
    impact_score: int = Field(default=7, ge=1, le=10)


class PositionStressResult(BaseModel):
    position_id: str
    name: str
    asset_type: AssetType
    market_value: float
    pnl: float
    pnl_pct: float | None
    loss_contribution: float
    explanation: str


class StressTestResponse(BaseModel):
    event_type: EventType
    impact_score: int
    shock_multiplier: float
    starting_nav: float
    total_pnl: float
    net_loss: float
    loss_pct: float
    gross_losses: float
    gross_gains: float
    assumptions: dict[str, float]
    positions: list[PositionStressResult]
    summary: str
