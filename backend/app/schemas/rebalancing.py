from pydantic import BaseModel, Field
from app.schemas.signal import RiskSignal


class PortfolioPosition(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    company: str
    sector: str = "Unknown"
    base_weight: float = Field(ge=0, le=1)
    current_weight: float = Field(ge=0, le=1)


class RebalanceRequest(BaseModel):
    positions: list[PortfolioPosition] = Field(min_length=1, max_length=100)
    signals: list[RiskSignal] = Field(default_factory=list)
    sensitivity: float = Field(default=0.75, ge=0, le=2)
    no_trade_band: float = Field(default=0.005, ge=0, le=0.05)
    max_turnover: float = Field(default=0.10, ge=0, le=1)


class PositionRecommendation(BaseModel):
    ticker: str
    company: str
    sector: str
    base_weight: float
    current_weight: float
    target_weight: float
    proposed_weight: float
    change: float
    signal_score: float
    event_override: bool
    rationale: str


class RebalanceResponse(BaseModel):
    positions: list[PositionRecommendation]
    signals_used: int
    one_way_turnover: float
    mandatory_turnover: float
    discretionary_turnover: float
    max_turnover: float
    turnover_cap_applied: bool
    summary: str
