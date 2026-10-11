from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


EventType = Literal[
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "M&A",
    "Earnings",
    "Regulatory/Legal",
    "Product/Tech",
    "Cyber/Operational",
    "Supply Chain/Commodity",
    "Other",
]


class SourceInfo(BaseModel):
    type: Literal["news", "social", "replay", "manual"]
    name: str
    url: HttpUrl | None = None


class EntityMention(BaseModel):
    ticker: str
    company: str
    sector: str | None = None
    confidence: float = Field(ge=0, le=1)


class SentimentResult(BaseModel):
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)


class EventResult(BaseModel):
    type: EventType
    confidence: float = Field(ge=0, le=1)


class ImpactResult(BaseModel):
    score: int = Field(ge=1, le=10)
    horizon: Literal["1h", "1d", "1w"] = "1d"


class RiskSignal(BaseModel):
    signal_id: str
    event_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source: SourceInfo
    entities: list[EntityMention] = Field(default_factory=list)
    sentiment: SentimentResult
    event: EventResult
    impact: ImpactResult
    novelty: float = Field(ge=0, le=1)
    rationale: str
    model_versions: dict[str, str] = Field(default_factory=dict)