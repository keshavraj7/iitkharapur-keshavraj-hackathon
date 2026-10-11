from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    document_id: str
    source_type: Literal["news", "social", "replay", "manual"]
    source_name: str
    url: str | None = None
    title: str
    text: str
    observed_at: datetime | None = None
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    language: str | None = None
    source_country: str | None = None
    ticker_candidates: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    source: str
    query: str
    timespan: str
    fetched_count: int
    normalized_count: int
    duplicates_removed: int
    documents: list[RawDocument]