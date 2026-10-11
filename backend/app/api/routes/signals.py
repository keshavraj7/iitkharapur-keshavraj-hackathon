from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.ingestion.deduplicate import deduplicate_documents
from app.ingestion.gdelt import DEFAULT_NEWS_QUERY, gdelt_client
from app.ingestion.normalize import normalize_gdelt_article
from app.schemas.document import RawDocument
from app.schemas.signal import RiskSignal
from app.services.signal_service import generate_risk_signal
from app.storage.sqlite_store import (
    get_signal,
    insert_signal,
    list_signals,
)


router = APIRouter(prefix="/signals", tags=["Signals"])


class AnalyzeDocumentsRequest(BaseModel):
    documents: list[RawDocument] = Field(min_length=1, max_length=100)


def _process_documents(
    documents: list[RawDocument],
) -> list[RiskSignal]:
    generated_signals = []

    for document in documents:
        generated = generate_risk_signal(document)
        insert_signal(generated)

        stored = get_signal(generated.signal_id)
        generated_signals.append(stored or generated)

    return generated_signals


@router.post("", response_model=RiskSignal, status_code=201)
def create_signal(signal: RiskSignal):
    inserted = insert_signal(signal)

    if not inserted:
        raise HTTPException(
            status_code=409,
            detail="Signal already exists",
        )

    return get_signal(signal.signal_id)


@router.post("/analyze", response_model=list[RiskSignal])
def analyze_documents(payload: AnalyzeDocumentsRequest):
    """Analyze supplied documents and persist the generated signals."""
    return _process_documents(payload.documents)


@router.get(
    "/analyze-news",
    response_model=list[RiskSignal],
)
async def analyze_news(
    query: str = Query(
        default=DEFAULT_NEWS_QUERY,
        min_length=3,
        max_length=300,
    ),
    timespan: Literal[
        "1h", "3h", "6h", "12h", "24h", "48h", "7d"
    ] = "24h",
    limit: int = Query(default=20, ge=1, le=50),
):
    """Fetch GDELT articles and convert them into persistent risk signals."""
    raw_articles = await gdelt_client.fetch_articles(
        query=query,
        timespan=timespan,
        limit=limit,
    )

    documents = []

    for article in raw_articles:
        document = normalize_gdelt_article(article)

        if document is not None:
            documents.append(document)

    unique_documents, _ = deduplicate_documents(documents)

    return _process_documents(unique_documents)


@router.get("", response_model=list[RiskSignal])
def get_signals(limit: int = Query(default=50, ge=1, le=200)):
    return list_signals(limit)


@router.get("/{signal_id}", response_model=RiskSignal)
def get_signal_by_id(signal_id: str):
    signal = get_signal(signal_id)

    if signal is None:
        raise HTTPException(
            status_code=404,
            detail="Signal not found",
        )

    return signal
