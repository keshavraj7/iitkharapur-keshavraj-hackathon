import logging
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.ingestion.deduplicate import deduplicate_documents
from app.ingestion.gdelt import (
    DEFAULT_NEWS_QUERY,
    gdelt_client,
)
from app.ingestion.normalize import normalize_gdelt_article
from app.schemas.document import IngestionResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.get("/news", response_model=IngestionResponse)
async def ingest_news(
    query: str = Query(
        default=DEFAULT_NEWS_QUERY,
        min_length=3,
        max_length=300,
    ),
    timespan: Literal[
        "1h", "3h", "6h", "12h", "24h", "48h", "7d"
    ] = Query(default="24h"),
    limit: int = Query(default=20, ge=1, le=50),
):
    try:
        raw_articles = await gdelt_client.fetch_articles(
            query=query,
            timespan=timespan,
            limit=limit,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        logger.exception("GDELT news ingestion failed")

        raise HTTPException(
            status_code=502,
            detail=f"News provider request failed: {exc}",
        ) from exc

    normalized_documents = []

    for article in raw_articles:
        document = normalize_gdelt_article(article)

        if document is not None:
            normalized_documents.append(document)

    unique_documents, duplicates_removed = (
        deduplicate_documents(normalized_documents)
    )

    return IngestionResponse(
        source="GDELT DOC API",
        query=query,
        timespan=timespan,
        fetched_count=len(raw_articles),
        normalized_count=len(normalized_documents),
        duplicates_removed=duplicates_removed,
        documents=unique_documents,
    )