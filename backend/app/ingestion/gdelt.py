import time
from typing import Any

import httpx


GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_NEWS_QUERY = (
    "(finance OR banking OR stocks OR inflation OR bankruptcy OR earnings)"
)

CACHE_TTL_SECONDS = 300


class GdeltClient:
    def __init__(self):
        self._cache: dict[
            tuple[str, str, int],
            tuple[float, list[dict[str, Any]]]
        ] = {}

    async def fetch_articles(
        self,
        query: str = DEFAULT_NEWS_QUERY,
        timespan: str = "24h",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        cache_key = (query.strip(), timespan, limit)
        now = time.monotonic()

        cached = self._cache.get(cache_key)

        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return list(cached[1])

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": limit,
            "timespan": timespan,
            "sort": "datedesc",
        }

        headers = {
            "User-Agent": "FinancialRiskIntelligence/0.1"
        }

        async with httpx.AsyncClient(
            timeout=20.0,
            headers=headers,
        ) as client:
            response = await client.get(
                GDELT_ENDPOINT,
                params=params,
            )
            response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"GDELT returned a non-JSON response: "
                f"{response.text[:200]}"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected GDELT response format")

        articles = payload.get("articles", [])

        if not isinstance(articles, list):
            raise RuntimeError("GDELT articles field is invalid")

        valid_articles = [
            article for article in articles
            if isinstance(article, dict)
        ]

        self._cache[cache_key] = (
            time.monotonic(),
            valid_articles,
        )

        return list(valid_articles)


gdelt_client = GdeltClient()