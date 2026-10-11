import hashlib
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.schemas.document import RawDocument


def parse_gdelt_seen_date(value: Any) -> datetime | None:
    if not value:
        return None

    date_text = str(value).strip()

    formats = [
        "%Y%m%dT%H%M%SZ",
        "%Y%m%dT%H%M%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]

    for date_format in formats:
        try:
            parsed = datetime.strptime(date_text, date_format)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(
            date_text.replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def normalize_gdelt_article(
    article: dict[str, Any],
) -> RawDocument | None:
    title = str(article.get("title") or "").strip()
    url = str(article.get("url") or "").strip() or None

    if not title:
        return None

    domain = str(article.get("domain") or "").strip()

    if not domain and url:
        domain = urlparse(url).netloc

    source_name = domain or "GDELT"

    identifier = url or f"{source_name}:{title.casefold()}"
    document_hash = hashlib.sha256(
        identifier.encode("utf-8")
    ).hexdigest()[:16]

    language = str(article.get("language") or "").strip() or None
    source_country = (
        str(article.get("sourcecountry") or "").strip() or None
    )

    tone = article.get("tone")

    try:
        tone = float(tone) if tone is not None else None
    except (TypeError, ValueError):
        tone = None

    return RawDocument(
        document_id=f"doc_{document_hash}",
        source_type="news",
        source_name=source_name,
        url=url,
        title=title,
        text=title,
        observed_at=parse_gdelt_seen_date(
            article.get("seendate")
        ),
        language=language,
        source_country=source_country,
        metadata={
            "provider": "GDELT DOC API",
            "timestamp_kind": "gdelt_seen_time",
            "gdelt_tone": tone,
            "social_image": article.get("socialimage"),
        },
    )