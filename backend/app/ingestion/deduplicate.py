import re
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

from app.schemas.document import RawDocument


TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "ref",
    "referrer",
    "cmpid",
}


def canonicalize_url(url: str | None) -> str | None:
    if not url:
        return None

    try:
        parts = urlsplit(url)

        if not parts.scheme or not parts.netloc:
            return url.strip().casefold()

        query_params = [
            (key, value)
            for key, value in parse_qsl(
                parts.query,
                keep_blank_values=True,
            )
            if not key.casefold().startswith("utm_")
            and key.casefold() not in TRACKING_PARAMS
        ]

        return urlunsplit((
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/") or "/",
            urlencode(sorted(query_params)),
            "",
        ))
    except ValueError:
        return url.strip().casefold()


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().casefold()


def deduplicate_documents(
    documents: list[RawDocument],
) -> tuple[list[RawDocument], int]:
    unique_documents = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for document in documents:
        canonical_url = canonicalize_url(document.url)
        normalized_title = normalize_title(document.title)

        duplicate_url = (
            canonical_url is not None
            and canonical_url in seen_urls
        )

        duplicate_title = (
            bool(normalized_title)
            and normalized_title in seen_titles
        )

        if duplicate_url or duplicate_title:
            continue

        if canonical_url:
            seen_urls.add(canonical_url)

        if normalized_title:
            seen_titles.add(normalized_title)

        unique_documents.append(document)

    removed_count = len(documents) - len(unique_documents)

    return unique_documents, removed_count