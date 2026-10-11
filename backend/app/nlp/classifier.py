import re

from app.schemas.signal import EntityMention, EventType


POSITIVE_TERMS = (
    "growth", "profit", "profits", "record", "beat", "beats",
    "strong", "surge", "upgrade", "upgraded", "recovery",
    "resilient", "positive", "gain", "gains", "improve",
    "improved", "robust", "exceed", "exceeds", "outperform",
    "approved", "approval", "expansion",
)

NEGATIVE_TERMS = (
    "loss", "losses", "decline", "declines", "fall", "falls",
    "drop", "drops", "plunge", "plunges", "layoff", "layoffs",
    "default", "bankruptcy", "fraud", "fine", "fined", "lawsuit",
    "investigation", "downgrade", "downgraded", "weak", "warning",
    "crisis", "crash", "recession", "inflation", "disruption",
    "breach", "attack", "sanctions", "shortfall", "debt",
    "missed payment", "insolvency",
)

EVENT_KEYWORDS: dict[EventType, tuple[str, ...]] = {
    "Geopolitical": (
        "war", "armed conflict", "sanctions", "invasion",
        "geopolitical", "military attack",
    ),
    "Macroeconomic": (
        "inflation", "interest rate", "central bank", "GDP",
        "recession", "unemployment", "monetary policy",
        "consumer price index", "CPI",
    ),
    "Credit Event": (
        "bankruptcy", "default", "debt restructuring",
        "missed payment", "insolvency", "credit downgrade",
        "rating downgrade", "debt distress", "liquidity crisis",
        "bond default",
    ),
    "M&A": (
        "merger", "acquisition", "acquire", "takeover",
        "buyout", "merges with",
    ),
    "Earnings": (
        "earnings", "revenue", "quarterly results", "EPS",
        "profit", "profit warning", "financial results",
        "earnings guidance",
    ),
    "Regulatory/Legal": (
        "lawsuit", "regulator", "antitrust", "investigation",
        "SEC", "court ruling", "fine", "regulatory action",
        "compliance violation",
    ),
    "Product/Tech": (
        "product launch", "artificial intelligence", "AI model",
        "chip", "software", "patent", "cloud computing",
        "technology launch",
    ),
    "Cyber/Operational": (
        "cyberattack", "data breach", "ransomware",
        "service outage", "operational disruption",
        "data leak", "system outage",
    ),
    "Supply Chain/Commodity": (
        "supply chain", "oil prices", "natural gas", "copper",
        "commodity prices", "shipping disruption", "OPEC",
        "raw material shortage", "commodity shortage",
    ),
    "Other": (),
}

# Ticker: (company name, sector, recognized name aliases)
COMPANIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "AAPL": ("Apple Inc.", "Technology", ("apple inc", "apple")),
    "MSFT": ("Microsoft", "Technology", ("microsoft",)),
    "NVDA": ("NVIDIA", "Technology", ("nvidia",)),
    "TSLA": ("Tesla", "Automotive", ("tesla",)),
    "AMZN": ("Amazon", "Consumer / Technology", ("amazon",)),
    "GOOGL": ("Alphabet", "Technology", ("alphabet", "google")),
    "META": ("Meta Platforms", "Technology", ("meta platforms",)),
    "JPM": ("JPMorgan Chase", "Financials", ("jpmorgan chase", "jpmorgan")),
    "BAC": ("Bank of America", "Financials", ("bank of america",)),
    "GS": ("Goldman Sachs", "Financials", ("goldman sachs",)),
    "MS": ("Morgan Stanley", "Financials", ("morgan stanley",)),
    "WFC": ("Wells Fargo", "Financials", ("wells fargo",)),
    "XOM": ("Exxon Mobil", "Energy", ("exxon mobil", "exxon")),
    "CVX": ("Chevron", "Energy", ("chevron",)),
    "WMT": ("Walmart", "Consumer Staples", ("walmart",)),
    "COST": ("Costco", "Consumer Staples", ("costco",)),
    "KO": ("Coca-Cola", "Consumer Staples", ("coca-cola",)),
    "PFE": ("Pfizer", "Healthcare", ("pfizer",)),
}


def _matches(text: str, terms: tuple[str, ...]) -> list[str]:
    """Return distinct vocabulary terms found using case-insensitive matching."""
    normalized = text.casefold()
    return [
        term
        for term in terms
        if re.search(
            rf"(?<!\w){re.escape(term.casefold())}(?!\w)",
            normalized,
        )
    ]


def analyze_sentiment(
    text: str,
) -> tuple[float, float, list[str], list[str]]:
    """Return heuristic score, confidence, positive matches and negative matches."""
    positive = _matches(text, POSITIVE_TERMS)
    negative = _matches(text, NEGATIVE_TERMS)
    evidence_count = len(positive) + len(negative)

    if evidence_count == 0:
        return 0.0, 0.35, positive, negative

    score = (len(positive) - len(negative)) / (evidence_count + 1)
    confidence = min(0.90, 0.55 + 0.10 * (evidence_count - 1))

    return round(max(-1.0, min(1.0, score)), 3), confidence, positive, negative


def classify_event(text: str) -> tuple[EventType, float, list[str]]:
    """Choose the event category with the most matching keywords."""
    candidates = []

    for event_type, terms in EVENT_KEYWORDS.items():
        if event_type == "Other":
            continue

        matches = _matches(text, terms)

        if matches:
            candidates.append((event_type, matches))

    if not candidates:
        return "Other", 0.40, []

    event_type, matches = max(
        candidates,
        key=lambda candidate: len(candidate[1]),
    )
    confidence = min(0.90, 0.60 + 0.10 * (len(matches) - 1))

    return event_type, confidence, matches


def extract_entities(text: str) -> list[EntityMention]:
    """Recognize a small, explicit company alias and ticker dictionary."""
    entities = []

    for ticker, (company, sector, aliases) in COMPANIES.items():
        # Explicit uppercase tickers or dollar-prefixed tickers only.
        ticker_pattern = (
            rf"(?<![A-Za-z0-9])(?:\${ticker}|{ticker})(?![A-Za-z0-9])"
        )
        explicit_ticker = re.search(ticker_pattern, text) is not None
        named_company = bool(_matches(text, aliases))

        if explicit_ticker or named_company:
            entities.append(
                EntityMention(
                    ticker=ticker,
                    company=company,
                    sector=sector,
                    confidence=0.95 if explicit_ticker else 0.80,
                )
            )

    return entities
