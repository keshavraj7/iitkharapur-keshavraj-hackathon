from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / "data/cache/fnspid/Stock_news/All_external.jsonl",
    ROOT / "data/cache/fnspid/Stock_news/nasdaq_exteral_data.jsonl",
]
OUTPUT = ROOT / "data/processed/historical_event_returns.parquet"
SUMMARY = ROOT / "data/processed/historical_event_returns_summary.json"

OUTPUT_SCHEMA = pa.schema([
    pa.field("event_timestamp_utc", pa.timestamp("ns", tz="UTC")),
    pa.field("source_dataset", pa.string()),
    pa.field("ticker", pa.string()),
    pa.field("headline", pa.string()),
    pa.field("article_text", pa.string()),
    pa.field("publisher", pa.string()),
    pa.field("url", pa.string()),
    pa.field("previous_session_date", pa.string()),
    pa.field("event_session_date", pa.string()),
    pa.field("next_session_date", pa.string()),
    pa.field("event_window_return_pct", pa.float64()),
    pa.field("event_window_abnormal_return_pct", pa.float64()),
    pa.field("forward_1d_return_pct", pa.float64()),
    pa.field("forward_1d_abnormal_return_pct", pa.float64()),
])


def clean_text(value, limit=1200):
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"", "n/a", "na", "nan", "none", "null"}:
        return ""
    return value[:limit]


def normalize_ticker(value):
    value = clean_text(value, 40).upper().replace(".", "-")
    return "" if value in {"", "N/A", "NONE"} else value


def load_prices():
    parquet_path = ROOT / "data/cache/market/adjusted_close.parquet"
    csv_path = ROOT / "data/cache/market/adjusted_close.csv"

    frame = None
    errors = []

    for path in (parquet_path, csv_path):
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            print(f"Loaded market prices: {path.relative_to(ROOT)}")
            break
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if frame is None:
        details = "\n".join(errors)
        raise FileNotFoundError(
            "Could not load adjusted_close.parquet or adjusted_close.csv.\n" + details
        )

    # Handle multi-level columns such as those produced by yfinance.
    if isinstance(frame.columns, pd.MultiIndex):
        flattened = False
        for level in range(frame.columns.nlevels):
            labels = {
                str(x).strip().lower(): x
                for x in frame.columns.get_level_values(level).unique()
            }
            selected = next(
                (labels[x] for x in ("adj close", "adjusted close", "close") if x in labels),
                None,
            )
            if selected is not None:
                frame = frame.xs(selected, axis=1, level=level, drop_level=True)
                flattened = True
                break
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                next((str(v) for v in col if str(v).lower() not in {
                    "adj close", "adjusted close", "close", "price"
                }), str(col[-1]))
                for col in frame.columns
            ]

    # Support either wide format (one column per ticker) or long format.
    lower = {str(c).strip().lower(): c for c in frame.columns}
    date_col = next(
        (lower[x] for x in ("date", "datetime", "timestamp") if x in lower),
        None,
    )

    ticker_col = next(
        (lower[x] for x in ("ticker", "symbol", "stock_symbol") if x in lower),
        None,
    )
    price_col = next(
        (lower[x] for x in ("adj close", "adjusted close", "close", "price") if x in lower),
        None,
    )

    if date_col is not None and ticker_col is not None and price_col is not None:
        dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        temp = pd.DataFrame({
            "date": dates,
            "ticker": frame[ticker_col].map(normalize_ticker),
            "price": pd.to_numeric(frame[price_col], errors="coerce"),
        }).dropna(subset=["date", "price"])
        temp = temp[temp["ticker"] != ""]
        frame = temp.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
    else:
        if date_col is not None:
            frame = frame.set_index(date_col)
        parsed_index = pd.to_datetime(frame.index, errors="coerce", utc=True)
        if parsed_index.isna().all():
            raise ValueError("Could not identify the date index in the market prices file.")
        frame.index = parsed_index.tz_convert(None).normalize()

        # Drop non-price columns and coerce ticker columns to numbers.
        frame.columns = [normalize_ticker(c) for c in frame.columns]
        frame = frame.loc[:, [c != "" for c in frame.columns]]
        frame = frame.apply(pd.to_numeric, errors="coerce")

    frame.index.name = "date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    # Resolve any ticker columns duplicated after normalization.
    if frame.columns.duplicated().any():
        frame = frame.T.groupby(level=0).first().T

    if "SPY" not in frame.columns:
        raise ValueError(
            "SPY is missing from adjusted prices. Available tickers: "
            + ", ".join(map(str, frame.columns[:40]))
        )

    print(f"Price coverage: {len(frame):,} sessions, {len(frame.columns):,} tickers")
    print(f"Price range: {frame.index.min().date()} to {frame.index.max().date()}")
    return frame


def build_market_lookup(prices):
    # Use dates with both the asset and SPY available so their returns align.
    lookup = {}
    for ticker in prices.columns:
        pair = prices[[ticker, "SPY"]].dropna()
        if len(pair) < 3:
            continue

        dates = pd.DatetimeIndex(pair.index)
        # Daily prices are dated by US session. Assume the official close is 4 PM New York.
        local_midnights = dates.tz_localize("America/New_York")
        close_utc = (local_midnights + pd.Timedelta(hours=16)).tz_convert("UTC")

        lookup[ticker] = {
            "dates": dates.to_numpy(dtype="datetime64[ns]"),
            "close_ns": close_utc.asi8,
            "asset": pair[ticker].to_numpy(dtype=np.float64),
            "spy": pair["SPY"].to_numpy(dtype=np.float64),
        }
    return lookup


def first_existing(columns, choices):
    lower = {str(c).strip().lower(): c for c in columns}
    return next((lower[c] for c in choices if c in lower), None)



def iter_jsonl_chunks(path: Path, chunk_size: int, stats: dict):
    """Stream JSONL records, skipping and logging malformed lines."""
    error_path = ROOT / "data/processed/fnspid_parse_errors.jsonl"
    batch = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle, \
         error_path.open("a", encoding="utf-8") as error_log:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise ValueError("JSON record is not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                stats["malformed_json_lines"] += 1
                json.dump({
                    "file": path.name,
                    "line": line_number,
                    "error": str(exc),
                    "sample": raw_line[:500],
                }, error_log, ensure_ascii=False)
                error_log.write("\n")
                continue

            batch.append(record)

            if len(batch) >= chunk_size:
                yield pd.DataFrame.from_records(batch)
                batch = []

        if batch:
            yield pd.DataFrame.from_records(batch)


def process_chunk(chunk, source_name, lookup, stats):
    stats["rows_read"] += len(chunk)

    date_col = first_existing(chunk.columns, ["date", "datetime", "timestamp"])
    ticker_col = first_existing(chunk.columns, ["stock_symbol", "ticker", "symbol"])
    headline_col = first_existing(chunk.columns, ["article_title", "headline", "title"])
    publisher_col = first_existing(chunk.columns, ["publisher", "source"])
    url_col = first_existing(chunk.columns, ["url", "link"])
    article_col = first_existing(chunk.columns, ["article", "lsa_summary", "textrank_summary", "luhn_summary", "lexrank_summary"])

    if date_col is None or ticker_col is None or headline_col is None:
        raise ValueError(
            f"{source_name}: missing required columns. Found: {list(chunk.columns)}"
        )

    work = pd.DataFrame(index=chunk.index)
    work["_ts"] = pd.to_datetime(chunk[date_col], errors="coerce", utc=True)
    work["_ticker"] = chunk[ticker_col].map(normalize_ticker)
    work["_headline"] = chunk[headline_col].map(lambda x: clean_text(x, 1000))
    work["_publisher"] = chunk[publisher_col].map(clean_text) if publisher_col else ""
    work["_url"] = chunk[url_col].map(lambda x: clean_text(x, 2000)) if url_col else ""
    work["_article"] = chunk[article_col].map(lambda x: clean_text(x, 1200)) if article_col else ""

    stats["invalid_timestamp_rows"] += int(work["_ts"].isna().sum())
    stats["missing_ticker_rows"] += int((work["_ticker"] == "").sum())

    outputs = []

    for ticker, group in work.groupby("_ticker", sort=False):
        if not ticker or ticker not in lookup:
            stats["unmatched_ticker_rows"] += len(group)
            continue

        group = group[group["_ts"].notna()].copy()
        if group.empty:
            continue

        market = lookup[ticker]
        timestamp_ns = group["_ts"].astype("int64").to_numpy()

        # First US market close strictly AFTER publication time.
        pos = np.searchsorted(market["close_ns"], timestamp_ns, side="right")
        valid = (pos > 0) & ((pos + 1) < len(market["dates"]))

        stats["rows_without_full_label"] += int((~valid).sum())
        if not valid.any():
            continue

        group = group.iloc[np.flatnonzero(valid)].copy()
        pos = pos[valid]

        asset = market["asset"]
        spy = market["spy"]
        prev_asset, event_asset, next_asset = asset[pos - 1], asset[pos], asset[pos + 1]
        prev_spy, event_spy, next_spy = spy[pos - 1], spy[pos], spy[pos + 1]

        event_ret = (event_asset / prev_asset - 1.0) * 100.0
        event_spy_ret = (event_spy / prev_spy - 1.0) * 100.0
        forward_ret = (next_asset / event_asset - 1.0) * 100.0
        forward_spy_ret = (next_spy / event_spy - 1.0) * 100.0

        market_dates = market["dates"]
        group["_event_timestamp_utc"] = group["_ts"]

        out = pd.DataFrame({
            "event_timestamp_utc": group["_event_timestamp_utc"].to_numpy(),
            "source_dataset": source_name,
            "ticker": ticker,
            "headline": group["_headline"].to_numpy(),
            "article_text": group["_article"].to_numpy(),
            "publisher": group["_publisher"].to_numpy(),
            "url": group["_url"].to_numpy(),
            "previous_session_date": market_dates[pos - 1].astype("datetime64[D]").astype(str),
            "event_session_date": market_dates[pos].astype("datetime64[D]").astype(str),
            "next_session_date": market_dates[pos + 1].astype("datetime64[D]").astype(str),
            "event_window_return_pct": event_ret,
            "event_window_abnormal_return_pct": event_ret - event_spy_ret,
            "forward_1d_return_pct": forward_ret,
            "forward_1d_abnormal_return_pct": forward_ret - forward_spy_ret,
        })
        outputs.append(out)
        stats["rows_written"] += len(out)

    if not outputs:
        return pd.DataFrame(columns=OUTPUT_SCHEMA.names)

    result = pd.concat(outputs, ignore_index=True)
    result["event_timestamp_utc"] = pd.to_datetime(result["event_timestamp_utc"], utc=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Build a leakage-aware historical news/market event dataset.")
    parser.add_argument("--chunk-size", type=int, default=25000)
    args = parser.parse_args()

    missing = [str(p) for p in INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    prices = load_prices()
    lookup = build_market_lookup(prices)
    if "SPY" not in lookup:
        raise ValueError("Insufficient overlapping SPY market-price history.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/processed/fnspid_parse_errors.jsonl").write_text("", encoding="utf-8")
    if OUTPUT.exists():
        OUTPUT.unlink()

    stats = {
        "malformed_json_lines": 0,
        "rows_read": 0,
        "rows_written": 0,
        "invalid_timestamp_rows": 0,
        "missing_ticker_rows": 0,
        "unmatched_ticker_rows": 0,
        "rows_without_full_label": 0,
    }
    source_stats = {}
    writer = None

    try:
        for path in INPUTS:
            print(f"\nProcessing: {path.name}")
            source_stats[path.name] = {"rows_read": 0, "rows_written": 0}
            before_read = stats["rows_read"]
            before_written = stats["rows_written"]

            for chunk_num, chunk in enumerate(
                iter_jsonl_chunks(path, args.chunk_size, stats), start=1
            ):
                result = process_chunk(chunk, path.stem, lookup, stats)
                source_stats[path.name]["rows_read"] += len(chunk)
                source_stats[path.name]["rows_written"] += len(result)

                if not result.empty:
                    table = pa.Table.from_pandas(
                        result, schema=OUTPUT_SCHEMA, preserve_index=False, safe=False
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(OUTPUT, OUTPUT_SCHEMA, compression="zstd")
                    writer.write_table(table)

                if chunk_num % 10 == 0:
                    print(
                        f"  chunks={chunk_num:,} | "
                        f"rows read={stats['rows_read']:,} | "
                        f"labeled events={stats['rows_written']:,}"
                    )

            print(
                f"  Finished {path.name}: "
                f"{stats['rows_read'] - before_read:,} rows read, "
                f"{stats['rows_written'] - before_written:,} labeled events"
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError(
            "No labeled events were generated. Check price ticker coverage and input schemas."
        )

    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs": [str(p.relative_to(ROOT)) for p in INPUTS],
        "market_prices": "data/cache/market/adjusted_close.parquet or .csv",
        "benchmark": "SPY",
        "assumed_market_close": "16:00 America/New_York",
        "label_definitions": {
            "event_window_return_pct": "Asset close return from the previous available session close to the first session close strictly after publication.",
            "event_window_abnormal_return_pct": "Event-window asset return minus SPY return over the same sessions.",
            "forward_1d_return_pct": "Asset close-to-close return from the event session to the following available session.",
            "forward_1d_abnormal_return_pct": "Forward asset return minus SPY return over the same sessions.",
        },
        "stats": stats,
        "per_source": source_stats,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_rows": stats["rows_written"],
        "limitations": [
            "Daily close timestamps are approximated as 16:00 America/New_York.",
            "Abnormal return is a simple SPY-relative return, not a causal estimate.",
            "Headlines may be duplicated across sources and need deduplication before model fitting.",
            "Ticker coverage is restricted to assets present in the downloaded adjusted-price table.",
        ],
    }

    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDONE")
    print(f"Malformed JSON lines skipped: {stats['malformed_json_lines']:,}")
    print(f"Rows scanned:          {stats['rows_read']:,}")
    print(f"Labeled events:        {stats['rows_written']:,}")
    print(f"Unmatched ticker rows: {stats['unmatched_ticker_rows']:,}")
    print(f"Invalid timestamps:    {stats['invalid_timestamp_rows']:,}")
    print(f"Missing full labels:   {stats['rows_without_full_label']:,}")
    print(f"Dataset: {OUTPUT}")
    print(f"Summary: {SUMMARY}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
