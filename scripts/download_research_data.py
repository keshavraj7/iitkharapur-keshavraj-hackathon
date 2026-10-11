import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from datasets import load_dataset
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "raw" / "benchmarks"
CACHE_DIR = ROOT / "data" / "cache"
HF_CACHE = CACHE_DIR / "huggingface"

SENTIMENT_LABELS = {
    0: "bearish",
    1: "bullish",
    2: "neutral",
}

TOPIC_LABELS = {
    0: "Analyst Update",
    1: "Fed | Central Banks",
    2: "Company | Product News",
    3: "Treasuries | Corporate Debt",
    4: "Dividend",
    5: "Earnings",
    6: "Energy | Oil",
    7: "Financials",
    8: "Currencies",
    9: "General News | Opinion",
    10: "Gold | Metals | Materials",
    11: "IPO",
    12: "Legal | Regulation",
    13: "M&A | Investments",
    14: "Macro",
    15: "Markets",
    16: "Politics",
    17: "Personnel Change",
    18: "Stock Commentary",
    19: "Stock Movement",
}

HF_DATASETS = [
    (
        "zeroshot/twitter-financial-news-sentiment",
        "twitter_sentiment",
        SENTIMENT_LABELS,
        "MIT",
    ),
    (
        "zeroshot/twitter-financial-news-topic",
        "twitter_topic",
        TOPIC_LABELS,
        "MIT",
    ),
]

ISHARES_HOLDINGS = {
    "LQD": (
        "https://www.ishares.com/us/products/239566/"
        "ishares-iboxx-investment-grade-corporate-bond-etf/"
        "latest-holdings.csv"
    ),
    "HYG": (
        "https://www.ishares.com/us/products/239565/"
        "hyg-ishares-iboxx-high-yield-corporate-bond-etf/"
        "latest-holdings.csv"
    ),
}


def create_directories() -> None:
    for directory in (
        BENCHMARK_DIR,
        CACHE_DIR,
        HF_CACHE,
        CACHE_DIR / "external",
        CACHE_DIR / "market",
        CACHE_DIR / "ishares",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def label_id(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.fullmatch(r"LABEL_(\d+)", str(value))
        return int(match.group(1)) if match else None


def download_hf_benchmark(
    repo_id: str,
    prefix: str,
    label_mapping: dict[int, str],
    license_name: str,
) -> dict:
    print(f"\n[Hugging Face] {repo_id}")

    dataset = load_dataset(
        repo_id,
        cache_dir=str(HF_CACHE),
    )

    manifest_entry = {
        "repo_id": repo_id,
        "license": license_name,
        "files": [],
    }

    for split_name, split_data in dataset.items():
        frame = split_data.to_pandas()

        text_column = next(
            (name for name in ("text", "sentence", "tweet") if name in frame.columns),
            None,
        )
        label_column = next(
            (name for name in ("label", "labels") if name in frame.columns),
            None,
        )

        if text_column is None or label_column is None:
            raise ValueError(
                f"Unexpected dataset columns for {repo_id}: "
                f"{list(frame.columns)}"
            )

        result = pd.DataFrame({
            "text": frame[text_column].fillna("").astype(str),
            "label_id": frame[label_column].map(label_id),
        })

        result["label_name"] = result["label_id"].map(label_mapping)
        result["split"] = split_name
        result = result[result["text"].str.strip().ne("")].reset_index(drop=True)

        if result["label_name"].isna().any():
            unknown = sorted(
                result.loc[result["label_name"].isna(), "label_id"]
                .dropna()
                .unique()
                .tolist()
            )
            raise ValueError(f"Unknown labels for {repo_id}: {unknown}")

        output_path = BENCHMARK_DIR / f"{prefix}_{split_name}.csv"
        result.to_csv(output_path, index=False, encoding="utf-8")

        print(f"  {split_name}: {len(result):,} rows -> {output_path.relative_to(ROOT)}")

        manifest_entry["files"].append({
            "path": str(output_path.relative_to(ROOT)),
            "split": split_name,
            "rows": len(result),
        })

    return manifest_entry


def download_phrasebank() -> dict:
    print("\n[Hugging Face] Financial PhraseBank — local evaluation copy")

    dataset = load_dataset(
        "takala/financial_phrasebank",
        "sentences_75agree",
        cache_dir=str(HF_CACHE),
        trust_remote_code=True,
    )["train"].to_pandas()

    if "sentence" not in dataset.columns or "label" not in dataset.columns:
        raise ValueError(
            f"Unexpected PhraseBank columns: {list(dataset.columns)}"
        )

    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

    result = pd.DataFrame({
        "text": dataset["sentence"].fillna("").astype(str),
        "label_id": dataset["label"].map(label_id),
    })
    result["label_name"] = result["label_id"].map(label_mapping)
    result = result[result["text"].str.strip().ne("")].reset_index(drop=True)

    if result["label_name"].isna().any():
        raise ValueError("Unexpected Financial PhraseBank labels.")

    output_path = CACHE_DIR / "external" / "financial_phrasebank_75agree.csv"
    result.to_csv(output_path, index=False, encoding="utf-8")

    print(f"  {len(result):,} rows -> {output_path.relative_to(ROOT)}")
    print("  Local cache only; check dataset terms before redistribution.")

    return {
        "repo_id": "takala/financial_phrasebank",
        "configuration": "sentences_75agree",
        "license": "CC BY-NC-SA 3.0",
        "path": str(output_path.relative_to(ROOT)),
        "rows": len(result),
        "redistribute": False,
    }


def download_market_prices(start_date: str) -> dict:
    print("\n[yfinance] Downloading daily adjusted closing prices")

    ticker_path = ROOT / "data" / "config" / "market_tickers.txt"
    if not ticker_path.exists():
        raise FileNotFoundError(f"Missing ticker configuration: {ticker_path}")

    tickers = list(dict.fromkeys(
        line.strip()
        for line in ticker_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))

    if not tickers:
        raise ValueError("The market ticker configuration is empty.")

    end_date = (
        datetime.now(timezone.utc).date() + timedelta(days=1)
    ).isoformat()

    price_series = {}
    failures = []

    for offset in range(0, len(tickers), 8):
        batch = tickers[offset:offset + 8]
        print(f"  Requesting {', '.join(batch)}")

        raw = yf.download(
            tickers=batch,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            multi_level_index=True,
            progress=False,
            threads=False,
            timeout=30,
        )

        if raw is None or raw.empty:
            failures.extend(batch)
            print("    No data returned for this batch.")
            continue

        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    first_level = raw.columns.get_level_values(0)
                    second_level = raw.columns.get_level_values(1)

                    if ticker in first_level:
                        one_ticker = raw[ticker]
                    elif ticker in second_level:
                        one_ticker = raw.xs(ticker, axis=1, level=1)
                    else:
                        failures.append(ticker)
                        continue
                else:
                    if len(batch) != 1:
                        failures.append(ticker)
                        continue
                    one_ticker = raw

                close_column = (
                    "Close" if "Close" in one_ticker.columns
                    else "Adj Close" if "Adj Close" in one_ticker.columns
                    else None
                )

                if close_column is None:
                    failures.append(ticker)
                    continue

                series = pd.to_numeric(
                    one_ticker[close_column],
                    errors="coerce",
                ).dropna()

                if series.empty:
                    failures.append(ticker)
                    continue

                series.name = ticker
                price_series[ticker] = series

            except (KeyError, TypeError, ValueError) as exc:
                failures.append(ticker)
                print(f"    Could not process {ticker}: {exc}")

    if not price_series:
        raise RuntimeError("No adjusted closing-price data was retrieved.")

    prices = pd.concat(price_series.values(), axis=1).sort_index()
    prices.index.name = "date"

    parquet_path = CACHE_DIR / "market" / "adjusted_close.parquet"
    csv_path = CACHE_DIR / "market" / "adjusted_close.csv"

    prices.to_parquet(parquet_path)
    prices.to_csv(csv_path, encoding="utf-8")

    metadata = {
        "source": "Yahoo Finance through yfinance",
        "price_basis": "auto_adjust=True; daily adjusted OHLC close field",
        "start_date": start_date,
        "end_date_exclusive": end_date,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers_requested": tickers,
        "tickers_downloaded": list(prices.columns),
        "tickers_failed": sorted(set(failures)),
        "rows": len(prices),
    }

    metadata_path = CACHE_DIR / "market" / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"  {len(prices):,} dates, {len(prices.columns)} tickers")
    print(f"  Parquet: {parquet_path.relative_to(ROOT)}")
    print(f"  CSV:     {csv_path.relative_to(ROOT)}")

    if failures:
        print(f"  Tickers with missing data: {', '.join(sorted(set(failures)))}")

    return metadata


def download_ishares_holdings() -> dict:
    print("\n[iShares] Downloading LQD and HYG holdings")

    output = CACHE_DIR / "ishares"
    metadata = {"source": "iShares official holdings exports", "files": []}

    for symbol, url in ISHARES_HOLDINGS.items():
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "FinancialRiskIntelligence/0.1"},
                timeout=60,
            )
            response.raise_for_status()

            content = response.content
            if b"Name,Sector,Asset Class" not in content:
                raise ValueError("Response did not contain the expected holdings CSV.")

            path = output / f"{symbol.lower()}_holdings.csv"
            path.write_bytes(content)

            # The export has metadata lines before its tabular header.
            holdings = pd.read_csv(path, skiprows=8)
            required = {"Name", "Sector", "Asset Class", "Mod. Duration", "Maturity"}

            if not required.issubset(set(holdings.columns)):
                raise ValueError(
                    f"Unexpected {symbol} columns: {list(holdings.columns)}"
                )

            print(f"  {symbol}: {len(holdings):,} rows")
            metadata["files"].append({
                "symbol": symbol,
                "path": str(path.relative_to(ROOT)),
                "rows": len(holdings),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as exc:
            print(f"  WARNING: {symbol} holdings download failed: {exc}")

    return metadata


def download_fnspid() -> dict:
    print("\n[FNSPID] Opt-in download: approximately 2.34 GB")

    output = CACHE_DIR / "fnspid"
    output.mkdir(parents=True, exist_ok=True)

    files = [
        "Stock_news/All_external.jsonl",
        "Stock_news/nasdaq_exteral_data.jsonl",
    ]

    result = {"source": "khaihernlow/fnspid", "license": "CC BY-NC 4.0", "files": []}

    for filename in files:
        local_path = hf_hub_download(
            repo_id="khaihernlow/fnspid",
            repo_type="dataset",
            filename=filename,
            local_dir=str(output),
        )
        local_file = Path(local_path)

        result["files"].append({
            "path": str(local_file.relative_to(ROOT)),
            "bytes": local_file.stat().st_size,
        })
        print(f"  Downloaded {filename}")

    return result


def download_sba() -> dict:
    print("\n[SBA] Downloading official public 7(a) CSV resources")

    api = "https://web.data.sba.gov/api/3/action/package_show?id=7-a-504-foia"
    response = requests.get(api, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if not payload.get("success"):
        raise RuntimeError("SBA open-data API did not return success.")

    resources = payload["result"].get("resources", [])
    selected = [
        item for item in resources
        if str(item.get("format", "")).lower() == "csv"
        and (
            "7(a)" in str(item.get("name", "")).lower()
            or "7a" in str(item.get("name", "")).lower()
        )
    ]

    if not selected:
        raise RuntimeError("No 7(a) CSV resources found in SBA catalog response.")

    output = CACHE_DIR / "sba"
    output.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for item in selected:
        filename = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            str(item.get("name") or item.get("id") or "sba_7a.csv"),
        )
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        url = item["url"]
        target = output / filename

        file_response = requests.get(url, stream=True, timeout=(20, 180))
        file_response.raise_for_status()

        with target.open("wb") as file:
            for chunk in file_response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

        downloaded.append({
            "name": filename,
            "url": url,
            "bytes": target.stat().st_size,
        })
        print(f"  Downloaded {filename}")

    return {
        "source": "U.S. SBA Open Data 7(a) & 504 FOIA",
        "license": "Public domain as listed by the source portal",
        "files": downloaded,
    }


def download_kaggle_tweets() -> dict:
    print("\n[Kaggle] Downloading stock tweet replay dataset")

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI was not found. Install it, configure Kaggle API "
            "credentials, then rerun with --include-kaggle-tweets."
        )

    output = CACHE_DIR / "kaggle_tweets"
    output.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "equinxx/stock-tweets-for-sentiment-analysis-and-prediction",
            "-p", str(output),
            "--unzip",
        ],
        check=True,
    )

    files = [
        str(path.relative_to(ROOT))
        for path in output.rglob("*")
        if path.is_file()
    ]

    return {
        "source": "https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction",
        "files": files,
        "note": "Replay/backtest input; not a sentiment-labelled evaluation set.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download/cache Financial Risk Intelligence research data."
    )
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-phrasebank", action="store_true")
    parser.add_argument("--include-fnspid", action="store_true")
    parser.add_argument("--include-sba", action="store_true")
    parser.add_argument("--include-kaggle-tweets", action="store_true")
    args = parser.parse_args()

    create_directories()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
        "market_prices": None,
        "bond_holdings": None,
        "optional_sources": [],
        "errors": [],
    }

    steps = []

    for repo_id, prefix, mapping, license_name in HF_DATASETS:
        steps.append((
            f"benchmark {prefix}",
            lambda r=repo_id, p=prefix, m=mapping, l=license_name:
                download_hf_benchmark(r, p, m, l),
            "datasets",
        ))

    if not args.skip_phrasebank:
        steps.append(("Financial PhraseBank", download_phrasebank, "datasets"))

    if not args.skip_prices:
        steps.append((
            "market prices",
            lambda: download_market_prices(args.start_date),
            "market_prices",
        ))

    steps.append(("iShares holdings", download_ishares_holdings, "bond_holdings"))

    if args.include_fnspid:
        steps.append(("FNSPID", download_fnspid, "optional_sources"))
    if args.include_sba:
        steps.append(("SBA", download_sba, "optional_sources"))
    if args.include_kaggle_tweets:
        steps.append(("Kaggle tweet replay", download_kaggle_tweets, "optional_sources"))

    for name, function, target_key in steps:
        try:
            result = function()

            if target_key == "datasets":
                manifest[target_key].append(result)
            elif target_key == "optional_sources":
                manifest[target_key].append(result)
            else:
                manifest[target_key] = result

        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            print(f"\nERROR — {message}", file=sys.stderr)
            manifest["errors"].append(message)

    manifest_path = BENCHMARK_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nManifest: {manifest_path.relative_to(ROOT)}")

    if manifest["errors"]:
        print("\nSome sources did not download. Other completed files are preserved.")
        return 1

    print("\nData acquisition completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
