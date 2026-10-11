# Financial Risk Intelligence — Data Sources

## MIT-licensed labelled benchmarks

### Twitter Financial News Sentiment
- Source: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
- License listed by source: MIT
- Labels: bearish, bullish, neutral
- Use: sentiment evaluation and optional baseline training.
- Local exported splits: `data/raw/benchmarks/twitter_sentiment_*.csv`

### Twitter Financial News Topic
- Source: https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic
- License listed by source: MIT
- Labels: 20 topic categories
- Use: event-classification training and evaluation, followed by mapping to
  the ten event categories in our RiskSignal schema.
- Local exported splits: `data/raw/benchmarks/twitter_topic_*.csv`

The downloader records actual split row counts in
`data/raw/benchmarks/download_manifest.json`.

## Additional local evaluation data

### Financial PhraseBank
- Source: https://huggingface.co/datasets/takala/financial_phrasebank
- Configuration: `sentences_75agree`
- License: CC BY-NC-SA 3.0
- Local path: `data/cache/external/financial_phrasebank_75agree.csv`
- Redistribution: excluded from Git by default; check the source license and
  the intended use before redistributing.

## Market prices

- Source: Yahoo Finance through `yfinance`.
- Instruments: configured in `data/config/market_tickers.txt`.
- Period: daily data from the configured start date through the download date.
- Price convention: adjusted daily closing prices.
- Local cache: `data/cache/market/`.
- Intended use: historical analysis, abnormal-return features and backtests.
- These downloaded caches are not committed by default. Re-fetch them with
  the acquisition script to reproduce the market-data preparation.

## Bond holdings

- Sources:
  - https://www.ishares.com/us/products/239566/ishares-iboxx-investment-grade-corporate-bond-etf
  - https://www.ishares.com/us/products/239565/hyg-ishares-iboxx-high-yield-corporate-bond-etf
- Fields include sectors, asset classes, maturity, coupon and modified duration.
- Local cache: `data/cache/ishares/`.
- Holdings change over time; each downloaded file is point-in-time data and
  should retain its download timestamp.

## Optional research datasets

### FNSPID
- Source: https://huggingface.co/datasets/khaihernlow/fnspid
- License listed by source: CC BY-NC 4.0.
- Approximate hosted file size: 2.34 GB.
- Download only for non-commercial research use compatible with that license.
- Local cache: `data/cache/fnspid/`.

### SBA 7(a) loan data
- Source: https://web.data.sba.gov/en/dataset/7-a-504-foia
- The official portal lists public-domain data resources.
- The current quarterly files may differ from older historical releases.
- Charge-off outcomes are proxies for credit-loss analysis, not direct
  universal PD/LGD labels.
- Local cache: `data/cache/sba/`.

### Stock Tweets for Sentiment Analysis and Prediction
- Source: https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction
- Intended use: timestamped replay and backtesting.
- The dataset has no sentiment ground-truth labels in the described source;
  do not use it as labelled sentiment evaluation.
- Local cache: `data/cache/kaggle_tweets/`.
- Verify the dataset license on Kaggle before redistribution.

## Organizer-provided transaction data

Organizer-provided files must be stored under `data/raw/organizer_transactions/`
after they are received. Their exact formats, source and license must be
documented before use. No organizer data is fabricated by this project.

## Reproducibility

Run `python scripts/download_research_data.py` from the repository root.
Optional large or authenticated sources are explicitly enabled by command-line
flags. The script writes a download manifest with source paths and row counts.
