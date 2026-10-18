from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/historical_event_returns.parquet"
SCORED = ROOT / "data/processed/historical_event_returns_scored.parquet"
PANEL = ROOT / "data/processed/historical_session_panel.parquet"
PREDICTIONS = ROOT / "data/processed/historical_holdout_predictions.parquet"
REPORT = ROOT / "outputs/calibration/historical_calibration_report.json"
MODEL_OUTPUT = ROOT / "outputs/calibration/historical_calibration_model.json"

SENTIMENT_MODEL = "ProsusAI/finbert"
TOPIC_MODEL = "leonas5555/finnews-topic-single-classify"


def predict_batches(model_id, texts, kind, batch_size, max_length):
    print(f"\nLoading {kind} model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Device: {device} | Texts: {len(texts):,}")

    label_map = {
        int(k): str(v).strip()
        for k, v in model.config.id2label.items()
    }
    lowered = {idx: label.lower() for idx, label in label_map.items()}

    outputs = []
    positive_idx = next((i for i, label in lowered.items()
                         if "positive" in label), None)
    negative_idx = next((i for i, label in lowered.items()
                         if "negative" in label), None)
    neutral_idx = next((i for i, label in lowered.items()
                        if "neutral" in label), None)

    if kind == "sentiment" and any(
        i is None for i in (positive_idx, negative_idx, neutral_idx)
    ):
        raise ValueError(f"Cannot identify sentiment labels: {label_map}")

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        for row in probs:
            best_idx = int(np.argmax(row))

            if kind == "sentiment":
                pos = float(row[positive_idx])
                neg = float(row[negative_idx])
                neu = float(row[neutral_idx])
                outputs.append({
                    "positive_probability": pos,
                    "negative_probability": neg,
                    "neutral_probability": neu,
                    "sentiment_score": pos - neg,
                    "sentiment_confidence": float(np.max(row)),
                })
            else:
                outputs.append({
                    "event_topic": label_map.get(best_idx, f"LABEL_{best_idx}"),
                    "event_topic_confidence": float(row[best_idx]),
                })

        if (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(texts):
            done = min(start + batch_size, len(texts))
            print(f"  {kind}: {done:,}/{len(texts):,}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs


def dominant_topic(values):
    values = values.dropna().astype(str)
    if values.empty:
        return "Unknown"
    return str(values.value_counts().index[0])


def fit_ridge(x_train, y_train, x_test, alpha=10.0):
    """Standardized ridge regression with training-only statistics."""
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0

    z_train = (x_train - x_mean) / x_scale
    z_test = (x_test - x_mean) / x_scale
    y_mean = float(np.mean(y_train))

    lhs = z_train.T @ z_train + alpha * np.eye(z_train.shape[1])
    rhs = z_train.T @ (y_train - y_mean)

    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    predictions = y_mean + z_test @ coefficients

    return {
        "predictions": predictions,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "coefficients": coefficients,
    }


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted, error = actual[valid], predicted[valid], error[valid]

    correlation = None
    if len(actual) > 1 and np.std(actual) > 1e-12 and np.std(predicted) > 1e-12:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])

    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson_correlation": correlation,
        "directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args()

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT}")

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))
    frame = pd.read_parquet(INPUT)

    required = {
        "ticker", "headline", "event_session_date",
        "forward_1d_abnormal_return_pct",
        "event_window_abnormal_return_pct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    frame["headline"] = frame["headline"].fillna("").astype(str).str.strip()
    fallback = frame.get(
        "article_text", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    frame["_model_text"] = frame["headline"].where(
        frame["headline"].str.len() > 0, fallback
    )
    frame["_model_text"] = frame["_model_text"].str.slice(0, 2000)

    frame = frame[frame["_model_text"].str.len() > 0].copy()
    if frame.empty:
        raise ValueError("No non-empty news text remains.")

    unique_texts = frame["_model_text"].drop_duplicates().tolist()
    print(f"Rows: {len(frame):,} | Unique texts: {len(unique_texts):,}")

    sentiment = predict_batches(
        SENTIMENT_MODEL, unique_texts, "sentiment",
        args.batch_size, args.max_length,
    )
    topics = predict_batches(
        TOPIC_MODEL, unique_texts, "topic",
        args.batch_size, args.max_length,
    )

    unique_predictions = pd.DataFrame({
        "_model_text": unique_texts,
        **{
            key: [result[key] for result in sentiment]
            for key in sentiment[0]
        },
        **{
            key: [result[key] for result in topics]
            for key in topics[0]
        },
    })

    scored = frame.merge(
        unique_predictions, on="_model_text", how="left",
        validate="many_to_one",
    )
    scored.drop(columns=["_model_text"], inplace=True)

    SCORED.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORED, index=False, compression="zstd")
    print(f"\nSaved model-scored events: {SCORED}")

    scored["session_date"] = pd.to_datetime(
        scored["event_session_date"], errors="coerce"
    )
    scored["forward_1d_abnormal_return_pct"] = pd.to_numeric(
        scored["forward_1d_abnormal_return_pct"], errors="coerce"
    )
    scored["event_window_abnormal_return_pct"] = pd.to_numeric(
        scored["event_window_abnormal_return_pct"], errors="coerce"
    )

    # One observation per ticker/session avoids overweighting days with many articles.
    panel = scored.groupby(
        ["ticker", "session_date"], as_index=False, dropna=False
    ).agg(
        headline_count=("headline", "size"),
        sentiment_score=("sentiment_score", "mean"),
        mean_abs_sentiment=(
            "sentiment_score", lambda s: float(np.mean(np.abs(s)))
        ),
        sentiment_confidence=("sentiment_confidence", "mean"),
        dominant_topic=("event_topic", dominant_topic),
        event_window_abnormal_return_pct=(
            "event_window_abnormal_return_pct", "first"
        ),
        forward_1d_abnormal_return_pct=(
            "forward_1d_abnormal_return_pct", "first"
        ),
    )

    panel = panel.dropna(subset=[
        "session_date", "forward_1d_abnormal_return_pct",
        "sentiment_score",
    ]).sort_values("session_date").reset_index(drop=True)

    if len(panel) < 100:
        raise ValueError(f"Too few ticker-session observations: {len(panel)}")

    unique_dates = np.sort(panel["session_date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_index = max(1, min(split_index, len(unique_dates) - 1))
    cutoff = pd.Timestamp(unique_dates[split_index])

    train_mask = panel["session_date"] < cutoff
    test_mask = ~train_mask
    train = panel.loc[train_mask].copy()
    test = panel.loc[test_mask].copy()

    if len(train) < 50 or len(test) < 20:
        raise ValueError(
            f"Insufficient temporal split: train={len(train)}, test={len(test)}"
        )

    # Ticker and predicted topic are explanatory features, not target-derived labels.
    feature_frame = panel[[
        "sentiment_score", "mean_abs_sentiment",
        "sentiment_confidence", "ticker", "dominant_topic",
    ]].copy()
    feature_frame["ticker"] = feature_frame["ticker"].fillna("UNKNOWN").astype(str)
    feature_frame["dominant_topic"] = (
        feature_frame["dominant_topic"].fillna("Unknown").astype(str)
    )
    x_frame = pd.get_dummies(
        feature_frame,
        columns=["ticker", "dominant_topic"],
        dtype=float,
    )

    x_all = x_frame.to_numpy(dtype=float)
    y_all = panel["forward_1d_abnormal_return_pct"].to_numpy(dtype=float)
    abs_y_all = np.abs(y_all)

    train_indices = np.flatnonzero(train_mask.to_numpy())
    test_indices = np.flatnonzero(test_mask.to_numpy())

    x_train, x_test = x_all[train_indices], x_all[test_indices]
    y_train, y_test = y_all[train_indices], y_all[test_indices]
    abs_y_train, abs_y_test = abs_y_all[train_indices], abs_y_all[test_indices]

    # Model A: predict signed next-session SPY-relative return.
    return_fit = fit_ridge(
        x_train, y_train, x_test, alpha=args.ridge_alpha
    )
    return_pred = return_fit["predictions"]

    signed_metrics = regression_metrics(y_test, return_pred)
    signed_metrics["zero_return_baseline_mae"] = float(np.mean(np.abs(y_test)))
    signed_metrics["zero_return_baseline_rmse"] = float(
        np.sqrt(np.mean(y_test ** 2))
    )
    signed_metrics["oos_r2_vs_zero"] = float(
        1.0 - np.mean((y_test - return_pred) ** 2) / max(
            np.mean(y_test ** 2), 1e-12
        )
    )

    # Model B: estimate the magnitude of the next-session abnormal move.
    magnitude_fit = fit_ridge(
        x_train, abs_y_train, x_test, alpha=args.ridge_alpha
    )
    predicted_magnitude = np.clip(
        magnitude_fit["predictions"], 0.0, max(float(np.max(abs_y_train)), 1e-8)
    )
    magnitude_metrics = regression_metrics(abs_y_test, predicted_magnitude)

    train_magnitude_mean = float(np.mean(abs_y_train))
    baseline_magnitude = np.full(len(abs_y_test), train_magnitude_mean)
    magnitude_metrics["training_mean_baseline_mae"] = float(
        np.mean(np.abs(abs_y_test - baseline_magnitude))
    )
    magnitude_metrics["training_mean_baseline_rmse"] = float(
        np.sqrt(np.mean((abs_y_test - baseline_magnitude) ** 2))
    )

    # Map predicted magnitude to an experimental 1-10 score using TRAIN deciles.
    thresholds = np.quantile(abs_y_train, np.arange(0.1, 1.0, 0.1))
    calibrated_impact = 1 + np.searchsorted(
        thresholds, predicted_magnitude, side="right"
    )

    holdout = test[[
        "ticker", "session_date", "headline_count", "sentiment_score",
        "mean_abs_sentiment", "sentiment_confidence", "dominant_topic",
        "event_window_abnormal_return_pct", "forward_1d_abnormal_return_pct",
    ]].copy()
    holdout["predicted_forward_abnormal_return_pct"] = return_pred
    holdout["actual_absolute_forward_abnormal_return_pct"] = abs_y_test
    holdout["predicted_absolute_forward_abnormal_return_pct"] = predicted_magnitude
    holdout["experimental_impact_score_1_10"] = calibrated_impact

    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL, index=False, compression="zstd")
    holdout.to_parquet(PREDICTIONS, index=False, compression="zstd")

    def serializable_fit(fit):
        return {
            "target_mean": fit["y_mean"],
            "feature_mean": fit["x_mean"].tolist(),
            "feature_scale": fit["x_scale"].tolist(),
            "coefficients": fit["coefficients"].tolist(),
        }

    model_artifact = {
        "method": "standardized ridge regression",
        "ridge_alpha": args.ridge_alpha,
        "feature_names": x_frame.columns.tolist(),
        "return_model": serializable_fit(return_fit),
        "absolute_return_model": serializable_fit(magnitude_fit),
        "impact_decile_thresholds_pct": thresholds.tolist(),
        "impact_scale": "experimental training-set deciles, 1-10",
        "fit_period_end_exclusive": cutoff.isoformat(),
        "warning": "Research calibration only; not yet validated for live investment decisions.",
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.write_text(
        json.dumps(model_artifact, indent=2), encoding="utf-8"
    )

    report = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input": str(INPUT.relative_to(ROOT)),
        "scored_events": int(len(scored)),
        "unique_headlines_scored": int(len(unique_texts)),
        "ticker_session_observations": int(len(panel)),
        "unique_tickers": int(panel["ticker"].nunique()),
        "unique_sessions": int(len(unique_dates)),
        "training_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "training_period": {
            "start": train["session_date"].min().date().isoformat(),
            "end": train["session_date"].max().date().isoformat(),
        },
        "holdout_period": {
            "start": test["session_date"].min().date().isoformat(),
            "end": test["session_date"].max().date().isoformat(),
        },
        "models": {
            "sentiment": SENTIMENT_MODEL,
            "event_topic": TOPIC_MODEL,
            "calibration": "ridge regression, fitted on first 80% of sessions",
        },
        "signed_return_holdout_metrics": signed_metrics,
        "absolute_return_magnitude_holdout_metrics": magnitude_metrics,
        "limitations": [
            "Only assets covered by the downloaded adjusted-price dataset are included.",
            "Topic model was trained on a financial social-media topic dataset; performance on publisher news may differ.",
            "SPY-relative return is a simple benchmark adjustment, not a causal estimate.",
            "Multiple articles are aggregated by ticker and event session.",
            "Headline sentiment and model predictions are not evidence of tradable alpha.",
            "Impact score is an experimental return-magnitude calibration, not probability of loss.",
        ],
        "outputs": {
            "scored_events": str(SCORED.relative_to(ROOT)),
            "ticker_session_panel": str(PANEL.relative_to(ROOT)),
            "holdout_predictions": str(PREDICTIONS.relative_to(ROOT)),
            "calibration_model": str(MODEL_OUTPUT.relative_to(ROOT)),
        },
    }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== HISTORICAL HOLDOUT EVALUATION COMPLETE ===")
    print(f"Unique headlines scored: {len(unique_texts):,}")
    print(f"Ticker-session rows:     {len(panel):,}")
    print(f"Train rows:              {len(train):,}")
    print(f"Holdout rows:            {len(test):,}")
    print(f"Holdout begins:          {cutoff.date()}")
    print("\nSigned return prediction:")
    print(json.dumps(signed_metrics, indent=2))
    print("\nAbsolute return magnitude:")
    print(json.dumps(magnitude_metrics, indent=2))
    print(f"\nReport: {REPORT}")
    print(f"Holdout predictions: {PREDICTIONS}")
    print(f"Calibration model: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
