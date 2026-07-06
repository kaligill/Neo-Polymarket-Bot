"""
Trains the optional quantitative prior model (ai/probability_engine.py's
`_quant_model`) on historical resolved markets. Requires a labeled dataset:
one row per resolved market with structural features and the final binary
outcome (1 if resolved YES, 0 if NO).

Run manually as the trade log accumulates enough resolved markets:
    python -m backtest.train_model --data logs/resolved_markets.csv --out models/quant_model.pkl

Feature set is intentionally simple (momentum, volatility, time-to-resolution,
volume, liquidity) — extend as you find signal in your own data.
"""
from __future__ import annotations

import argparse
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from logger import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS = ["momentum", "volatility", "days_to_resolution", "volume_24h", "liquidity"]
LABEL_COLUMN = "resolved_yes"


def train(data_path: str, out_path: str):
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    log.info("model_trained", auc=round(auc, 4), brier_score=round(brier, 4), n_train=len(X_train), n_test=len(X_test))

    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    log.info("model_saved", path=out_path)

    return model, {"auc": auc, "brier_score": brier}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV of resolved markets with features + outcome")
    parser.add_argument("--out", default="models/quant_model.pkl")
    args = parser.parse_args()
    train(args.data, args.out)
