from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/pipeline.joblib")
    parser.add_argument("--data", required=True, help="CSV with target column included")
    parser.add_argument("--target", default="productivity_score")
    parser.add_argument("--out", default="outputs/eval_metrics.json")
    args = parser.parse_args()

    pipe = load(args.model)
    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data.")

    y_true = pd.to_numeric(df[args.target], errors="coerce").dropna().clip(0, 100)
    X = df.loc[y_true.index].drop(columns=[args.target])

    y_pred = pipe.predict(X)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Evaluation Metrics")
    print(f"   MAE : {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²  : {r2:.5f}")
    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    main()
