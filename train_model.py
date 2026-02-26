"""
Train multiple regressors to predict productivity_score (0-100).

- Uses sklearn Pipeline + ColumnTransformer
- Models: RandomForestRegressor, GradientBoostingRegressor, XGBRegressor
- Group split by student_id (prevents leakage)
- Hyperparameter tuning (RandomizedSearchCV)
- Evaluates and prints metrics for all 3 models (MAE, RMSE, R²) on same test split
- Selects best by Test RMSE (lower is better), saves best pipeline
- Saves:
  - outputs/pipeline.joblib
  - outputs/metrics.json
  - outputs/feature_importance.png

Example:
  python train_model.py --data data/session_features.csv --target productivity_score --group student_id --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# XGBoost
try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Install it using: pip install xgboost"
    ) from e


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> Dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=rmse(y_true, y_pred),
        r2=float(r2_score(y_true, y_pred)),
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    # Works with sklearn >= 1.0
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def save_feature_importance(best_pipeline: Pipeline, out_png: str) -> None:
    import matplotlib.pyplot as plt

    pre = best_pipeline.named_steps["preprocess"]
    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        # For models without feature_importances_
        return

    importances = model.feature_importances_
    names = get_feature_names(pre)
    if len(names) != len(importances):
        # fallback: generic names
        names = [f"f{i}" for i in range(len(importances))]

    # top 25
    idx = np.argsort(importances)[::-1][:25]
    top_names = [names[i] for i in idx]
    top_vals = importances[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(list(reversed(top_names)), list(reversed(top_vals)))
    plt.title("Top Feature Importances (Best Model)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_preprocessor(df: pd.DataFrame, target: str, drop_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df.drop(columns=[target], errors="ignore").copy()

    # drop IDs/timestamps (high leakage / high cardinality)
    for c in drop_cols:
        if c in X.columns:
            X.drop(columns=[c], inplace=True)

    # split column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return pre, num_cols, cat_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV path")
    parser.add_argument("--target", default="productivity_score")
    parser.add_argument("--group", default="student_id", help="Group column for split/CV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--outputs-dir", default="outputs")

    # target cleaning policy (important for your dataset)
    parser.add_argument(
        "--zero-target-policy",
        choices=["keep", "drop"],
        default="drop",
        help="If productivity_score has many 0s (unlabeled), drop them by default.",
    )

    args = parser.parse_args()
    os.makedirs(args.outputs_dir, exist_ok=True)

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns.")

    if args.group not in df.columns:
        raise ValueError(f"Group column '{args.group}' not found in columns.")

    # Basic cleaning: numeric target, clamp to [0,100]
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.dropna(subset=[args.target]).copy()
    df[args.target] = df[args.target].clip(0, 100)

    # Handle many zeros (often means unlabeled sessions)
    zero_ratio = float((df[args.target] == 0).mean())
    if args.zero_target_policy == "drop" and zero_ratio > 0:
        df = df[df[args.target] != 0].copy()

    # Drop leakage columns
    drop_cols = ["session_id", "session_start_ts", "session_end_ts"]

    preprocessor, num_cols, cat_cols = build_preprocessor(df, args.target, drop_cols)

    # Prepare split
    groups = df[args.group].astype(str).values
    y = df[args.target].values
    X = df.drop(columns=[args.target], errors="ignore").copy()
    for c in drop_cols:
        if c in X.columns:
            X.drop(columns=[c], inplace=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]

    # CV splitter (group-aware)
    cv = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(g_train))))

    # -------------------------
    # Models + tuning spaces
    # -------------------------
    candidates: List[Tuple[str, Any, Dict[str, Any], int]] = []

    rf = RandomForestRegressor(random_state=args.seed, n_jobs=-1)
    rf_params = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.7, 0.9],
    }
    candidates.append(("RandomForestRegressor", rf, rf_params, 25))

    gbr = GradientBoostingRegressor(random_state=args.seed)
    gbr_params = {
        "model__n_estimators": [200, 400, 700, 1000],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    candidates.append(("GradientBoostingRegressor", gbr, gbr_params, 25))

    # NOTE: eval_metric is set in constructor, NOT in fit() -> fixes your error
    xgb = XGBRegressor(
        random_state=args.seed,
        n_estimators=800,
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        n_jobs=-1,
    )
    xgb_params = {
        "model__n_estimators": [400, 700, 1000, 1400],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5, 8],
        "model__reg_alpha": [0.0, 0.1, 0.5],
        "model__reg_lambda": [1.0, 1.5, 2.0, 3.0],
    }
    candidates.append(("XGBRegressor", xgb, xgb_params, 35))

    results: Dict[str, Any] = {
        "data": args.data,
        "target": args.target,
        "group": args.group,
        "seed": args.seed,
        "test_size": args.test_size,
        "zero_target_policy": args.zero_target_policy,
        "zero_ratio_before_policy": zero_ratio,
        "rows_after_cleaning": int(len(df)),
        "models": {},
    }

    best_name = None
    best_pipe = None
    best_test_rmse = float("inf")

    for name, model, param_dist, n_iter in candidates:
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train, y_train, groups=g_train)

        tuned = search.best_estimator_
        preds = tuned.predict(X_test)
        m = evaluate(y_test, preds)

        results["models"][name] = {
            "best_params": search.best_params_,
            "test_metrics": m.as_dict(),
        }

        print(f"\n✅ {name}")
        print(f"   Test MAE : {m.mae:.4f}")
        print(f"   Test RMSE: {m.rmse:.4f}")
        print(f"   Test R²  : {m.r2:.5f}")

        if m.rmse < best_test_rmse:
            best_test_rmse = m.rmse
            best_name = name
            best_pipe = tuned

    assert best_pipe is not None and best_name is not None

    # Save best model
    model_path = os.path.join(args.outputs_dir, "pipeline.joblib")
    dump(best_pipe, model_path)

    # Save feature importance
    fi_path = os.path.join(args.outputs_dir, "feature_importance.png")
    save_feature_importance(best_pipe, fi_path)

    results["best_model"] = {
        "name": best_name,
        "selected_by": "lowest_test_rmse",
        "pipeline_path": model_path,
        "feature_importance_path": fi_path,
    }

    # Save metrics.json
    metrics_path = os.path.join(args.outputs_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n🏁 Best model: {best_name} (Test RMSE={best_test_rmse:.4f})")
    print(f"✅ Saved: {model_path}")
    print(f"✅ Saved: {metrics_path}")
    print(f"✅ Saved: {fi_path}")


if __name__ == "__main__":
    main()
