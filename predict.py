# predict.py
"""
Shared feature engineering + inference.

Key goal: avoid train/serve skew:
- dataset_builder.py imports compute_session_features() and pseudo_label_productivity()

Also provides:
- predict_productivity(json_session_or_features) -> score

Usage examples:

1) Predict from raw events:
   from predict import predict_productivity, load_model
   model = load_model("outputs/pipeline.joblib")
   score = predict_productivity(events_list, model=model)

2) Predict from a precomputed feature dict:
   score = predict_productivity(feature_dict, model=model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd


# -----------------------------
# Constants / Thresholds
# -----------------------------

PRODUCTIVE_CATEGORIES = {"IDE", "Terminal", "Documentation"}
NON_PRODUCTIVE_CATEGORIES = {"Games", "Entertainment", "Social", "Media"}

# Browser is neutral but can be penalized using app-name hints (fake data uses these too)
BROWSER_CATEGORY = "Browser"
NON_EDU_HINT_APPS = {"youtube", "instagram", "discord", "spotify", "steam", "valorant", "vlc"}

MOUSE_LOW_THRESHOLD = 0.12

# Repetition heuristics
REPEAT_RATIO_HIGH = 0.22
LONG_REPEAT_STREAK_FLAG = 12

# "Spam pattern": small set of keys repeats heavily; we never store keys, only derived stats
SPAM_SMALL_SET_MAX_KEYS = 3
SPAM_MIN_TOTAL_KEYS = 120
SPAM_MIN_DOMINANCE = 0.75  # top K keys cover >= 75%


@dataclass(frozen=True)
class PseudoLabelWeights:
    idle_penalty_weight: float = 45.0
    low_mouse_penalty_weight: float = 20.0
    repetition_penalty_weight: float = 18.0
    non_productive_penalty_weight: float = 55.0
    chrome_non_edu_penalty_weight: float = 18.0


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


# -----------------------------
# Key-press derived stats (NO raw storage)
# -----------------------------

def derive_key_stats(keys_pressed: List[Any]) -> Dict[str, Any]:
    """
    Compute repetition metrics without saving content.
    keys_pressed is used only in-memory here.
    """
    # normalize to strings
    keys = [str(k) for k in keys_pressed if k is not None]
    n = len(keys)
    if n == 0:
        return {
            "repetitive_key_ratio": 0.0,
            "long_repeat_streak": 0,
            "spam_pattern_flag": 0,
        }

    repeats = 0
    longest = 1
    current = 1

    for i in range(1, n):
        if keys[i] == keys[i - 1]:
            repeats += 1
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    repetitive_ratio = safe_div(repeats, max(1, n - 1))

    # spam heuristic: small set dominates
    # (count frequencies in-memory)
    freq: Dict[str, int] = {}
    for k in keys:
        freq[k] = freq.get(k, 0) + 1
    top_counts = sorted(freq.values(), reverse=True)[:SPAM_SMALL_SET_MAX_KEYS]
    dominance = safe_div(sum(top_counts), n)

    spam = int(n >= SPAM_MIN_TOTAL_KEYS and dominance >= SPAM_MIN_DOMINANCE)

    return {
        "repetitive_key_ratio": float(repetitive_ratio),
        "long_repeat_streak": int(longest),
        "spam_pattern_flag": int(spam),
    }


# -----------------------------
# Feature engineering (session aggregation)
# -----------------------------

def compute_session_features(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert event-level JSON logs -> one session feature dict.
    Requires: all events belong to same (student_id, session_id).
    """

    # -----------------------------
    # accumulators
    # -----------------------------
    cat_dur = {}
    app_dur = {}

    total_duration = 0.0
    total_switches = 0.0

    typing_speeds = []
    keystrokes_total = 0
    mouse_clicks_total = 0
    idle_time_total = 0
    mouse_cov_values = []
    focus_values = []
    attention_time_total = 0
    attention_span_values = []
    compiler_runs_total = 0
    file_saves_total = 0
    compiler_set = set()

    low_mouse_count = 0
    repetitive_ratios = []
    long_streaks = []
    spam_flags = []

    chrome_time = 0.0
    games_time = 0.0
    media_time = 0.0
    non_productive_time = 0.0
    browser_non_edu_time = 0.0

    # -----------------------------
    # iterate events
    # -----------------------------
    for e in events:
        apps = e.get("application_usage", [])
        bm = e.get("behavioral_metrics", {})
        ca = e.get("coding_activity", {})
        et = e.get("eye_tracking", {})

        event_active = 0.0

        for a in apps:
            app = str(a.get("app", "unknown")).lower()
            cat = str(a.get("category", "Other"))
            dur = float(a.get("duration_seconds", 0))
            sw = float(a.get("switches", 0))

            event_active += dur
            total_switches += sw

            cat_dur[cat] = cat_dur.get(cat, 0) + dur
            app_dur[app] = app_dur.get(app, 0) + dur

            if cat in {"Games", "Entertainment", "Social", "Media"}:
                non_productive_time += dur
            if cat == "Games":
                games_time += dur
            if cat == "Media":
                media_time += dur
            if app in {"youtube", "instagram", "discord", "spotify", "steam", "vlc"}:
                browser_non_edu_time += dur
            if app == "chrome":
                chrome_time += dur

        idle = float(bm.get("idle_time", 0))
        slice_total = event_active + idle
        total_duration += slice_total
        idle_time_total += idle

        ts = bm.get("avg_typing_speed")
        if ts is not None:
            typing_speeds.append(float(ts))

        keystrokes_total += int(bm.get("keystrokes", 0))
        mouse_clicks_total += int(bm.get("mouse_clicks", 0))

        mc = bm.get("mouse_coverage_area")
        if mc is not None:
            mc = float(mc)
            mouse_cov_values.append(mc)
            if mc < 0.12:
                low_mouse_count += 1

        ks = derive_key_stats(list(bm.get("keys_pressed", [])))
        repetitive_ratios.append(ks["repetitive_key_ratio"])
        long_streaks.append(ks["long_repeat_streak"])
        spam_flags.append(ks["spam_pattern_flag"])

        focus = ca.get("focus_percentage")
        if focus is not None:
            focus_values.append(float(focus))

        compiler_runs_total += int(ca.get("compiler_runs", 0))
        file_saves_total += int(ca.get("file_saves", 0))
        for c in ca.get("compilers_used", []):
            compiler_set.add(c)

        attention_time_total += int(et.get("attention_time", 0))
        span = et.get("attention_span")
        if span is not None:
            attention_span_values.append(float(span))

    total = total_duration if total_duration > 0 else 1

    # -----------------------------
    # output features
    # -----------------------------
    out = {
        "total_duration_seconds": total_duration,
        "productive_duration_seconds": sum(
            cat_dur.get(c, 0) for c in {"IDE", "Terminal", "Documentation"}
        ),
        "non_productive_duration_seconds": non_productive_time + browser_non_edu_time,
        "total_app_switches": total_switches,
        "switch_rate_per_min": total_switches / (total / 60),
        "typing_speed_mean": float(np.mean(typing_speeds)) if typing_speeds else 0,
        "typing_speed_std": float(np.std(typing_speeds)) if typing_speeds else 0,
        "keystrokes_total": keystrokes_total,
        "mouse_clicks_total": mouse_clicks_total,
        "idle_time_total": idle_time_total,
        "mouse_coverage_mean": float(np.mean(mouse_cov_values)) if mouse_cov_values else 0,
        "focus_percentage_mean": float(np.mean(focus_values)) if focus_values else 0,
        "attention_time_total": attention_time_total,
        "attention_span_mean": float(np.mean(attention_span_values)) if attention_span_values else 0,
        "compiler_runs_total": compiler_runs_total,
        "file_saves_total": file_saves_total,
        "compiler_count": len(compiler_set),

        "app_category_time_share_ide": cat_dur.get("IDE", 0) / total * 100,
        "app_category_time_share_browser": cat_dur.get("Browser", 0) / total * 100,
        "app_category_time_share_games": cat_dur.get("Games", 0) / total * 100,
        "app_category_time_share_media": cat_dur.get("Media", 0) / total * 100,
        "app_category_time_share_other": (
            total_duration - sum(cat_dur.get(c, 0) for c in ["IDE", "Browser", "Games", "Media"])
        ) / total * 100,

        "top_app_name": max(app_dur, key=app_dur.get) if app_dur else "unknown",
        "top_app_category": max(cat_dur, key=cat_dur.get) if cat_dur else "Other",

        "low_mouse_coverage_ratio": low_mouse_count / max(1, len(mouse_cov_values)),
        "repetitive_key_ratio": float(np.mean(repetitive_ratios)) if repetitive_ratios else 0,
        "long_repeat_streak": max(long_streaks) if long_streaks else 0,
        "spam_pattern_flag": int(any(spam_flags)),

        "non_productive_app_time_share": (non_productive_time + browser_non_edu_time) / total * 100,
        "chrome_time_share": chrome_time / total * 100,
        "games_time_share": games_time / total * 100,
        "media_time_share": media_time / total * 100,
        "browser_non_edu_time_share": browser_non_edu_time / total * 100,

        "productive_time_share": (
            sum(cat_dur.get(c, 0) for c in {"IDE", "Terminal", "Documentation"}) / total * 100
        ),
        "idle_time_share": idle_time_total / total * 100,

        # 🔥 FIX: REQUIRED BY TRAINED MODEL
        "event_count": len(events),
    }

    return out



# -----------------------------
# Pseudo-labeling (bootstrap option)
# -----------------------------

def pseudo_label_productivity(features: Dict[str, Any], w: Optional[PseudoLabelWeights] = None) -> float:
    """
    Base = productive_time_share * 1.0 (already 0..100)
    Penalties based on:
      - idle_time_share
      - low_mouse_coverage_ratio
      - high repetition (repetitive_key_ratio, long_repeat_streak, spam_pattern_flag)
      - games/media time
      - browser non-edu hints (browser_non_edu_time_share)
    """
    w = w or PseudoLabelWeights()

    base = float(features.get("productive_time_share", 0.0))

    idle_share = float(features.get("idle_time_share", 0.0)) / 100.0
    low_mouse_ratio = float(features.get("low_mouse_coverage_ratio", 0.0))
    rep_ratio = float(features.get("repetitive_key_ratio", 0.0))
    long_streak = int(features.get("long_repeat_streak", 0))
    spam_flag = int(features.get("spam_pattern_flag", 0))

    non_prod_share = float(features.get("non_productive_app_time_share", 0.0)) / 100.0
    browser_non_edu_share = float(features.get("browser_non_edu_time_share", 0.0)) / 100.0

    # penalties (keep them smooth, not too harsh)
    p_idle = w.idle_penalty_weight * min(1.0, idle_share / 0.40)  # 40% idle -> full penalty
    p_mouse = w.low_mouse_penalty_weight * min(1.0, low_mouse_ratio / 0.60)  # 60% low-mouse events
    p_nonprod = w.non_productive_penalty_weight * min(1.0, non_prod_share / 0.35)  # 35%+ nonprod
    p_chrome_non_edu = w.chrome_non_edu_penalty_weight * min(1.0, browser_non_edu_share / 0.25)

    # repetition penalty
    rep_strength = 0.0
    if rep_ratio > REPEAT_RATIO_HIGH:
        rep_strength += min(1.0, (rep_ratio - REPEAT_RATIO_HIGH) / 0.25)
    if long_streak >= LONG_REPEAT_STREAK_FLAG:
        rep_strength += min(1.0, (long_streak - LONG_REPEAT_STREAK_FLAG) / 20.0)
    if spam_flag:
        rep_strength += 1.0
    rep_strength = min(1.5, rep_strength)
    p_rep = w.repetition_penalty_weight * min(1.0, rep_strength)

    score = base - (p_idle + p_mouse + p_nonprod + p_chrome_non_edu + p_rep)
    return clamp(score, 0.0, 100.0)


# -----------------------------
# Inference helpers
# -----------------------------

def load_model(path: str = "outputs/pipeline.joblib"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Train first: python train_model.py --data data/session_features.csv --target productivity_score"
        )
    return joblib.load(p)


def _features_to_frame(feature_dict: Dict[str, Any]) -> pd.DataFrame:
    # Model expects the same feature columns used during training.
    # We'll just pass a single-row DataFrame; extra columns are ignored if not used.
    return pd.DataFrame([feature_dict])


JsonLike = Union[List[Dict[str, Any]], Dict[str, Any]]


def predict_productivity(json_session_or_features: JsonLike, model=None, model_path: str = "outputs/pipeline.joblib") -> float:
    """
    Accepts:
      - list of event dicts (raw session events)
      - OR a dict of session-level features
    Returns: predicted productivity score (0..100)
    """
    if model is None:
        model = load_model(model_path)

    if isinstance(json_session_or_features, list):
        feat = compute_session_features(json_session_or_features)
    elif isinstance(json_session_or_features, dict):
        # if it's raw event with keys like student_id, treat as single event list
        if "application_usage" in json_session_or_features and "behavioral_metrics" in json_session_or_features:
            feat = compute_session_features([json_session_or_features])
        else:
            feat = json_session_or_features
    else:
        raise TypeError("Input must be a list of events or a feature dict.")

    X = _features_to_frame(feat)
    pred = float(model.predict(X)[0])
    return clamp(pred, 0.0, 100.0)
