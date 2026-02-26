# dataset_builder.py
"""
Build a session-level dataset from event-level JSON logs.

Input:
- JSONL file (one JSON record per line) OR JSON file containing a list of events

Output:
- session_features.csv (one row per student_id + session_id)
- NEVER stores raw keys_pressed list (only derived stats)

Optional:
- pseudo-label productivity_score (0-100) using rules (for bootstrapping)

Example:
  python dataset_builder.py --input data/fake_events.jsonl --output data/session_features.csv --pseudo-label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from predict import compute_session_features, pseudo_label_productivity


def read_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    events: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
    else:
        obj = json.loads(text)
        if isinstance(obj, list):
            events = obj
        else:
            raise ValueError("JSON input must be a list of events, or use .jsonl for line-delimited JSON.")
    return events


def group_by_session(events: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for e in events:
        sid = str(e.get("student_id", "")).strip()
        ses = str(e.get("session_id", "")).strip()
        if not sid or not ses:
            continue
        grouped.setdefault((sid, ses), []).append(e)

    # sort each group by timestamp for stable aggregations
    for k in grouped:
        grouped[k].sort(key=lambda x: x.get("timestamp", ""))
    return grouped


def build_session_dataframe(
    events: List[Dict[str, Any]],
    add_pseudo_label: bool = False,
) -> pd.DataFrame:
    grouped = group_by_session(events)
    rows: List[Dict[str, Any]] = []

    for (student_id, session_id), sess_events in grouped.items():
        features = compute_session_features(sess_events)
        features["student_id"] = student_id
        features["session_id"] = session_id

        # helpful for dashboard
        features["session_start_ts"] = sess_events[0].get("timestamp")
        features["session_end_ts"] = sess_events[-1].get("timestamp")
        features["event_count"] = len(sess_events)

        if add_pseudo_label:
            features["productivity_score"] = pseudo_label_productivity(features)

        rows.append(features)

    df = pd.DataFrame(rows)

    # stable column order (ids first)
    preferred_first = [
        "student_id",
        "session_id",
        "session_start_ts",
        "session_end_ts",
        "event_count",
    ]
    cols = preferred_first + [c for c in df.columns if c not in preferred_first]
    df = df[cols].sort_values(["student_id", "session_id"]).reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Path to .jsonl or .json event logs")
    p.add_argument("--output", type=str, default="data/session_features.csv")
    p.add_argument(
        "--pseudo-label",
        action="store_true",
        help="Add pseudo productivity_score (0-100) for bootstrapped training",
    )
    args = p.parse_args()

    events = read_events(Path(args.input))
    if not events:
        raise SystemExit("No events found in input.")

    df = build_session_dataframe(events, add_pseudo_label=args.pseudo_label)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} session rows -> {out_path}")
    print("Columns:", list(df.columns)[:30], "...")


if __name__ == "__main__":
    main()
