# fake_data_generator.py
"""
Generate realistic fake event logs in JSONL format (one JSON per line) matching the schema.

- Creates multiple students and sessions
- Simulates app usage, typing/mouse behavior, idle time, repetition/spam patterns
- Writes: data/fake_events.jsonl

Example:
  python fake_data_generator.py --out data/fake_events.jsonl --students 20 --sessions-per-student 3 --events-per-session 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------------
# Config / Constants
# -----------------------------

PRODUCTIVE_CATEGORIES = {"IDE", "Terminal", "Documentation"}
NON_PRODUCTIVE_CATEGORIES = {"Games", "Entertainment", "Social", "Media"}
NEUTRAL_CATEGORIES = {"Browser", "Other"}

APP_CATALOG: List[Tuple[str, str]] = [
    ("vscode", "IDE"),
    ("pycharm", "IDE"),
    ("terminal", "Terminal"),
    ("docs", "Documentation"),
    ("chrome", "Browser"),
    ("firefox", "Browser"),
    ("youtube", "Media"),
    ("spotify", "Media"),
    ("instagram", "Social"),
    ("discord", "Social"),
    ("steam", "Games"),
    ("valorant", "Games"),
    ("vlc", "Entertainment"),
    ("file_explorer", "Other"),
]

COMPILERS = ["python", "gcc", "g++", "node", "java", "javac"]

EDU_BROWSER_APPS = {"docs", "stackoverflow", "w3schools", "geeksforgeeks", "github"}
NON_EDU_BROWSER_HINTS = {"youtube", "instagram", "discord", "spotify", "steam", "valorant", "vlc"}


@dataclass(frozen=True)
class GenConfig:
    students: int
    sessions_per_student: int
    events_per_session: int
    seed: int
    start_utc: str  # ISO-like
    min_slice_seconds: int
    max_slice_seconds: int


def _iso_z(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _weighted_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    total = sum(w for _, w in items)
    pick = rng.random() * total
    upto = 0.0
    for name, w in items:
        upto += w
        if upto >= pick:
            return name
    return items[-1][0]


def _sample_apps_for_slice(rng: random.Random, profile: str) -> List[Dict]:
    """
    Returns list of {app, category, duration_seconds, switches}.
    """
    # Profiles push different behavior
    if profile == "high_productive":
        weights = [
            ("IDE", 3.0),
            ("Terminal", 1.2),
            ("Documentation", 1.2),
            ("Browser", 0.8),
            ("Other", 0.2),
            ("Media", 0.1),
            ("Games", 0.05),
            ("Social", 0.05),
            ("Entertainment", 0.05),
        ]
    elif profile == "mixed":
        weights = [
            ("IDE", 2.0),
            ("Terminal", 0.7),
            ("Documentation", 0.8),
            ("Browser", 1.5),
            ("Other", 0.3),
            ("Media", 0.5),
            ("Social", 0.3),
            ("Entertainment", 0.2),
            ("Games", 0.1),
        ]
    else:  # low_productive
        weights = [
            ("IDE", 0.8),
            ("Terminal", 0.2),
            ("Documentation", 0.3),
            ("Browser", 1.6),
            ("Other", 0.4),
            ("Media", 1.2),
            ("Social", 0.9),
            ("Entertainment", 0.6),
            ("Games", 0.8),
        ]

    chosen_cat = _weighted_choice(rng, weights)
    # number of apps in one slice (sometimes multi-task)
    n_apps = 1 if rng.random() < 0.75 else 2
    apps = []

    candidates = [a for a in APP_CATALOG if a[1] == chosen_cat]
    if not candidates:
        candidates = [a for a in APP_CATALOG if a[1] in {"Other", "Browser"}]

    for _ in range(n_apps):
        app, category = rng.choice(candidates)
        apps.append(
            {
                "app": app,
                "category": category,
                "duration_seconds": 0,  # fill later
                "switches": max(0, int(rng.gauss(3, 2))),
            }
        )
    return apps


def _make_keys_pressed_stats(rng: random.Random, profile: str, duration: int) -> List[str]:
    """
    IMPORTANT: This returns a list for the event payload only.
    Downstream code must NEVER store it, only derive stats.
    """
    # Baseline keys
    letters = list("abcdefghijklmnopqrstuvwxyz")
    special = ["Backspace", "Enter", "Tab", "Space"]
    pool = letters + special

    # determine if this event has spam/repeat pattern
    spam = False
    if profile == "low_productive" and rng.random() < 0.25:
        spam = True
    if profile == "mixed" and rng.random() < 0.10:
        spam = True
    if profile == "high_productive" and rng.random() < 0.04:
        spam = True

    # approximate keystrokes count for this slice
    # (downstream will also have behavioral_metrics["keystrokes"] as an integer)
    approx = max(0, int(rng.gauss(80, 30) * (duration / 60)))
    approx = min(approx, 600)

    if approx == 0:
        return []

    if spam:
        # small set of keys repeating
        spam_keys = rng.sample(pool, k=2)
        out = []
        streak = rng.randint(10, min(60, approx))
        key = spam_keys[0]
        for i in range(approx):
            if i % streak == 0:
                key = rng.choice(spam_keys)
            out.append(key)
        return out

    # normal typing: some repeats, some backspaces
    out = []
    last = None
    for _ in range(approx):
        if last is not None and rng.random() < 0.12:
            k = last  # repeat
        else:
            k = rng.choice(pool)
        # add occasional triple Backspace burst
        if rng.random() < 0.02:
            out.extend(["Backspace", "Backspace", "Backspace"])
            last = "Backspace"
            continue
        out.append(k)
        last = k
    return out[:approx]


def _student_profile(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.45:
        return "high_productive"
    if r < 0.80:
        return "mixed"
    return "low_productive"


def _browser_is_non_edu_hint(rng: random.Random) -> bool:
    # simulate browsing content; if non-edu, it may be penalized later
    return rng.random() < 0.35


def generate_events(cfg: GenConfig) -> List[Dict]:
    rng = random.Random(cfg.seed)

    start = datetime.fromisoformat(cfg.start_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    events: List[Dict] = []

    for s in range(1, cfg.students + 1):
        student_id = f"STU{s:03d}"
        profile = _student_profile(rng)

        for ses in range(1, cfg.sessions_per_student + 1):
            session_id = f"SES{start.strftime('%Y%m%d')}_{student_id}_{ses:03d}"
            t = start + timedelta(minutes=rng.randint(0, 240))

            # session-level tendency
            # low coverage more common in low_productive
            if profile == "low_productive":
                base_mouse_cov = rng.uniform(0.05, 0.18)
            elif profile == "mixed":
                base_mouse_cov = rng.uniform(0.10, 0.28)
            else:
                base_mouse_cov = rng.uniform(0.14, 0.38)

            for _ in range(cfg.events_per_session):
                slice_seconds = rng.randint(cfg.min_slice_seconds, cfg.max_slice_seconds)

                # decide idle chunk
                idle_time = 0
                if profile == "high_productive":
                    idle_time = max(0, int(rng.gauss(6, 8)))
                elif profile == "mixed":
                    idle_time = max(0, int(rng.gauss(15, 20)))
                else:
                    idle_time = max(0, int(rng.gauss(35, 25)))
                idle_time = min(idle_time, slice_seconds)

                apps = _sample_apps_for_slice(rng, profile)
                # split duration across apps (excluding idle)
                active_seconds = max(0, slice_seconds - idle_time)
                if apps:
                    share = active_seconds // len(apps)
                    leftover = active_seconds - share * len(apps)
                    for i, a in enumerate(apps):
                        a["duration_seconds"] = share + (1 if i < leftover else 0)

                # special chrome + non-edu hint: represent via app name sometimes
                # (still category Browser, but app_name suggests "youtube"/etc.)
                for a in apps:
                    if a["app"] in {"chrome", "firefox"} and a["category"] == "Browser":
                        if _browser_is_non_edu_hint(rng):
                            # pretend current tab is non-edu by swapping app label sometimes
                            a["app"] = rng.choice(list(NON_EDU_BROWSER_HINTS))
                        else:
                            a["app"] = rng.choice(list(EDU_BROWSER_APPS))

                # typing speed and keystrokes
                if profile == "high_productive":
                    typing_speed = max(5, rng.gauss(55, 10))
                elif profile == "mixed":
                    typing_speed = max(5, rng.gauss(40, 15))
                else:
                    typing_speed = max(3, rng.gauss(25, 18))

                # keystrokes roughly proportional to typing speed and active time
                keystrokes = max(0, int((typing_speed * active_seconds) / 3))
                keystrokes = min(keystrokes, 5000)

                keys_pressed = _make_keys_pressed_stats(rng, profile, slice_seconds)

                # mouse clicks
                if profile == "high_productive":
                    mouse_clicks = max(0, int(rng.gauss(18, 10) * (active_seconds / 60)))
                elif profile == "mixed":
                    mouse_clicks = max(0, int(rng.gauss(14, 12) * (active_seconds / 60)))
                else:
                    mouse_clicks = max(0, int(rng.gauss(10, 15) * (active_seconds / 60)))
                mouse_clicks = min(mouse_clicks, 800)

                # mouse coverage around baseline; idle reduces movement
                mouse_cov = max(0.0, min(1.0, rng.gauss(base_mouse_cov, 0.05)))
                if idle_time > slice_seconds * 0.5:
                    mouse_cov *= rng.uniform(0.4, 0.8)

                # coding activity
                # focus percentage
                if profile == "high_productive":
                    focus = max(10, min(100, rng.gauss(82, 10)))
                elif profile == "mixed":
                    focus = max(5, min(100, rng.gauss(65, 18)))
                else:
                    focus = max(0, min(100, rng.gauss(45, 25)))

                # compilers used and runs
                compiler_runs = 0
                file_saves = 0
                compilers_used: List[str] = []
                if any(a["category"] in {"IDE", "Terminal"} for a in apps) and active_seconds > 0:
                    if profile == "high_productive":
                        compiler_runs = max(0, int(rng.gauss(6, 3) * (active_seconds / 60)))
                        file_saves = max(0, int(rng.gauss(10, 5) * (active_seconds / 60)))
                    elif profile == "mixed":
                        compiler_runs = max(0, int(rng.gauss(3, 3) * (active_seconds / 60)))
                        file_saves = max(0, int(rng.gauss(6, 6) * (active_seconds / 60)))
                    else:
                        compiler_runs = max(0, int(rng.gauss(2, 4) * (active_seconds / 60)))
                        file_saves = max(0, int(rng.gauss(4, 7) * (active_seconds / 60)))

                    compiler_runs = min(compiler_runs, 50)
                    file_saves = min(file_saves, 80)

                    # compilers used (0-2)
                    if compiler_runs > 0 and rng.random() < 0.75:
                        compilers_used = rng.sample(COMPILERS, k=rng.choice([1, 1, 2]))

                # eye tracking
                if profile == "high_productive":
                    attention_span = max(1.0, rng.gauss(9.5, 2.0))
                    attention_time = max(0, int(rng.gauss(active_seconds * 0.70, active_seconds * 0.12)))
                elif profile == "mixed":
                    attention_span = max(1.0, rng.gauss(7.0, 2.8))
                    attention_time = max(0, int(rng.gauss(active_seconds * 0.55, active_seconds * 0.18)))
                else:
                    attention_span = max(0.8, rng.gauss(5.0, 3.0))
                    attention_time = max(0, int(rng.gauss(active_seconds * 0.40, active_seconds * 0.22)))
                attention_time = min(attention_time, active_seconds)

                event = {
                    "student_id": student_id,
                    "session_id": session_id,
                    "timestamp": _iso_z(t),
                    "application_usage": apps,
                    "behavioral_metrics": {
                        "avg_typing_speed": round(float(typing_speed), 2),
                        "keystrokes": int(keystrokes),
                        "keys_pressed": keys_pressed,  # DO NOT STORE downstream
                        "mouse_coverage_area": round(float(mouse_cov), 4),
                        "mouse_clicks": int(mouse_clicks),
                        "idle_time": int(idle_time),
                    },
                    "coding_activity": {
                        "compilers_used": compilers_used,
                        "compiler_runs": int(compiler_runs),
                        "file_saves": int(file_saves),
                        "focus_percentage": round(float(focus), 2),
                    },
                    "eye_tracking": {
                        "attention_span": round(float(attention_span), 2),
                        "attention_time": int(attention_time),
                    },
                }
                events.append(event)
                t += timedelta(seconds=slice_seconds)

    # sort globally by time
    events.sort(key=lambda e: (e["student_id"], e["session_id"], e["timestamp"]))
    return events


def write_jsonl(events: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/fake_events.jsonl")
    p.add_argument("--students", type=int, default=15)
    p.add_argument("--sessions-per-student", type=int, default=3)
    p.add_argument("--events-per-session", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-utc", type=str, default="2025-01-15T09:00:00Z")
    p.add_argument("--min-slice-seconds", type=int, default=60)
    p.add_argument("--max-slice-seconds", type=int, default=180)
    args = p.parse_args()

    cfg = GenConfig(
        students=args.students,
        sessions_per_student=args.sessions_per_student,
        events_per_session=args.events_per_session,
        seed=args.seed,
        start_utc=args.start_utc,
        min_slice_seconds=args.min_slice_seconds,
        max_slice_seconds=args.max_slice_seconds,
    )

    events = generate_events(cfg)
    write_jsonl(events, Path(args.out))

    # quick summary
    unique_sessions = len({(e["student_id"], e["session_id"]) for e in events})
    print(f"Wrote {len(events)} events for {unique_sessions} sessions -> {args.out}")
    print("Sample event:")
    print(json.dumps(events[0], indent=2)[:1200])


if __name__ == "__main__":
    main()
