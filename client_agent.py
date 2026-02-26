"""
Lightweight client agent that batches local telemetry into 5-minute sessions
and ships them to the FastAPI ingestion server.

Usage examples:
  # run in fake/demo mode (default) with 5-minute windows
  python client_agent.py --backend-url http://localhost:8000

  # replay existing JSONL events (for testing without sensors)
  python client_agent.py --mode replay --source data/fake_events.jsonl

  # collect minimal real telemetry via psutil (process list only)
  python client_agent.py --mode psutil --poll-seconds 30

The agent keeps an in-memory buffer for the current session window, computes
features + a local prediction (optional), then posts the batch to /ingest.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from predict import compute_session_features, load_model, predict_productivity


# -----------------------------
# Telemetry providers
# -----------------------------


class TelemetryProvider:
    def collect_slice(self, duration_seconds: int) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class FakeSliceProvider(TelemetryProvider):
    """
    Generates synthetic slices that respect the event schema.
    Useful for local testing or when sensors are not wired yet.
    """

    APPS = [
        ("vscode", "IDE"),
        ("pycharm", "IDE"),
        ("terminal", "Terminal"),
        ("docs", "Documentation"),
        ("chrome", "Browser"),
        ("firefox", "Browser"),
        ("youtube", "Media"),
        ("spotify", "Media"),
        ("discord", "Social"),
        ("steam", "Games"),
        ("valorant", "Games"),
        ("file_explorer", "Other"),
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def collect_slice(self, duration_seconds: int) -> Dict[str, Any]:
        app, category = self.rng.choice(self.APPS)
        switches = max(0, int(self.rng.gauss(2, 1)))
        typing_speed = max(0, self.rng.gauss(55, 18))
        keystrokes = max(0, int(self.rng.gauss(120, 60)))
        mouse_cov = max(0.02, min(0.6, self.rng.gauss(0.22, 0.08)))

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "application_usage": [
                {
                    "app": app,
                    "category": category,
                    "duration_seconds": duration_seconds,
                    "switches": switches,
                }
            ],
            "behavioral_metrics": {
                "avg_typing_speed": typing_speed,
                "keystrokes": keystrokes,
                "keys_pressed": ["a", "b", "c"] * 3,
                "mouse_coverage_area": mouse_cov,
                "mouse_clicks": max(0, int(self.rng.gauss(5, 3))),
                "idle_time": max(0, self.rng.expovariate(1 / 6)),
            },
            "coding_activity": {
                "compilers_used": ["python"] if self.rng.random() < 0.2 else [],
                "compiler_runs": int(self.rng.random() < 0.2),
                "file_saves": max(0, int(self.rng.gauss(2, 1))),
                "focus_percentage": max(0, min(100, self.rng.gauss(70, 10))),
            },
            "eye_tracking": {
                "attention_span": max(0, self.rng.gauss(4, 1.5)),
                "attention_time": int(duration_seconds * self.rng.uniform(0.6, 0.95)),
            },
        }
        return event


class ReplayProvider(TelemetryProvider):
    """Streams existing JSONL events; stops when the file ends."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.idx = 0

    def collect_slice(self, duration_seconds: int) -> Dict[str, Any]:
        if self.idx >= len(self.events):
            raise StopIteration
        ev = self.events[self.idx]
        self.idx += 1
        return ev


class PsutilProvider(TelemetryProvider):
    """
    Minimal psutil-based provider (process list only).
    Falls back to FakeSliceProvider if psutil is missing.
    """

    def __init__(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            self._fallback = FakeSliceProvider()
            self.psutil = None
            return
        self.psutil = psutil
        self._fallback = None

    def collect_slice(self, duration_seconds: int) -> Dict[str, Any]:
        if self.psutil is None:
            return self._fallback.collect_slice(duration_seconds)

        procs = list(self.psutil.process_iter(["name", "cpu_percent"]))
        procs.sort(key=lambda p: p.info.get("cpu_percent", 0), reverse=True)
        top = procs[0].info.get("name", "unknown") if procs else "idle"

        category = self._categorize(top.lower())
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "application_usage": [
                {
                    "app": top.lower(),
                    "category": category,
                    "duration_seconds": duration_seconds,
                    "switches": 0,
                }
            ],
            "behavioral_metrics": {
                "avg_typing_speed": None,
                "keystrokes": 0,
                "keys_pressed": [],
                "mouse_coverage_area": None,
                "mouse_clicks": 0,
                "idle_time": 0,
            },
            "coding_activity": {"compilers_used": [], "compiler_runs": 0, "file_saves": 0, "focus_percentage": None},
            "eye_tracking": {"attention_span": None, "attention_time": 0},
        }
        return event

    @staticmethod
    def _categorize(proc_name: str) -> str:
        mapping = {
            "code": "IDE",
            "vscode": "IDE",
            "pycharm": "IDE",
            "idea": "IDE",
            "powershell": "Terminal",
            "cmd": "Terminal",
            "terminal": "Terminal",
            "chrome": "Browser",
            "msedge": "Browser",
            "firefox": "Browser",
            "steam": "Games",
            "valorant": "Games",
            "spotify": "Media",
            "vlc": "Media",
            "discord": "Social",
        }
        for key, cat in mapping.items():
            if key in proc_name:
                return cat
        return "Other"


# -----------------------------
# Agent
# -----------------------------


@dataclass
class AgentConfig:
    backend_url: str
    student_id: str
    mode: str = "fake"
    session_minutes: int = 5
    poll_seconds: int = 30
    source: Optional[Path] = None
    model_path: str = "outputs/pipeline.joblib"
    local_score: bool = True
    once: bool = False


class ClientAgent:
    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.provider = self._make_provider(cfg)
        self.model = load_model(cfg.model_path) if cfg.local_score else None

    def _make_provider(self, cfg: AgentConfig) -> TelemetryProvider:
        if cfg.mode == "replay":
            if not cfg.source:
                raise ValueError("--source is required for replay mode")
            return ReplayProvider(cfg.source)
        if cfg.mode == "psutil":
            return PsutilProvider()
        return FakeSliceProvider()

    def _new_session_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"SES_{self.cfg.student_id}_{ts}"

    def run(self) -> None:
        session_id = self._new_session_id()
        session_start = datetime.now(timezone.utc)
        buffer: List[Dict[str, Any]] = []
        print(f"[agent] started for student_id={self.cfg.student_id} mode={self.cfg.mode}")

        while True:
            try:
                event = self.provider.collect_slice(self.cfg.poll_seconds)
            except StopIteration:
                if buffer:
                    self._flush(buffer, session_id, session_start)
                print("[agent] replay finished")
                return

            now = datetime.now(timezone.utc)
            event["student_id"] = self.cfg.student_id
            event.setdefault("session_id", session_id)
            buffer.append(event)

            if (now - session_start) >= timedelta(minutes=self.cfg.session_minutes):
                self._flush(buffer, session_id, session_start)
                if self.cfg.once:
                    return
                buffer = []
                session_start = datetime.now(timezone.utc)
                session_id = self._new_session_id()

            time.sleep(self.cfg.poll_seconds)

    def _flush(self, events: List[Dict[str, Any]], session_id: str, session_start: datetime) -> None:
        if not events:
            return
        session_end = datetime.now(timezone.utc)
        feat = compute_session_features(events)
        feat["student_id"] = self.cfg.student_id
        feat["session_id"] = session_id
        local_score = float(predict_productivity(feat, model=self.model)) if self.model else None

        payload = {
            "student_id": self.cfg.student_id,
            "session_id": session_id,
            "session_start_ts": session_start.isoformat(),
            "session_end_ts": session_end.isoformat(),
            "events": events,
        }

        try:
            resp = requests.post(f"{self.cfg.backend_url.rstrip('/')}/ingest", json=payload, timeout=10)
            resp.raise_for_status()
            server_score = resp.json().get("predicted_score")
            local_txt = f"{local_score:.2f}" if local_score is not None else "n/a"
            print(
                f"[agent] sent session={session_id} events={len(events)} "
                f"local_score={local_txt} server_score={server_score}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[agent] failed to send session {session_id}: {exc}")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> AgentConfig:
    p = argparse.ArgumentParser(description="Client agent for student-performance ingestion")
    p.add_argument("--backend-url", default="http://localhost:8000", help="Ingestion server base URL")
    p.add_argument("--student-id", default=socket.gethostname(), help="Unique ID for this student machine")
    p.add_argument("--mode", choices=["fake", "replay", "psutil"], default="fake", help="Telemetry provider")
    p.add_argument("--source", type=str, help="Path to JSONL file (replay mode)")
    p.add_argument("--session-minutes", type=int, default=5, help="Length of one session window")
    p.add_argument("--poll-seconds", type=int, default=30, help="Slice collection interval")
    p.add_argument("--model-path", default="outputs/pipeline.joblib", help="Local model path for optional scoring")
    p.add_argument("--no-local-score", action="store_true", help="Skip local scoring; rely on server")
    p.add_argument("--once", action="store_true", help="Send a single session then exit")
    args = p.parse_args()

    source_path = Path(args.source) if args.source else None

    return AgentConfig(
        backend_url=args.backend_url,
        student_id=args.student_id,
        mode=args.mode,
        session_minutes=args.session_minutes,
        poll_seconds=args.poll_seconds,
        source=source_path,
        model_path=args.model_path,
        local_score=not args.no_local_score,
        once=args.once,
    )


def main() -> None:
    cfg = parse_args()
    agent = ClientAgent(cfg)
    agent.run()


if __name__ == "__main__":
    main()
