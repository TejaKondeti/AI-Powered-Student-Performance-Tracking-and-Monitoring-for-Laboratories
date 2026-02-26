"""
Simple FastAPI ingestion service for student-performance events.

Responsibilities
- Receive batched events for a single student/session (typically a 5‑minute window)
- Compute session features + predicted productivity using the existing pipeline
- Persist raw events to `data/live/events.jsonl`
- Append session-level features (+ prediction) to `data/live/session_features.csv`
- Serve lightweight GET endpoints that the Streamlit dashboard can read (`/events`, `/sessions`)

Run (development):
  uvicorn ingestion_server:app --reload --port 8000

The service is intentionally small so it can sit on the instructor server
while thin client agents push data every 5 minutes.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from predict import compute_session_features, load_model, predict_productivity

app = FastAPI(title="Student Performance Ingestion", version="1.0.0")

DATA_DIR = Path("data/live")
EVENTS_PATH = DATA_DIR / "events.jsonl"
SESSIONS_PATH = DATA_DIR / "session_features.csv"
MODEL_PATH = "outputs/pipeline.joblib"

_model_lock = threading.Lock()
_csv_lock = threading.Lock()
_model = None


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_model(MODEL_PATH)
    return _model


class AppUsage(BaseModel):
    app: str
    category: str
    duration_seconds: float = 0.0
    switches: int = 0


class BehavioralMetrics(BaseModel):
    avg_typing_speed: Optional[float] = None
    keystrokes: int = 0
    keys_pressed: List[Any] = Field(default_factory=list)
    mouse_coverage_area: Optional[float] = None
    mouse_clicks: int = 0
    idle_time: float = 0.0


class CodingActivity(BaseModel):
    compilers_used: List[str] = Field(default_factory=list)
    compiler_runs: int = 0
    file_saves: int = 0
    focus_percentage: Optional[float] = None


class EyeTracking(BaseModel):
    attention_span: Optional[float] = None
    attention_time: int = 0


class Event(BaseModel):
    student_id: str
    session_id: Optional[str] = None
    timestamp: datetime
    application_usage: List[AppUsage] = Field(default_factory=list)
    behavioral_metrics: BehavioralMetrics = BehavioralMetrics()
    coding_activity: CodingActivity = CodingActivity()
    eye_tracking: EyeTracking = EyeTracking()


class SessionPayload(BaseModel):
    student_id: str
    session_id: Optional[str] = None
    session_start_ts: Optional[datetime] = None
    session_end_ts: Optional[datetime] = None
    events: List[Event]

    @validator("events")
    def _require_events(cls, v: List[Event]) -> List[Event]:
        if not v:
            raise ValueError("events list cannot be empty")
        return v

    @validator("session_id", always=True)
    def _fill_session_id(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if v:
            return v
        ts = datetime.utcnow().strftime("%Y%m%d%H%M")
        sid = values.get("student_id", "UNKNOWN")
        return f"SES_{sid}_{ts}"

    @validator("events")
    def _enforce_consistency(cls, v: List[Event], values: Dict[str, Any]) -> List[Event]:
        sid = values.get("student_id")
        session_id = values.get("session_id")
        for e in v:
            if e.student_id != sid:
                raise ValueError("event.student_id mismatch")
            if e.session_id and session_id and e.session_id != session_id:
                raise ValueError("event.session_id mismatch")
        return v


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _append_session_row(features: Dict[str, Any]) -> None:
    df = pd.DataFrame([features])
    header = not SESSIONS_PATH.exists()
    mode = "a" if SESSIONS_PATH.exists() else "w"
    with _csv_lock:
        df.to_csv(SESSIONS_PATH, mode=mode, header=header, index=False)


@app.on_event("startup")
def _startup() -> None:
    _ensure_dirs()
    # Warm model once to fail fast if missing
    _get_model()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Student Performance Ingestion API",
        "endpoints": ["/health", "/ingest", "/events", "/sessions", "/latest/{student_id}", "/docs"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model_loaded": str(Path(MODEL_PATH).exists())}


@app.post("/ingest")
def ingest(payload: SessionPayload) -> Dict[str, Any]:
    events = [e.dict() for e in payload.events]
    session_start = payload.session_start_ts or min(e["timestamp"] for e in events)
    session_end = payload.session_end_ts or max(e["timestamp"] for e in events)

    # persist raw events first
    _ensure_dirs()
    for e in events:
        if not e.get("session_id"):
            e["session_id"] = payload.session_id
        _append_jsonl(EVENTS_PATH, e)

    feat = compute_session_features(events)
    feat["student_id"] = payload.student_id
    feat["session_id"] = payload.session_id
    feat["session_start_ts"] = session_start
    feat["session_end_ts"] = session_end
    feat["event_count"] = len(events)

    model = _get_model()
    predicted_score = float(predict_productivity(feat, model=model))
    feat["predicted_score"] = predicted_score

    _append_session_row(feat)

    return {
        "status": "ok",
        "session_id": payload.session_id,
        "predicted_score": predicted_score,
        "event_count": len(events),
    }


def _read_jsonl_tail(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    # simple tail reader; file sizes expected to be modest
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    lines = lines[-limit:] if limit else lines
    return [json.loads(line) for line in lines]


@app.get("/events")
def list_events(limit: int = 400) -> List[Dict[str, Any]]:
    return _read_jsonl_tail(EVENTS_PATH, limit)


@app.get("/sessions")
def list_sessions(limit: int = 200) -> List[Dict[str, Any]]:
    if not SESSIONS_PATH.exists():
        return []
    df = pd.read_csv(SESSIONS_PATH)
    if limit:
        df = df.tail(limit)
    # convert timestamps to ISO for JSON friendliness
    for col in ["session_start_ts", "session_end_ts"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df.to_dict(orient="records")


@app.get("/latest/{student_id}")
def latest_for_student(student_id: str) -> Dict[str, Any]:
    sessions = list_sessions(limit=0)
    matches = [s for s in sessions if str(s.get("student_id")) == student_id]
    if not matches:
        raise HTTPException(status_code=404, detail="student_id not found")
    latest = max(matches, key=lambda x: x.get("session_end_ts") or "")
    return latest
