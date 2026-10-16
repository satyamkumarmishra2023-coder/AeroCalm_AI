"""
AeroCalm AI  V6.0  ATC Fatigue Monitoring System
Voice-Only Scoring  Eye Closure Warnings  Epoch Stamping
Head Department Dashboard  Daily and Weekly Reports

V6.0 Changes
------------
[ATC-ONLY]   Removed all pilot/crew/manager roles.
             System built exclusively for Air Traffic Controllers.

[VOICE-ONLY] Camera no longer contributes to fatigue score.
             W_VOICE = 1.0 (100% voice biomarkers).
             Vision only sends eye-closure warning after
             EYE_CLOSE_WARNING_SEC of sustained eye closure.

[CALIBRATION] First session enters calibration mode.
              No score shown until personal baseline is built
              from sufficient natural speech (30-40 seconds).
              Baseline saved permanently to DB.
              All future sessions load baseline instantly.

[EPOCH]      Every 5 minutes, average of all voice scores
             collected during that window is stamped to DB.
             Only chunks where speech was detected are counted.
             Prevents 1-second spikes from creating false stamps.

[SESSION]    Operators click Start Duty and End Duty.
             Each duty creates a session record in DB.
             Session summary auto-generated on End Duty.

[DASHBOARD]  Head Department panel shows all operators,
             today status, recent alerts.
             Daily report per operator per date.
             Weekly report auto-generated every 7 days.

pip install:
    pip install fastapi uvicorn librosa numpy opencv-python mediapipe scipy

Run:
    uvicorn main:app --host 127.0.0.1 --port 8000
"""

import asyncio
import base64
import collections
import json
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import librosa
import numpy as np
from scipy.signal import find_peaks
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-14s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("AeroCalm")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("mediapipe").setLevel(logging.WARNING)


# ── Constants ─────────────────────────────────────────────────────

SAMPLE_RATE:    int   = 22_050
BUFFER_SECONDS: int   = 5
BUFFER_SIZE:    int   = SAMPLE_RATE * BUFFER_SECONDS

SMOOTH_WINDOW:  int   = 6
SILENCE_GATE:   float = 0.03
RMS_REFERENCE:  float = 0.15
ZONE_YELLOW:    float = 40.0
ZONE_RED:       float = 65.0
MIN_BUFFER_SEC: int   = 2
STREAM_TIMEOUT: float = 25.0
CONFIDENCE_SATURATION: float = 0.75

# Biomarker calibration
JITTER_SATURATION:     float = 3.0
SHIMMER_SATURATION:    float = 0.8
BASELINE_CHUNKS:       int   = 8
F0_DROP_SATURATION:    float = 20.0
CENTROID_ALERT_STD:    float = 1500.0
CENTROID_FATIGUED_STD: float = 300.0
HNR_HEALTHY_DB:        float = 25.0
HNR_BASELINE_DROP:     float = 5.0
SPEECH_RATIO_NORMAL:   float = 0.60
SYLLABLE_RATE_NORMAL:  float = 4.0
SPEECH_PAUSE_FACTOR:   float = 0.5
N_MFCC:                int   = 13
MFCC_SATURATION_DIST:  float = 0.25

# Spectral subtraction
SS_NOISE_FRAMES: int   = 8
SS_ALPHA:        float = 2.0
SS_BETA:         float = 0.01
SS_N_FFT:        int   = 512

# Vision — warning only, no score weight
EAR_BLINK_THRESH:          float = 0.20   # EAR below this = eye closed
EYE_CLOSE_WARNING_SEC:     float = 3.0    # Sustained closure before warning
EYE_CLOSE_FRAMES_THRESH:   int   = 6      # 3 sec × 2 fps = 6 frames
EYE_OPEN_RESET_FRAMES:     int   = 2      # 1 sec of open eyes resets counter

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Epoch stamping — 5 minutes for ATC
EPOCH_DURATION_SEC:         int = 300
MIN_SPEECH_FOR_STAMP:       int = 3      # Min speech chunks to generate a stamp

# Adaptive baseline
BASELINE_UPDATE_MARGIN:    float = 5.0
ALERT_THROTTLE_SEC:        float = 120.0  # Min seconds between same-zone alerts

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aerocalm")


# ── Database ──────────────────────────────────────────────────────

DB_PATH = Path("aerocalm_atc.db")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = get_db()
    c    = conn.cursor()

    # ATC operators
    c.execute("""
        CREATE TABLE IF NOT EXISTS atc_operators (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            operator_id        TEXT    UNIQUE NOT NULL,
            name               TEXT    NOT NULL,
            f0_baseline        REAL    DEFAULT NULL,
            hnr_baseline       REAL    DEFAULT NULL,
            mfcc_baseline      TEXT    DEFAULT NULL,
            baseline_freshness REAL    DEFAULT 0,
            baseline_set       INTEGER DEFAULT 0,
            created_at         TEXT    DEFAULT (datetime('now'))
        )
    """)

    # Duty sessions
    c.execute("""
        CREATE TABLE IF NOT EXISTS duty_sessions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            operator_id    TEXT    NOT NULL,
            operator_name  TEXT    NOT NULL,
            start_time     TEXT    NOT NULL,
            end_time       TEXT    DEFAULT NULL,
            duration_min   REAL    DEFAULT 0,
            avg_score      REAL    DEFAULT 0,
            max_score      REAL    DEFAULT 0,
            stamp_count    INTEGER DEFAULT 0,
            green_stamps   INTEGER DEFAULT 0,
            yellow_stamps  INTEGER DEFAULT 0,
            red_stamps     INTEGER DEFAULT 0,
            status         TEXT    DEFAULT 'active'
        )
    """)

    # 5-minute epoch stamps
    c.execute("""
        CREATE TABLE IF NOT EXISTS fatigue_stamps (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            operator_id    TEXT    NOT NULL,
            operator_name  TEXT    NOT NULL,
            session_id     INTEGER DEFAULT NULL,
            timestamp      TEXT    NOT NULL,
            date           TEXT    NOT NULL,
            avg_score      REAL    NOT NULL,
            zone           TEXT    NOT NULL,
            green_count    INTEGER DEFAULT 0,
            yellow_count   INTEGER DEFAULT 0,
            red_count      INTEGER DEFAULT 0,
            speech_chunks  INTEGER DEFAULT 0,
            duration_sec   INTEGER DEFAULT 0,
            alert_message  TEXT
        )
    """)

    # Eye closure warnings
    c.execute("""
        CREATE TABLE IF NOT EXISTS eye_warnings (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            operator_id    TEXT    NOT NULL,
            operator_name  TEXT    NOT NULL,
            session_id     INTEGER DEFAULT NULL,
            timestamp      TEXT    NOT NULL,
            duration_sec   REAL    NOT NULL,
            session_elapsed TEXT
        )
    """)

    conn.commit()
    conn.close()
    log.info("DATABASE  | aerocalm_atc.db ready.")


# ── Operator helpers ──────────────────────────────────────────────

def db_add_operator(operator_id: str, name: str) -> Dict:
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO atc_operators (operator_id, name) VALUES (?, ?)",
            (operator_id.upper().strip(), name.strip()),
        )
        conn.commit()
        return {"status": "ok", "operator_id": operator_id.upper(), "name": name}
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"Operator ID '{operator_id}' already registered.")
    finally:
        conn.close()


def db_list_operators() -> List[Dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT operator_id, name, baseline_set, baseline_freshness, created_at "
        "FROM atc_operators ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_delete_operator(operator_id: str) -> None:
    conn = get_db()
    conn.execute("DELETE FROM atc_operators WHERE operator_id=?", (operator_id.upper(),))
    conn.commit()
    conn.close()


def db_get_operator(operator_id: str) -> Optional[Dict]:
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM atc_operators WHERE operator_id=?", (operator_id.upper(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def db_get_name(operator_id: str) -> str:
    op = db_get_operator(operator_id)
    return op["name"] if op else operator_id


# ── Baseline helpers ──────────────────────────────────────────────

def voice_freshness(hnr: float, jitter_pct: float, shimmer_db: float) -> float:
    hnr_s  = min(hnr / 25.0, 1.0)
    jit_s  = max(0.0, 1.0 - (jitter_pct / 3.0))
    shim_s = max(0.0, 1.0 - (shimmer_db / 0.8))
    return round((hnr_s * 0.5 + jit_s * 0.3 + shim_s * 0.2) * 100.0, 1)


def db_load_baseline(operator_id: str) -> Dict:
    conn = get_db()
    row  = conn.execute(
        "SELECT f0_baseline, hnr_baseline, mfcc_baseline, "
        "baseline_set, baseline_freshness "
        "FROM atc_operators WHERE operator_id=?",
        (operator_id.upper(),)
    ).fetchone()
    conn.close()
    if not row or not row["baseline_set"]:
        return {"f0": None, "hnr": None, "mfcc": None,
                "freshness": 0.0, "ready": False}
    mfcc = (np.array(json.loads(row["mfcc_baseline"]), dtype=np.float32)
            if row["mfcc_baseline"] else None)
    return {"f0": row["f0_baseline"], "hnr": row["hnr_baseline"],
            "mfcc": mfcc, "freshness": row["baseline_freshness"] or 0.0,
            "ready": True}


def db_save_baseline(operator_id: str, f0: float, hnr: float,
                     mfcc: np.ndarray, freshness: float) -> None:
    conn = get_db()
    conn.execute("""
        UPDATE atc_operators
        SET f0_baseline=?, hnr_baseline=?, mfcc_baseline=?,
            baseline_freshness=?, baseline_set=1
        WHERE operator_id=?
    """, (f0, hnr, json.dumps(mfcc.tolist()), freshness, operator_id.upper()))
    conn.commit()
    conn.close()
    log.info("BASELINE SAVED | op=%-10s | F0=%.1fHz HNR=%.1fdB fresh=%.1f",
             operator_id, f0, hnr, freshness)


def db_update_baseline_if_better(
    operator_id: str, f0: float, hnr: float,
    mfcc: np.ndarray, jitter_pct: float,
    shimmer_db: float, stored_fresh: float,
) -> Tuple[bool, float]:
    new_fresh = voice_freshness(hnr, jitter_pct, shimmer_db)
    if new_fresh <= stored_fresh + BASELINE_UPDATE_MARGIN:
        return False, stored_fresh
    conn = get_db()
    conn.execute("""
        UPDATE atc_operators
        SET f0_baseline=?, hnr_baseline=?, mfcc_baseline=?, baseline_freshness=?
        WHERE operator_id=?
    """, (f0, hnr, json.dumps(mfcc.tolist()), new_fresh, operator_id.upper()))
    conn.commit()
    conn.close()
    log.info("BASELINE UPGRADE | op=%-10s | %.1f -> %.1f fresh",
             operator_id, stored_fresh, new_fresh)
    return True, new_fresh


# ── Session helpers ───────────────────────────────────────────────

def db_start_session(operator_id: str, operator_name: str) -> int:
    """Create new duty session. Returns session_id."""
    conn = get_db()
    # Close any lingering active sessions first
    conn.execute("""
        UPDATE duty_sessions SET status='completed', end_time=datetime('now')
        WHERE operator_id=? AND status='active'
    """, (operator_id.upper(),))
    cursor = conn.execute("""
        INSERT INTO duty_sessions (operator_id, operator_name, start_time, status)
        VALUES (?, ?, datetime('now'), 'active')
    """, (operator_id.upper(), operator_name))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    log.info("DUTY START | op=%-10s | session_id=%d", operator_id, session_id)
    return session_id


def db_end_session(operator_id: str) -> Optional[Dict]:
    """Close active session and return summary."""
    conn = get_db()
    session = conn.execute(
        "SELECT * FROM duty_sessions WHERE operator_id=? AND status='active'",
        (operator_id.upper(),)
    ).fetchone()
    if not session:
        conn.close()
        return None

    session_id = session["id"]

    # Compute summary from stamps
    stamps = conn.execute(
        "SELECT avg_score, zone FROM fatigue_stamps WHERE session_id=?",
        (session_id,)
    ).fetchall()

    avg_score    = float(np.mean([s["avg_score"] for s in stamps])) if stamps else 0.0
    max_score    = float(max([s["avg_score"] for s in stamps])) if stamps else 0.0
    green_stamps = sum(1 for s in stamps if s["zone"] == "GREEN")
    yellow_stamps= sum(1 for s in stamps if s["zone"] == "YELLOW")
    red_stamps   = sum(1 for s in stamps if s["zone"] == "RED")

    # Duration in minutes
    start_dt    = datetime.fromisoformat(session["start_time"])
    duration_min= (datetime.utcnow() - start_dt).total_seconds() / 60.0

    conn.execute("""
        UPDATE duty_sessions
        SET status='completed', end_time=datetime('now'),
            duration_min=?, avg_score=?, max_score=?,
            stamp_count=?, green_stamps=?, yellow_stamps=?, red_stamps=?
        WHERE id=?
    """, (round(duration_min, 1), round(avg_score, 1), round(max_score, 1),
          len(stamps), green_stamps, yellow_stamps, red_stamps, session_id))
    conn.commit()
    conn.close()
    log.info("DUTY END | op=%-10s | duration=%.1fmin avg=%.1f",
             operator_id, duration_min, avg_score)
    return {
        "session_id":   session_id,
        "duration_min": round(duration_min, 1),
        "avg_score":    round(avg_score, 1),
        "max_score":    round(max_score, 1),
        "stamp_count":  len(stamps),
        "green_stamps": green_stamps,
        "yellow_stamps":yellow_stamps,
        "red_stamps":   red_stamps,
    }


def db_get_active_session(operator_id: str) -> Optional[int]:
    conn = get_db()
    row  = conn.execute(
        "SELECT id FROM duty_sessions WHERE operator_id=? AND status='active'",
        (operator_id.upper(),)
    ).fetchone()
    conn.close()
    return row["id"] if row else None


# ── Stamp helpers ─────────────────────────────────────────────────

def db_log_stamp(
    operator_id: str, operator_name: str, session_id: Optional[int],
    avg_score: float, zone: str,
    green_count: int, yellow_count: int, red_count: int,
    speech_chunks: int, duration_sec: int, alert_message: str,
) -> None:
    try:
        today = date.today().isoformat()
        conn  = get_db()
        conn.execute("""
            INSERT INTO fatigue_stamps
            (operator_id, operator_name, session_id, timestamp, date,
             avg_score, zone, green_count, yellow_count, red_count,
             speech_chunks, duration_sec, alert_message)
            VALUES (?,?,?,datetime('now'),?,?,?,?,?,?,?,?,?)
        """, (operator_id.upper(), operator_name, session_id, today,
              avg_score, zone, green_count, yellow_count, red_count,
              speech_chunks, duration_sec, alert_message))
        conn.commit()
        conn.close()
        log.info("STAMP | op=%-10s | zone=%-6s | avg=%.1f | speech=%d chunks",
                 operator_id, zone, avg_score, speech_chunks)
    except Exception as exc:
        log.error("Stamp write error: %s", exc)


def db_log_eye_warning(
    operator_id: str, operator_name: str,
    session_id: Optional[int], duration_sec: float, elapsed: str,
) -> None:
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO eye_warnings
            (operator_id, operator_name, session_id, timestamp, duration_sec, session_elapsed)
            VALUES (?,?,?,datetime('now'),?,?)
        """, (operator_id.upper(), operator_name, session_id, duration_sec, elapsed))
        conn.commit()
        conn.close()
        log.info("EYE WARN | op=%-10s | closed=%.1fs", operator_id, duration_sec)
    except Exception as exc:
        log.error("Eye warning write error: %s", exc)


# ── Alert messages ────────────────────────────────────────────────

def generate_alert(zone: str, score: float) -> str:
    if zone == "GREEN":
        return "Vocal biomarkers nominal. No fatigue indicators detected."
    if zone == "YELLOW":
        return (
            f"Elevated fatigue detected (Score: {score:.1f}/100). "
            "Recommended: 15-minute rest break before next sector. "
            "Hydration advised. Supervisor notified."
        )
    return (
        f"High fatigue level detected (Score: {score:.1f}/100). "
        "Immediate rest strongly advised. "
        "Please consult the duty medical officer before resuming operations."
    )


# ── Dashboard and report helpers ──────────────────────────────────

def db_dashboard_today() -> List[Dict]:
    """All operators with today's status."""
    today = date.today().isoformat()
    conn  = get_db()
    operators = conn.execute(
        "SELECT operator_id, name FROM atc_operators ORDER BY name"
    ).fetchall()

    result = []
    for op in operators:
        oid  = op["operator_id"]
        name = op["name"]

        # Active session?
        session = conn.execute(
            "SELECT id, start_time FROM duty_sessions "
            "WHERE operator_id=? AND status='active'",
            (oid,)
        ).fetchone()

        # Today's stamps
        stamps = conn.execute(
            "SELECT avg_score, zone FROM fatigue_stamps "
            "WHERE operator_id=? AND date=? ORDER BY id DESC LIMIT 20",
            (oid, today)
        ).fetchall()

        # Last eye warning today
        eye_warn = conn.execute(
            "SELECT timestamp FROM eye_warnings "
            "WHERE operator_id=? AND date(timestamp)=? ORDER BY id DESC LIMIT 1",
            (oid, today)
        ).fetchone()

        last_score = stamps[0]["avg_score"] if stamps else None
        last_zone  = stamps[0]["zone"]      if stamps else None

        result.append({
            "operator_id":      oid,
            "name":             name,
            "on_duty":          session is not None,
            "duty_start":       session["start_time"] if session else None,
            "last_score":       last_score,
            "last_zone":        last_zone,
            "today_stamps":     len(stamps),
            "today_red":        sum(1 for s in stamps if s["zone"] == "RED"),
            "today_yellow":     sum(1 for s in stamps if s["zone"] == "YELLOW"),
            "last_eye_warning": eye_warn["timestamp"] if eye_warn else None,
        })

    conn.close()
    return result


def db_daily_report(operator_id: str, report_date: str) -> Dict:
    """Generate daily report for one operator."""
    conn = get_db()

    stamps = conn.execute(
        "SELECT * FROM fatigue_stamps WHERE operator_id=? AND date=? ORDER BY id ASC",
        (operator_id.upper(), report_date)
    ).fetchall()

    sessions = conn.execute(
        "SELECT * FROM duty_sessions "
        "WHERE operator_id=? AND date(start_time)=? ORDER BY id ASC",
        (operator_id.upper(), report_date)
    ).fetchall()

    eye_warnings = conn.execute(
        "SELECT * FROM eye_warnings "
        "WHERE operator_id=? AND date(timestamp)=? ORDER BY id ASC",
        (operator_id.upper(), report_date)
    ).fetchall()

    conn.close()

    if not stamps:
        return {
            "operator_id": operator_id.upper(),
            "date":        report_date,
            "status":      "no_data",
            "message":     "No duty data recorded for this date.",
        }

    scores        = [s["avg_score"] for s in stamps]
    daily_avg     = round(float(np.mean(scores)), 1)
    daily_max     = round(float(max(scores)), 1)
    green_count   = sum(1 for s in stamps if s["zone"] == "GREEN")
    yellow_count  = sum(1 for s in stamps if s["zone"] == "YELLOW")
    red_count     = sum(1 for s in stamps if s["zone"] == "RED")
    total_duty_min= sum(s["duration_min"] for s in sessions) if sessions else 0

    zone = classify_zone(daily_avg)

    suggestion = _daily_suggestion(daily_avg, red_count, yellow_count, len(stamps))

    return {
        "operator_id":      operator_id.upper(),
        "operator_name":    db_get_name(operator_id),
        "date":             report_date,
        "status":           "ok",
        "daily_avg_score":  daily_avg,
        "daily_max_score":  daily_max,
        "overall_zone":     zone["zone"],
        "total_stamps":     len(stamps),
        "green_stamps":     green_count,
        "yellow_stamps":    yellow_count,
        "red_stamps":       red_count,
        "total_duty_min":   round(total_duty_min, 1),
        "sessions":         len(sessions),
        "eye_warnings":     len(eye_warnings),
        "suggestion":       suggestion,
        "stamps_timeline":  [
            {"time": s["timestamp"], "score": s["avg_score"], "zone": s["zone"]}
            for s in stamps
        ],
    }


def db_weekly_report(operator_id: str) -> Dict:
    """Generate 7-day rolling report."""
    today  = date.today()
    days   = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]

    conn   = get_db()
    daily  = []

    for d in days:
        stamps = conn.execute(
            "SELECT avg_score, zone FROM fatigue_stamps "
            "WHERE operator_id=? AND date=?",
            (operator_id.upper(), d)
        ).fetchall()

        if stamps:
            scores = [s["avg_score"] for s in stamps]
            daily.append({
                "date":         d,
                "avg_score":    round(float(np.mean(scores)), 1),
                "max_score":    round(float(max(scores)), 1),
                "stamps":       len(stamps),
                "red_count":    sum(1 for s in stamps if s["zone"] == "RED"),
                "yellow_count": sum(1 for s in stamps if s["zone"] == "YELLOW"),
                "has_data":     True,
            })
        else:
            daily.append({
                "date": d, "avg_score": None, "max_score": None,
                "stamps": 0, "red_count": 0, "yellow_count": 0,
                "has_data": False,
            })

    conn.close()

    # Trend calculation
    scores_with_data = [d["avg_score"] for d in daily if d["has_data"]]
    trend = "INSUFFICIENT_DATA"
    suggestion = "Not enough data for trend analysis yet."

    if len(scores_with_data) >= 4:
        first_half = float(np.mean(scores_with_data[:len(scores_with_data)//2]))
        second_half= float(np.mean(scores_with_data[len(scores_with_data)//2:]))
        diff       = second_half - first_half

        if diff > 5:
            trend = "WORSENING"
            suggestion = (
                "Fatigue trend is increasing over the week. "
                "Review shift rotation schedule. "
                "Consider scheduling an additional rest day."
            )
        elif diff < -5:
            trend = "IMPROVING"
            suggestion = (
                "Fatigue levels are improving. "
                "Current rest and schedule pattern is working well. "
                "Maintain current rotation."
            )
        else:
            trend = "STABLE"
            avg_all = float(np.mean(scores_with_data))
            if avg_all >= ZONE_RED:
                suggestion = (
                    "Score is stable but critically high. "
                    "Immediate schedule review recommended. "
                    "Medical consultation advised."
                )
            elif avg_all >= ZONE_YELLOW:
                suggestion = (
                    "Score is stable in elevated range. "
                    "Monitor closely. Consider shift length adjustment."
                )
            else:
                suggestion = (
                    "Performance is stable and healthy. "
                    "No immediate action required."
                )

    total_red = sum(d["red_count"] for d in daily)
    if total_red >= 3:
        suggestion += (
            f" Note: {total_red} critical fatigue events this week — "
            "medical review strongly recommended."
        )

    return {
        "operator_id":   operator_id.upper(),
        "operator_name": db_get_name(operator_id),
        "period_start":  days[0],
        "period_end":    days[-1],
        "trend":         trend,
        "suggestion":    suggestion,
        "total_red":     total_red,
        "days_with_data":len(scores_with_data),
        "daily":         daily,
    }


def _daily_suggestion(avg: float, red: int, yellow: int, total: int) -> str:
    if total == 0:
        return "No data recorded."
    if red >= 2:
        return (
            f"{red} critical fatigue periods detected today. "
            "Mandatory rest before next duty. Medical officer consultation advised."
        )
    if avg >= ZONE_RED:
        return (
            "Overall fatigue level critical today. "
            "Extended rest period required before return to duty."
        )
    if avg >= ZONE_YELLOW or yellow >= 2:
        return (
            "Elevated fatigue detected during duty. "
            "Ensure adequate rest before next shift. "
            "Review shift length if pattern continues."
        )
    return (
        "Fatigue levels within acceptable range today. "
        "Maintain current rest schedule."
    )


def classify_zone(score: float) -> Dict:
    if score < ZONE_YELLOW:
        return {"zone": "GREEN",  "label": "FIT FOR DUTY",
                "message": "Biomarkers nominal. No fatigue indicators detected."}
    if score < ZONE_RED:
        return {"zone": "YELLOW", "label": "ELEVATED FATIGUE",
                "message": "Fatigue rising. Rest break recommended before next sector."}
    return {"zone": "RED",    "label": "REST REQUIRED",
            "message": "High fatigue. Rest and consult duty medical officer."}


# ── Pydantic models ───────────────────────────────────────────────

class OperatorCreate(BaseModel):
    operator_id: str
    name:        str


class SessionStart(BaseModel):
    operator_id: str


class SessionEnd(BaseModel):
    operator_id: str


# ── App ───────────────────────────────────────────────────────────

app = FastAPI(title="AeroCalm AI ATC", version="6.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    try:
        with open("index.html", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h3>index.html not found.</h3>", status_code=404)


@app.get("/health", tags=["Ops"])
def health():
    return {
        "status":      "online",
        "version":     "6.0.0",
        "mode":        "ATC-Only Voice Scoring",
        "epoch_sec":   EPOCH_DURATION_SEC,
        "eye_warn_sec":EYE_CLOSE_WARNING_SEC,
    }


# ── Operator API ──────────────────────────────────────────────────

@app.post("/api/operators", tags=["Operators"])
def add_operator(p: OperatorCreate):
    if not p.operator_id.strip() or not p.name.strip():
        raise HTTPException(400, "operator_id and name are required.")
    return db_add_operator(p.operator_id, p.name)


@app.get("/api/operators", tags=["Operators"])
def list_operators():
    return {"operators": db_list_operators()}


@app.delete("/api/operators/{operator_id}", tags=["Operators"])
def delete_operator(operator_id: str):
    db_delete_operator(operator_id)
    return {"status": "deleted", "operator_id": operator_id.upper()}


# ── Session API ───────────────────────────────────────────────────

@app.post("/api/session/start", tags=["Session"])
def start_duty(p: SessionStart):
    op = db_get_operator(p.operator_id)
    if not op:
        raise HTTPException(404, f"Operator '{p.operator_id}' not registered.")
    session_id = db_start_session(p.operator_id, op["name"])
    return {"status": "ok", "session_id": session_id,
            "operator_id": p.operator_id.upper(), "operator_name": op["name"]}


@app.post("/api/session/end", tags=["Session"])
def end_duty(p: SessionEnd):
    summary = db_end_session(p.operator_id)
    if not summary:
        raise HTTPException(404, "No active session found.")
    return {"status": "ok", **summary}


@app.get("/api/session/active/{operator_id}", tags=["Session"])
def get_active_session(operator_id: str):
    session_id = db_get_active_session(operator_id)
    return {"active": session_id is not None, "session_id": session_id}


# ── Dashboard API ─────────────────────────────────────────────────

@app.get("/api/dashboard/today", tags=["Dashboard"])
def dashboard_today():
    return {"operators": db_dashboard_today(),
            "generated_at": datetime.utcnow().isoformat()}


@app.get("/api/dashboard/alerts", tags=["Dashboard"])
def recent_alerts(limit: int = 30):
    conn = get_db()
    stamps = conn.execute("""
        SELECT operator_id, operator_name, timestamp, avg_score, zone, alert_message
        FROM fatigue_stamps
        WHERE zone IN ('YELLOW','RED')
        ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    eye = conn.execute("""
        SELECT operator_id, operator_name, timestamp, duration_sec
        FROM eye_warnings ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return {
        "fatigue_alerts": [dict(r) for r in stamps],
        "eye_warnings":   [dict(r) for r in eye],
    }


# ── Reports API ───────────────────────────────────────────────────

@app.get("/api/reports/daily/{operator_id}", tags=["Reports"])
def daily_report(operator_id: str, report_date: str = ""):
    if not report_date:
        report_date = date.today().isoformat()
    return db_daily_report(operator_id, report_date)


@app.get("/api/reports/weekly/{operator_id}", tags=["Reports"])
def weekly_report(operator_id: str):
    return db_weekly_report(operator_id)


# ── Stamps API ────────────────────────────────────────────────────

@app.get("/api/stamps/{operator_id}", tags=["Stamps"])
def get_stamps(operator_id: str, limit: int = 20):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM fatigue_stamps WHERE operator_id=? ORDER BY id DESC LIMIT ?",
        (operator_id.upper(), limit)
    ).fetchall()
    conn.close()
    return {"stamps": [dict(r) for r in rows]}


# ── Vision engine ─────────────────────────────────────────────────

def _euclidean(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _ear_calc(landmarks, indices) -> float:
    p = [landmarks[i] for i in indices]
    vert  = _euclidean(p[1], p[5]) + _euclidean(p[2], p[4])
    horiz = 2.0 * _euclidean(p[0], p[3])
    return vert / horiz if horiz > 1e-6 else 0.30


def process_video_frame(frame_b64: str, face_mesh) -> Tuple[float, bool]:
    """Decode JPEG, run MediaPipe, return (ear, face_detected)."""
    try:
        arr    = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
        frame  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return 0.0, False
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return 0.0, False
        lm  = result.multi_face_landmarks[0].landmark
        ear = (_ear_calc(lm, LEFT_EYE) + _ear_calc(lm, RIGHT_EYE)) / 2.0
        return round(ear, 4), True
    except Exception as exc:
        log.debug("Video frame: %s", exc)
        return 0.0, False


# ── Noise preprocessing ───────────────────────────────────────────

def _spectral_subtraction(audio: np.ndarray) -> np.ndarray:
    if len(audio) < SS_N_FFT * SS_NOISE_FRAMES:
        return audio
    hop = SS_N_FFT // 4
    try:
        D         = librosa.stft(audio, n_fft=SS_N_FFT, hop_length=hop)
        noise_pwr = np.mean(np.abs(D[:, :SS_NOISE_FRAMES]) ** 2,
                            axis=1, keepdims=True)
        sig_pwr   = np.abs(D) ** 2
        clean_pwr = np.maximum(sig_pwr - SS_ALPHA * noise_pwr, SS_BETA * sig_pwr)
        D_clean   = np.sqrt(clean_pwr) * np.exp(1j * np.angle(D))
        return librosa.istft(D_clean, hop_length=hop,
                             length=len(audio)).astype(np.float32)
    except Exception:
        return audio


# ── Voice biomarker engine (9 markers, voice-only) ────────────────

def _dynamic_vad(audio: np.ndarray) -> np.ndarray:
    gate   = max(0.015, float(np.mean(np.abs(audio))) * 1.5)
    active = audio[np.abs(audio) > gate]
    return active if len(active) >= SAMPLE_RATE * 0.5 else np.array([], dtype=np.float32)


def _clean_f0(audio: np.ndarray) -> np.ndarray:
    f0  = librosa.yin(audio, fmin=55, fmax=285, sr=SAMPLE_RATE)
    f0r = f0[(f0 > 55) & (f0 < 285)]
    if len(f0r) < 4:
        return np.array([], dtype=np.float32)
    med, sig = float(np.median(f0r)), float(np.std(f0r))
    if sig < 1e-6:
        return f0r
    f0c = f0r[np.abs(f0r - med) < 2.0 * sig]
    return f0c if len(f0c) >= 4 else np.array([], dtype=np.float32)


def _bio_rms(active):
    rms = float(np.sqrt(np.mean(active ** 2)))
    return rms, float(np.clip(1.0 - rms / RMS_REFERENCE, 0.0, 1.0)) * 100.0


def _bio_flatness(audio):
    stft = np.abs(librosa.stft(audio))
    val  = float(np.mean(librosa.feature.spectral_flatness(S=stft)))
    return float(np.clip(val / 0.5, 0.0, 1.0)) * 100.0


def _bio_jitter(audio):
    f0c = _clean_f0(audio)
    if len(f0c) < 5:
        return 0.0, 0.0
    periods = 1.0 / f0c
    mean_T  = float(np.mean(periods))
    if mean_T < 1e-9:
        return 0.0, 0.0
    pert = [abs(periods[i] - (periods[i-1]+periods[i]+periods[i+1])/3.0)
            for i in range(1, len(periods)-1)]
    if not pert:
        return 0.0, 0.0
    rap = float(np.mean(pert)) / mean_T * 100.0
    return round(rap, 3), round(float(np.clip(rap / JITTER_SATURATION, 0.0, 1.0)) * 100.0, 2)


def _bio_shimmer(audio):
    frms = librosa.feature.rms(y=audio, hop_length=256)[0]
    vrms = frms[frms > SILENCE_GATE]
    if len(vrms) < 3:
        return 0.0, 0.0
    eps    = 1e-10
    ratios = np.abs(vrms[:-1]) / (np.abs(vrms[1:]) + eps)
    ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
    if len(ratios) == 0:
        return 0.0, 0.0
    shim = float(np.mean(np.abs(20.0 * np.log10(ratios + eps))))
    return round(shim, 4), round(float(np.clip(shim / SHIMMER_SATURATION, 0.0, 1.0)) * 100.0, 2)


def _bio_f0_drop(audio, baseline):
    f0c = _clean_f0(audio)
    if len(f0c) < 4:
        return 0.0, 0.0
    curr = float(np.mean(f0c))
    if baseline is None:
        return curr, 0.0
    drop = baseline - curr
    if drop <= 0:
        return curr, 0.0
    return curr, round(float(np.clip(drop / F0_DROP_SATURATION, 0.0, 1.0)) * 100.0, 2)


def _bio_formant(audio):
    cents = librosa.feature.spectral_centroid(
        y=audio, sr=SAMPLE_RATE, hop_length=256)[0]
    if len(cents) < 4:
        return 0.0
    span = CENTROID_ALERT_STD - CENTROID_FATIGUED_STD
    return round(float(np.clip(
        (CENTROID_ALERT_STD - float(np.std(cents))) / span, 0.0, 1.0
    )) * 100.0, 2)


def _bio_hnr(audio, hnr_baseline):
    f0c = _clean_f0(audio)
    if len(f0c) < 4:
        return 0.0, 0.0, 0.0
    mean_f0 = float(np.mean(f0c))
    T0 = int(round(SAMPLE_RATE / mean_f0)) if mean_f0 > 1 else 0
    if T0 < 2 or T0 >= len(audio) // 2:
        return 0.0, 0.0, 0.0
    try:
        acf    = np.correlate(audio, audio, mode='full')[len(audio)-1:]
        r0     = float(acf[0])
        if r0 < 1e-10:
            return 0.0, 0.0, 0.0
        rT     = float(np.clip(float(acf[T0]) / r0, -0.9999, 0.9999))
        hnr_db = float(np.clip(
            10.0 * np.log10(max(rT, 1e-10) / (1.0 - max(rT, 1e-10))),
            0.0, 40.0
        )) if rT > 0 else 0.0
        abs_s  = float(np.clip(1.0 - hnr_db / HNR_HEALTHY_DB, 0.0, 1.0)) * 100.0
        rel_s  = float(np.clip(
            (hnr_baseline - hnr_db) / HNR_BASELINE_DROP, 0.0, 1.0
        )) * 100.0 if hnr_baseline and (hnr_baseline - hnr_db) > 0 else abs_s
        return round(hnr_db, 2), round(abs_s, 2), round(rel_s, 2)
    except Exception:
        return 0.0, 0.0, 0.0


def _bio_speech_rate(audio):
    hop, fl = 512, 1024
    try:
        energy  = librosa.feature.rms(y=audio, hop_length=hop, frame_length=fl)[0]
        zcr     = librosa.feature.zero_crossing_rate(
            audio, hop_length=hop, frame_length=fl)[0]
        thresh  = float(np.mean(energy)) * 0.5
        voiced  = (energy > thresh) & (zcr < 0.15)
        sr_r    = int(np.sum(voiced)) / max(len(energy), 1)
        smooth  = np.convolve(energy, np.ones(5) / 5.0, mode='same')
        peaks,_ = find_peaks(smooth, height=thresh, distance=3)
        syl_r   = float(len(peaks)) / max(len(audio) / SAMPLE_RATE, 0.1)
        p_s     = float(np.clip(1.0 - sr_r / SPEECH_RATIO_NORMAL, 0.0, 1.0)) * 100.0
        r_s     = float(np.clip(1.0 - syl_r / SYLLABLE_RATE_NORMAL, 0.0, 1.0)) * 100.0
        return (round(sr_r, 3), round(syl_r, 2),
                round(SPEECH_PAUSE_FACTOR * p_s + (1 - SPEECH_PAUSE_FACTOR) * r_s, 2))
    except Exception:
        return 0.0, 0.0, 0.0


def _bio_mfcc(audio, mfcc_baseline):
    try:
        mfcc_mean = np.mean(
            librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE,
                                 n_mfcc=N_MFCC, hop_length=256),
            axis=1
        ).astype(np.float32)
        if mfcc_baseline is None:
            return mfcc_mean, 0.0
        nc, nb = np.linalg.norm(mfcc_mean), np.linalg.norm(mfcc_baseline)
        if nc < 1e-6 or nb < 1e-6:
            return mfcc_mean, 0.0
        dist = 1.0 - float(np.clip(
            np.dot(mfcc_mean, mfcc_baseline) / (nc * nb), -1.0, 1.0
        ))
        return mfcc_mean, round(
            float(np.clip(dist / MFCC_SATURATION_DIST, 0.0, 1.0)) * 100.0, 2
        )
    except Exception:
        return np.zeros(N_MFCC, dtype=np.float32), 0.0


def extract_voice_score(
    raw_audio: np.ndarray,
    f0_bl: Optional[float],
    hnr_bl: Optional[float],
    mfcc_bl: Optional[np.ndarray],
) -> Tuple[float, float, bool, float, float, np.ndarray, Dict]:
    """
    Full 9-biomarker voice pipeline.
    Returns: (score, confidence, speech_detected,
              current_f0, hnr_db, jitter_pct, shimmer_db, mfcc_mean, sub_scores)
    """
    W_RMS=0.10; W_FLAT=0.05; W_JIT=0.18; W_SHIM=0.18
    W_F0=0.10;  W_FORM=0.05; W_HNR=0.14; W_SR=0.12; W_MFCC=0.08

    EMPTY = (0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0,
             np.zeros(N_MFCC, dtype=np.float32), {})

    if len(raw_audio) < SAMPLE_RATE // 2:
        return EMPTY

    try:
        audio  = _spectral_subtraction(raw_audio)
        active = _dynamic_vad(audio)
        if len(active) == 0:
            return EMPTY

        rms, rms_s = _bio_rms(active)
        if rms < SILENCE_GATE:
            return EMPTY

        flat_s               = _bio_flatness(audio)
        jit_pct, jit_s       = _bio_jitter(audio)
        shim_db, shim_s      = _bio_shimmer(audio)
        curr_f0, f0_s        = _bio_f0_drop(audio, f0_bl)
        form_s               = _bio_formant(audio)
        hnr_db, _, hnr_s     = _bio_hnr(audio, hnr_bl)
        sr_r, syl_r, sr_s    = _bio_speech_rate(audio)
        mfcc_mean, mfcc_s    = _bio_mfcc(audio, mfcc_bl)

        composite = (rms_s*W_RMS + flat_s*W_FLAT + jit_s*W_JIT + shim_s*W_SHIM +
                     f0_s*W_F0 + form_s*W_FORM + hnr_s*W_HNR + sr_s*W_SR + mfcc_s*W_MFCC)

        score = round(float(np.clip(composite, 0.0, 100.0)), 2)
        conf  = round(float(np.clip(
            (len(active) / len(audio)) / CONFIDENCE_SATURATION, 0.0, 1.0
        )) * 100.0, 1)

        sub = {
            "rms": round(rms_s, 2), "flatness": round(flat_s, 2),
            "jitter_rap_pct": round(jit_pct, 3), "jitter": round(jit_s, 2),
            "shimmer_db": round(shim_db, 4), "shimmer": round(shim_s, 2),
            "f0_drop": round(f0_s, 2), "current_f0": round(curr_f0, 2),
            "formant_sluggishness": round(form_s, 2),
            "hnr_db": round(hnr_db, 2), "hnr": round(hnr_s, 2),
            "speech_ratio": round(sr_r, 3), "syllable_rate": round(syl_r, 2),
            "speech_rate": round(sr_s, 2), "mfcc_deviation": round(mfcc_s, 2),
        }
        return (score, conf, True, curr_f0, hnr_db,
                jit_pct, shim_db, mfcc_mean, sub)

    except Exception:
        log.exception("Voice pipeline error.")
        return EMPTY


# ── Session state ─────────────────────────────────────────────────

class ATCSession:
    """
    Per-WebSocket session for one ATC operator.

    Calibration:
      If no baseline in DB → CALIBRATION mode.
      Collect F0, HNR, MFCC for BASELINE_CHUNKS chunks.
      Save to DB. Scoring begins immediately after.

    Epoch stamping (5 minutes):
      Collect scores only when speech is detected.
      Every 5 minutes, average collected scores and stamp to DB.
      Requires MIN_SPEECH_FOR_STAMP chunks to generate a stamp.

    Eye closure warning:
      Track consecutive frames where EAR < threshold.
      After EYE_CLOSE_FRAMES_THRESH consecutive closed frames → warning.
      Warning logged to DB. Counter resets on eye opening.
    """

    def __init__(self, op_id: str, op_name: str, session_id: Optional[int]) -> None:
        self.op_id      = op_id
        self.op_name    = op_name
        self.session_id = session_id

        # Audio ring buffer
        self._buf       = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self._write_pos = 0
        self._filled    = False

        self.score_history: Deque[float] = collections.deque(maxlen=SMOOTH_WINDOW)
        self.chunk_count = 0
        self._t0 = time.monotonic()

        # Baselines
        self.f0_baseline:      Optional[float]      = None
        self.hnr_baseline:     Optional[float]      = None
        self.mfcc_baseline:    Optional[np.ndarray] = None
        self.stored_freshness: float                = 0.0
        self.calibration_done: bool                 = False
        self._baseline_saved:  bool                 = False
        self._f0_s:   list = []
        self._hnr_s:  list = []
        self._mfcc_s: list = []
        self._last_jitter_pct: float = 0.0
        self._last_shimmer_db: float = 0.0

        # Epoch
        self._epoch_scores:  list  = []
        self._epoch_green:   int   = 0
        self._epoch_yellow:  int   = 0
        self._epoch_red:     int   = 0
        self._epoch_chunks:  int   = 0
        self._epoch_start:   float = time.monotonic()

        # Eye closure warning (no score contribution)
        self._eye_closed_frames:  int   = 0
        self._eye_open_frames:    int   = 0
        self._eye_warning_sent:   bool  = False
        self.face_detected:       bool  = False
        self.ear_raw:             float = 0.0
        self.vision_frames:       int   = 0

        # Alert throttle
        self.last_zone:     Optional[str] = None
        self.last_log_time: float         = 0.0

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def load_baseline(self) -> None:
        stored = db_load_baseline(self.op_id)
        if stored["ready"]:
            self.f0_baseline      = stored["f0"]
            self.hnr_baseline     = stored["hnr"]
            self.mfcc_baseline    = stored["mfcc"]
            self.stored_freshness = stored["freshness"]
            self.calibration_done = True
            self._baseline_saved  = True
            log.info("BASELINE LOADED | op=%-10s | fresh=%.1f",
                     self.op_id, self.stored_freshness)
        else:
            log.info("CALIBRATION     | op=%-10s | no baseline yet", self.op_id)

    def ingest_audio(self, raw_bytes: bytes) -> None:
        if not raw_bytes or len(raw_bytes) % 4 != 0:
            return
        try:
            data = np.frombuffer(raw_bytes, dtype=np.float32)
            n, end = len(data), self._write_pos + len(data)
            if end <= BUFFER_SIZE:
                self._buf[self._write_pos:end] = data
            else:
                split = BUFFER_SIZE - self._write_pos
                self._buf[self._write_pos:] = data[:split]
                self._buf[:n - split]       = data[split:]
            self._write_pos = (self._write_pos + n) % BUFFER_SIZE
            if end >= BUFFER_SIZE:
                self._filled = True
            self.chunk_count += 1
        except ValueError as exc:
            log.error("[%s] Audio decode: %s", self.op_id, exc)

    def ingest_video(self, frame_b64: str) -> Optional[Dict]:
        """
        Process one video frame.
        Returns eye_warning dict if warning triggered, else None.
        Eye closure logic:
          - Track consecutive closed frames (EAR < threshold)
          - After EYE_CLOSE_FRAMES_THRESH closed frames → warning
          - Reset after EYE_OPEN_RESET_FRAMES open frames
        """
        ear, found = process_video_frame(frame_b64, self._face_mesh)
        self.ear_raw      = ear
        self.face_detected = found
        self.vision_frames += 1

        if not found:
            # No face — slowly reset counters
            self._eye_closed_frames = max(0, self._eye_closed_frames - 1)
            self._eye_warning_sent  = False
            return None

        if ear < EAR_BLINK_THRESH:
            # Eye closed/closing
            self._eye_closed_frames += 1
            self._eye_open_frames    = 0

            if (self._eye_closed_frames >= EYE_CLOSE_FRAMES_THRESH and
                    not self._eye_warning_sent):
                # Trigger warning
                self._eye_warning_sent = True
                duration_sec = round(
                    self._eye_closed_frames / 2.0, 1
                )  # 2fps assumption
                return {
                    "type":         "eye_warning",
                    "duration_sec": duration_sec,
                    "message":      (
                        f"Eyes closed for {duration_sec:.1f} seconds detected. "
                        "Please remain alert."
                    ),
                }
        else:
            # Eye open
            self._eye_open_frames += 1
            if self._eye_open_frames >= EYE_OPEN_RESET_FRAMES:
                self._eye_closed_frames = 0
                self._eye_warning_sent  = False
                self._eye_open_frames   = 0

        return None

    def compute_voice(self) -> Tuple[float, float, bool, Dict]:
        """
        Run voice pipeline.
        Returns (score, confidence, calibration_done, sub_scores).
        During calibration → score=0, calibration_done=False.
        """
        audio = self._buf if self._filled else self._buf[:self._write_pos]

        result = extract_voice_score(
            audio, self.f0_baseline, self.hnr_baseline, self.mfcc_baseline
        )
        score, conf, speech_detected = result[0], result[1], result[2]
        curr_f0, hnr_db, jit_pct, shim_db, mfcc_mean, sub = (
            result[3], result[4], result[5], result[6], result[7], result[8]
        )

        self._last_jitter_pct = jit_pct
        self._last_shimmer_db = shim_db

        # Calibration phase
        if not self._baseline_saved:
            if speech_detected:
                if curr_f0 > 0:
                    self._f0_s.append(curr_f0)
                if hnr_db > 0:
                    self._hnr_s.append(hnr_db)
                if np.any(mfcc_mean != 0):
                    self._mfcc_s.append(mfcc_mean.copy())

            if (len(self._f0_s) >= BASELINE_CHUNKS and
                    len(self._hnr_s) >= BASELINE_CHUNKS and
                    len(self._mfcc_s) >= BASELINE_CHUNKS):

                f0_bl   = float(np.mean(self._f0_s))
                hnr_bl  = float(np.mean(self._hnr_s))
                mfcc_bl = np.mean(np.stack(self._mfcc_s), axis=0).astype(np.float32)
                fresh   = voice_freshness(hnr_bl, float(np.mean([jit_pct or 1.0])),
                                          float(np.mean([shim_db or 0.3])))

                self.f0_baseline      = f0_bl
                self.hnr_baseline     = hnr_bl
                self.mfcc_baseline    = mfcc_bl
                self.stored_freshness = fresh
                self._baseline_saved  = True
                self.calibration_done = True

                self._f0_s.clear()
                self._hnr_s.clear()
                self._mfcc_s.clear()

                db_save_baseline(self.op_id, f0_bl, hnr_bl, mfcc_bl, fresh)

            return 0.0, 0.0, False, {}

        # Scoring active
        if speech_detected:
            self.score_history.append(score)

        smoothed = round(float(np.mean(self.score_history)), 2) if self.score_history else 0.0
        return smoothed, conf, True, sub

    def try_adaptive_update(self) -> bool:
        if not self._baseline_saved or self.mfcc_baseline is None:
            return False
        updated, new_fresh = db_update_baseline_if_better(
            self.op_id, self.f0_baseline or 0.0,
            self.hnr_baseline or 0.0, self.mfcc_baseline,
            self._last_jitter_pct, self._last_shimmer_db,
            self.stored_freshness,
        )
        if updated:
            self.stored_freshness = new_fresh
        return updated

    def tick_epoch(self, score: float, zone: str, speech: bool) -> Optional[Dict]:
        """
        Add score to epoch window (only if speech detected).
        At 5-minute mark, compute and return stamp dict (or None if insufficient).
        """
        if speech and score > 0:
            self._epoch_scores.append(score)
            self._epoch_chunks += 1
            if zone == "GREEN":
                self._epoch_green  += 1
            elif zone == "YELLOW":
                self._epoch_yellow += 1
            else:
                self._epoch_red    += 1

        elapsed = time.monotonic() - self._epoch_start
        if elapsed < EPOCH_DURATION_SEC:
            return None

        # Time to stamp
        stamp = None
        if self._epoch_chunks >= MIN_SPEECH_FOR_STAMP:
            avg_score = round(float(np.mean(self._epoch_scores)), 2)
            stamp = {
                "avg_score":    avg_score,
                "zone":         classify_zone(avg_score)["zone"],
                "label":        classify_zone(avg_score)["label"],
                "green_count":  self._epoch_green,
                "yellow_count": self._epoch_yellow,
                "red_count":    self._epoch_red,
                "speech_chunks":self._epoch_chunks,
                "duration_sec": int(elapsed),
            }

        # Reset epoch
        self._epoch_scores  = []
        self._epoch_green   = 0
        self._epoch_yellow  = 0
        self._epoch_red     = 0
        self._epoch_chunks  = 0
        self._epoch_start   = time.monotonic()
        return stamp

    @property
    def calibration_progress(self) -> float:
        n = max(len(self._f0_s), len(self._hnr_s), len(self._mfcc_s))
        return round(min(n / BASELINE_CHUNKS, 1.0) * 100.0, 0)

    @property
    def elapsed(self) -> str:
        s = int(time.monotonic() - self._t0)
        return f"{s // 60:02d}:{s % 60:02d}"

    @property
    def buffer_fill_pct(self) -> float:
        return 100.0 if self._filled else round(
            (self._write_pos / BUFFER_SIZE) * 100, 1
        )

    @property
    def has_minimum_audio(self) -> bool:
        return self._filled or self._write_pos >= SAMPLE_RATE * MIN_BUFFER_SEC

    def __del__(self):
        try:
            self._face_mesh.close()
        except Exception:
            pass


# ── WebSocket endpoint ────────────────────────────────────────────

@app.websocket("/ws/atc/{operator_id}")
async def atc_stream(websocket: WebSocket, operator_id: str) -> None:
    """
    WebSocket for one ATC operator's monitoring session.

    Message types sent to client:
      handshake        Session confirmed. baseline_ready flag included.
      buffering        Audio buffering in progress (first 2 sec).
      calibrating      Building personal baseline. Includes progress %.
      analysis         Normal fatigue score. Sent every chunk.
      stamp            5-minute epoch average. Sent to DB and client.
      eye_warning      Eyes closed too long. Logged to DB.
      baseline_ready   Calibration just completed this session.
      baseline_upgraded Adaptive baseline silently updated.
      timeout          No audio for STREAM_TIMEOUT seconds.
    """
    await websocket.accept()

    op   = db_get_operator(operator_id)
    if not op:
        await websocket.send_json({
            "type":    "error",
            "message": f"Operator '{operator_id}' not registered. Please register first.",
        })
        await websocket.close()
        return

    op_name    = op["name"]
    session_id = db_get_active_session(operator_id)
    session    = ATCSession(operator_id, op_name, session_id)
    loop       = asyncio.get_running_loop()

    session.load_baseline()

    log.info("WS START | op=%-10s | name=%-20s | session=%s | cal=%s",
             operator_id, op_name, session_id,
             "SKIP" if session.calibration_done else "NEEDED")

    await websocket.send_json({
        "type":           "handshake",
        "operator_id":    operator_id,
        "operator_name":  op_name,
        "session_id":     session_id,
        "baseline_ready": session.calibration_done,
        "epoch_sec":      EPOCH_DURATION_SEC,
        "message": (
            f"Session started for {op_name}. Baseline loaded — monitoring active."
            if session.calibration_done else
            f"Session started for {op_name}. "
            "Speak naturally for 30-40 seconds to build your voice profile."
        ),
        "timestamp":      time.strftime("%H:%M:%S"),
        "version":        "6.0.0",
    })

    try:
        while True:
            try:
                raw_msg = await asyncio.wait_for(
                    websocket.receive(), timeout=STREAM_TIMEOUT
                )
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type":    "timeout",
                    "message": f"No data for {STREAM_TIMEOUT:.0f}s. Session closing.",
                })
                break

            # ── Audio chunk ──────────────────────────────────────
            if raw_msg.get("bytes"):
                session.ingest_audio(raw_msg["bytes"])

                if not session.has_minimum_audio:
                    await websocket.send_json({
                        "type":     "buffering",
                        "message":  f"Buffering... ({session.chunk_count} chunks)",
                        "timestamp":time.strftime("%H:%M:%S"),
                    })
                    continue

                score, conf, cal_done, sub = await loop.run_in_executor(
                    _EXECUTOR, session.compute_voice
                )

                # Calibrating
                if not cal_done:
                    await websocket.send_json({
                        "type":       "calibrating",
                        "message":    "Building personal voice profile. Keep speaking naturally.",
                        "progress":   session.calibration_progress,
                        "timestamp":  time.strftime("%H:%M:%S"),
                    })
                    continue

                # Calibration just completed → notify
                if not getattr(session, '_cal_notified', False) and cal_done:
                    session._cal_notified = True
                    await websocket.send_json({
                        "type":    "baseline_ready",
                        "message": "Voice profile built. Fatigue monitoring now fully active.",
                        "timestamp": time.strftime("%H:%M:%S"),
                    })

                zone          = classify_zone(score)
                alert_message = generate_alert(zone["zone"], score)

                # Adaptive baseline
                baseline_upgraded = await loop.run_in_executor(
                    _EXECUTOR, session.try_adaptive_update
                )

                # Epoch tick (speech_detected via score > 0 and conf > 20)
                speech_in_chunk = score > 0 and conf > 20.0
                stamp = session.tick_epoch(score, zone["zone"], speech_in_chunk)

                if stamp:
                    stamp_zone    = classify_zone(stamp["avg_score"])
                    stamp_alert   = generate_alert(stamp_zone["zone"], stamp["avg_score"])
                    await loop.run_in_executor(
                        _EXECUTOR, db_log_stamp,
                        operator_id, op_name, session_id,
                        stamp["avg_score"], stamp["zone"],
                        stamp["green_count"], stamp["yellow_count"],
                        stamp["red_count"], stamp["speech_chunks"],
                        stamp["duration_sec"], stamp_alert,
                    )
                    await websocket.send_json({
                        "type":          "stamp",
                        "timestamp":     time.strftime("%H:%M:%S"),
                        "operator_id":   operator_id,
                        "avg_score":     stamp["avg_score"],
                        "zone":          stamp["zone"],
                        "label":         stamp["label"],
                        "green_count":   stamp["green_count"],
                        "yellow_count":  stamp["yellow_count"],
                        "red_count":     stamp["red_count"],
                        "speech_chunks": stamp["speech_chunks"],
                        "duration_sec":  stamp["duration_sec"],
                        "alert_message": stamp_alert,
                    })

                log.info(
                    "ANALYSIS | op=%-10s | score=%5.1f | zone=%-6s | conf=%4.1f%% | t=%s",
                    operator_id, score, zone["zone"], conf, session.elapsed,
                )

                await websocket.send_json({
                    "type":               "analysis",
                    "timestamp":          time.strftime("%H:%M:%S"),
                    "session_elapsed":    session.elapsed,
                    "operator_id":        operator_id,
                    "operator_name":      op_name,
                    "fatigue_score":      score,
                    "confidence":         conf,
                    "chunk_count":        session.chunk_count,
                    "buffer_fill_pct":    session.buffer_fill_pct,
                    "f0_baseline":        session.f0_baseline,
                    "hnr_baseline":       session.hnr_baseline,
                    "baseline_freshness": session.stored_freshness,
                    "baselines_ready":    session.calibration_done,
                    "baseline_upgraded":  baseline_upgraded,
                    "alert_message":      alert_message,
                    "sub_scores":         sub,
                    **zone,
                })

                if baseline_upgraded:
                    await websocket.send_json({
                        "type":      "baseline_upgraded",
                        "message":   "Voice profile updated — fresher voice recorded today.",
                        "freshness": session.stored_freshness,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })

            # ── Video frame ──────────────────────────────────────
            elif raw_msg.get("text"):
                try:
                    msg = json.loads(raw_msg["text"])
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "video_frame" and msg.get("frame"):
                    eye_warn = await loop.run_in_executor(
                        _EXECUTOR, session.ingest_video, msg["frame"]
                    )
                    if eye_warn:
                        # Log to DB
                        await loop.run_in_executor(
                            _EXECUTOR, db_log_eye_warning,
                            operator_id, op_name, session_id,
                            eye_warn["duration_sec"], session.elapsed,
                        )
                        # Send to frontend
                        await websocket.send_json({
                            **eye_warn,
                            "timestamp":      time.strftime("%H:%M:%S"),
                            "session_elapsed":session.elapsed,
                        })

    except WebSocketDisconnect as exc:
        log.info("WS DISCONNECT | op=%-10s | code=%s | chunks=%d | t=%s",
                 operator_id, exc.code, session.chunk_count, session.elapsed)
    except Exception:
        log.exception("WS ERROR | op=%-10s", operator_id)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        log.info("WS END | op=%-10s | chunks=%d | t=%s",
                 operator_id, session.chunk_count, session.elapsed)


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info("=" * 64)
    log.info("  AeroCalm AI  V6.0  ATC Fatigue Monitoring")
    log.info("  Voice-Only Scoring  |  Eye Closure Warnings")
    log.info("  5-Minute Epoch Stamps  |  Daily & Weekly Reports")
    log.info("  Dashboard  ->  http://127.0.0.1:8000")
    log.info("  Operators  ->  http://127.0.0.1:8000/api/operators")
    log.info("  Dashboard  ->  http://127.0.0.1:8000/api/dashboard/today")
    log.info("=" * 64)
    uvicorn.run("main:app", host="127.0.0.1", port=8000,
                reload=False, log_level="warning")
