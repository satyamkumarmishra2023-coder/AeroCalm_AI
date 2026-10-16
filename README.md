# AeroCalm AI

**Voice-based fatigue monitoring prototype for Air Traffic Controllers.**

AeroCalm AI listens to an operator's natural speech during duty, compares it against that operator's *own* stored voice baseline, and produces a fatigue score every few seconds. Scores are averaged into 5-minute epochs and logged so a supervisor can see trends by shift, day and week.

> **Status: research prototype (V6.0).** Not validated against labelled fatigue data. Not a medical device. Not a safety-critical system. Do not use it to make operational or clinical decisions.

---

## Table of contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Voice features](#voice-features)
4. [Privacy and data handling](#privacy-and-data-handling)
5. [Quick start](#quick-start)
6. [Using the dashboard](#using-the-dashboard)
7. [API reference](#api-reference)
8. [Project structure](#project-structure)
9. [Configuration](#configuration)
10. [Known limitations](#known-limitations)
11. [Roadmap](#roadmap)
12. [Disclaimer](#disclaimer)

---

## What it does

- **ATC-only.** One role (ATC operator). No pilot, crew or manager roles.
- **Voice-only scoring.** The fatigue score comes 100% from voice features.
- **Personal baseline.** Each operator is calibrated once from natural speech. Later sessions load the stored baseline immediately.
- **Adaptive baseline.** If an operator sounds clearly fresher than their stored baseline, the baseline is meant to update (see [Known limitations](#known-limitations)).
- **5-minute epoch stamps.** Scores are averaged over 5 minutes, using only chunks where speech was detected, so a one-second spike cannot create a stamp on its own.
- **Eye-closure warning (optional).** If a webcam is used, sustained eye closure triggers a logged warning. Eye state does **not** affect the fatigue score.
- **Duty sessions.** Operators press *Start Duty* and *End Duty*. Each duty produces a session summary.
- **Command Centre.** Supervisor view with all operators, an alerts feed, registration, and daily and weekly reports with a simple trend and suggestion.

## How it works

```
Browser (mic, optional camera)
   │  raw PCM audio (22,050 Hz, float32)  +  JPEG frames (optional)
   ▼
FastAPI WebSocket  /ws/atc/{operator_id}
   │
   ├─ 5-second ring buffer (in RAM)
   ├─ Spectral subtraction (noise reduction)
   ├─ Dynamic voice-activity detection
   ├─ 9 voice features  ─►  weighted fatigue score (0–100)
   ├─ Smoothing (last 6 scores)
   ├─ 5-minute epoch average  ─►  stamp  ─►  SQLite
   └─ Eye-closure counter (optional)  ─►  warning  ─►  SQLite
```

**Calibration.** On an operator's first session the system collects features from natural speech until it has enough chunks (`BASELINE_CHUNKS = 8`, roughly 30–40 seconds of speech). No score is shown during this time. The baseline (mean F0, HNR and a 13-dimensional MFCC vector) is then saved to the database.

**Zones.**

| Score | Zone | Meaning |
|---|---|---|
| < 40 | GREEN | No fatigue indicators |
| 40 – 64 | YELLOW | Elevated; rest break suggested |
| ≥ 65 | RED | High; rest and medical consultation suggested |

The 40 and 65 cut-offs are hand-set and have not been validated against real data.

## Voice features

Names below describe what the code actually computes. Several are **proxies** inspired by voice-fatigue literature, not exact clinical measurements.

| Feature | Weight | What the code computes |
|---|---|---|
| RMS energy | 10% | Loudness of voiced samples vs. a fixed reference |
| Spectral flatness | 5% | Mean spectral flatness (breathiness proxy) |
| Jitter | 18% | Period-to-period F0 variation from frame-level YIN pitch (a proxy for cycle-level jitter) |
| Shimmer | 18% | Frame-to-frame RMS ratio in dB (a proxy for cycle-level shimmer) |
| F0 drop | 10% | Decline of mean F0 below the personal baseline |
| Formant sluggishness | 5% | Variability of spectral centroid (a proxy; not LPC formants) |
| HNR | 14% | Autocorrelation at the mean-F0 lag over the whole buffer, compared to baseline |
| Speech rate | 12% | Voiced-frame ratio and energy-peak syllable rate |
| MFCC deviation | 8% | Cosine distance of mean MFCC vector from the personal baseline |

Weights sum to 1.0 and are hand-tuned, not learned.

## Privacy and data handling

- Audio is streamed from the browser to the backend over a WebSocket and processed **in RAM**. The code does not write audio to disk.
- The database stores **numbers and text only**: operator ID and name, baseline features (F0, HNR, MFCC vector), scores, zones, timestamps, and alert messages.
- Video frames, if the camera is enabled, are decoded in memory and are not stored.
- "Local" holds only when the browser and the backend run on the **same machine** (the default `127.0.0.1` setup). If you deploy the backend elsewhere, audio crosses the network.
- There is currently **no authentication or encryption**. See [Known limitations](#known-limitations).

## Quick start

**Requirements:** Python 3.10 or newer, a microphone, a modern browser (Chrome or Edge recommended).

```bash
git clone <your-repo-url>
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install fastapi uvicorn librosa numpy scipy opencv-python mediapipe pydantic

uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser. `index.html` must sit next to `main.py`.

The SQLite database `aerocalm_atc.db` is created automatically on first run.

## Using the dashboard

**Supervisor (Command Centre tab)**
1. Open **Command Centre → Register ATC Operator** and add an ID and name.
2. Use the same tab to see today's status, recent alerts, and daily or weekly reports.

**Operator (Operator Panel tab)**
1. Log in with the Operator ID.
2. Press **Start Duty** and speak naturally. On the first session, wait for the voice-profile progress to reach 100%.
3. Watch the score and the 5-minute stamps.
4. Press **End Duty** to close the session and get a summary.

## API reference

Base URL: `http://127.0.0.1:8000`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status and key settings |
| POST | `/api/operators` | Register an operator `{operator_id, name}` |
| GET | `/api/operators` | List operators |
| DELETE | `/api/operators/{id}` | Remove an operator |
| POST | `/api/session/start` | Start a duty session `{operator_id}` |
| POST | `/api/session/end` | End the session and return a summary |
| GET | `/api/session/active/{id}` | Is this operator on duty? |
| GET | `/api/dashboard/today` | Status of all operators today |
| GET | `/api/dashboard/alerts?limit=30` | Recent fatigue alerts and eye warnings |
| GET | `/api/stamps/{id}?limit=20` | Recent epoch stamps |
| GET | `/api/reports/daily/{id}?report_date=YYYY-MM-DD` | Daily report |
| GET | `/api/reports/weekly/{id}` | 7-day report with trend |
| WS | `/ws/atc/{operator_id}` | Live monitoring stream |

**WebSocket messages from server:** `handshake`, `buffering`, `calibrating`, `baseline_ready`, `analysis`, `stamp`, `eye_warning`, `baseline_upgraded`, `timeout`, `error`.

Interactive docs are available at `/docs` (FastAPI Swagger UI).

## Project structure

```
.
├── main.py          # FastAPI backend: DSP pipeline, WebSocket, SQLite, REST API
├── index.html       # Single-file dashboard (Operator Panel + Command Centre)
├── aerocalm_atc.db  # Created at runtime (not committed)
└── README.md
```

Add `aerocalm_atc.db*` to `.gitignore`.

## Configuration

Key constants at the top of `main.py`:

| Constant | Default | Meaning |
|---|---|---|
| `SAMPLE_RATE` | 22050 | Audio sample rate (browser sends the same) |
| `BUFFER_SECONDS` | 5 | Analysis window |
| `BASELINE_CHUNKS` | 8 | Chunks needed to build a baseline |
| `ZONE_YELLOW` / `ZONE_RED` | 40 / 65 | Zone thresholds |
| `EPOCH_DURATION_SEC` | 300 | Epoch (stamp) length |
| `MIN_SPEECH_FOR_STAMP` | 3 | Minimum speech chunks for a stamp |
| `EYE_CLOSE_FRAMES_THRESH` | 6 | Closed-eye frames before a warning (about 3 s at 2 fps) |
| `EAR_BLINK_THRESH` | 0.20 | Eye-aspect-ratio threshold for "closed" |
| `BASELINE_UPDATE_MARGIN` | 5.0 | Freshness gain needed to update the baseline |

## Known limitations

These are known issues in V6.0. Read them before demoing or evaluating the project.

**Correctness and performance**
- **Processing can fall behind real time.** HNR uses a full-length `np.correlate` on a 5-second buffer, which is slow on typical laptops, and analysis runs on every incoming chunk. Expect growing latency. Planned fix: FFT-based autocorrelation and analysis about once per second.
- **Reconnect can duplicate streams.** After a WebSocket drop, the browser may start a second microphone and camera pipeline. Planned fix: start capture once per duty, not once per connection.
- **Adaptive baseline does not truly adapt yet.** The update path recomputes a freshness score but writes back the old baseline values. Treat adaptive update as unfinished.
- **Time zones in reports.** Stamps use local dates; sessions and eye warnings use SQLite UTC dates. Reports near midnight, and overnight shifts, can be split or misattributed. Planned fix: report per duty session.
- **Browser audio processing.** The page requests echo cancellation and noise suppression, which can distort jitter, shimmer and HNR. Planned fix: disable both.
- **Camera starts automatically** when a session connects and `cv2` and `mediapipe` are required imports. Planned fix: camera as an opt-in toggle.

**Security**
- No login or roles: anyone who can reach the server can use Command Centre, including deleting operators.
- Operator names are inserted into the page with `innerHTML` (script-injection risk).
- CORS is open (`*`) and traffic is unencrypted.
- Alert text says "Supervisor notified", but no notification is sent. It only logs.

**Science**
- Not validated against labelled fatigue data or against a reference tool such as Praat.
- Thresholds and weights are hand-set. Speech-rate references assume normal conversation; ATC phraseology is fast and radio audio is band-limited.
- If an operator is already tired at first calibration, later scores are relative to a tired baseline.
- Single microphone type tested; results will differ across headsets and rooms.

## Roadmap

- [ ] Fix the known issues above (V6.1)
- [ ] CSV export of numeric features and scores for validation studies
- [ ] Compare features against Praat on the same recordings
- [ ] Authentication and supervisor/operator roles
- [ ] Optional camera toggle, off by default
- [ ] Consent screen and data-retention controls
- [ ] Pilot study with a real ATC facility, with ethics approval and supervisor sign-off
- [ ] Threshold calibration from collected data

## Disclaimer

AeroCalm AI is an experimental student project. It does not diagnose fatigue, illness or fitness for duty, and it does not replace medical professionals, rostering rules or regulatory fatigue-risk-management processes. Any real-world trial needs explicit consent from participants and approval from the relevant organisation.

## License
Copyright (c) 2026 Satyam Kumar Mishra.
All Rights Reserved.
This project is closed-source. You may not copy, distribute, modify, or use this code without explicit written permission from the author.
