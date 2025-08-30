from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import sqlite3
import json
import os

from anomaly_model import RollingZScoreDetector, IFDetector, label_threat, METRIC_ORDER

DB_PATH = os.getenv("DB_PATH", "backend.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id TEXT PRIMARY KEY,
        station_id TEXT,
        timestamp TEXT,
        payload TEXT
    )")
    cur.execute("
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        station_id TEXT,
        timestamp TEXT,
        threat_type TEXT,
        severity TEXT,
        confidence REAL,
        payload TEXT,
        acknowledged INTEGER DEFAULT 0
    )")
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="Coastal Threat Alert Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

det_z = RollingZScoreDetector(window=60, z_thresh=3.0)
IFOREST_PATH = os.getenv("IFOREST_PATH", "models/iforest.joblib")
det_if = IFDetector(model_path=IFOREST_PATH if os.path.exists(IFOREST_PATH) else None)

class IngestPayload(BaseModel):
    source: Optional[str]
    timestamp: str
    station_id: str
    lat: Optional[float]
    lon: Optional[float]
    metrics: Dict[str, float]
    anomaly: Optional[Dict[str, Any]] = None

@app.post("/ingest")
async def ingest(payload: IngestPayload):
    # store reading
    rid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO sensor_readings (id, station_id, timestamp, payload) VALUES (?, ?, ?, ?)",
                (rid, payload.station_id, payload.timestamp, json.dumps(payload.dict())))
    conn.commit()
    conn.close()

    # If payload already flagged anomaly, create alert; else run detection heuristics
    labels = []
    if payload.anomaly and payload.anomaly.get("is_anomaly"):
        labels = payload.anomaly.get("labels", [])
    else:
        z = det_z.update(payload.station_id, payload.metrics)
        if_score = det_if.score(payload.metrics)
        labels = label_threat(payload.metrics, z, if_score)

    if labels:
        aid = str(uuid.uuid4())
        threat = labels[0]
        severity = "low"
        if "possible_oil_spill" in labels or "possible_flooding" in labels:
            severity = "high"
        elif "possible_algal_bloom" in labels:
            severity = "medium"
        confidence = 0.5
        # try read iforest
        try:
            confidence = float(det_if.score(payload.metrics))
        except Exception:
            pass

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO alerts (id, station_id, timestamp, threat_type, severity, confidence, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (aid, payload.station_id, payload.timestamp, threat, severity, confidence, json.dumps(payload.dict())))
        conn.commit()
        conn.close()
        return {"status": "alert_created", "alert_id": aid, "labels": labels}

    return {"status": "ok", "labels": labels}

@app.get("/alerts")
async def get_alerts():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, station_id, timestamp, threat_type, severity, confidence, acknowledged FROM alerts ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    alerts = []
    for r in rows:
        alerts.append({
            "id": r[0],
            "station_id": r[1],
            "timestamp": r[2],
            "threat_type": r[3],
            "severity": r[4],
            "confidence": r[5],
            "acknowledged": bool(r[6])
        })
    return {"alerts": alerts}

@app.post("/alerts/{alert_id}/ack")
async def ack_alert(alert_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if not changed:
        raise HTTPException(status_code=404, detail="alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}
