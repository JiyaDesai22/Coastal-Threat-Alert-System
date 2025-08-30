import os, time, json, random, math
import requests
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from backend.anomaly_model import RollingZScoreDetector, IFDetector, label_threat, METRIC_ORDER

STATIONS = [
    {"station_id": "IN-GJ-001", "lat": 21.64, "lon": 69.61},
    {"station_id": "IN-MH-002", "lat": 18.96, "lon": 72.82},
    {"station_id": "IN-GA-003", "lat": 15.49, "lon": 73.82},
    {"station_id": "IN-TN-004", "lat": 13.08, "lon": 80.27},
    {"station_id": "IN-WB-005", "lat": 21.94, "lon": 88.27},
]

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POST_MODE = os.getenv("POST_MODE", "ingest")
CADENCE_SEC = float(os.getenv("CADENCE_SEC", "3"))
IFOREST_PATH = os.getenv("IFOREST_PATH", "models/iforest.joblib")

Path("out").mkdir(exist_ok=True, parents=True)
det_z = RollingZScoreDetector(window=60, z_thresh=3.0)
det_if = IFDetector(model_path=IFOREST_PATH if os.path.exists(IFOREST_PATH) else None)

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def seasonal_tide(t, base=1.2):
    return base + 0.4 * math.sin(2*math.pi * (t % 3600) / 3600.0)

def gen_metrics(t, station, event=None, rng=None):
    rng = rng or np.random.default_rng()
    tide = seasonal_tide(t) + 0.1 * rng.normal()
    wave = 1.5 + 0.3 * rng.normal()
    wind = 6.0 + 1.5 * rng.normal()
    rain = max(0.0, 0.1 * rng.normal())
    turb = max(0.0, 4.0 + 0.7 * rng.normal())
    chl = max(0.0, 1.5 + 0.4 * rng.normal())
    oil = max(0.0, 0.03 + 0.02 * rng.normal())

    if event == "storm":
        tide += rng.uniform(0.6, 1.2)
        wave += rng.uniform(0.7, 1.5)
        wind += rng.uniform(4, 9)
        rain += rng.uniform(5, 20)
    elif event == "bloom":
        chl += rng.uniform(2.0, 5.0)
        turb += rng.uniform(2.0, 5.0)
    elif event == "oil":
        oil += rng.uniform(0.4, 0.9)
        turb += rng.uniform(0.0, 1.5)

    return {
        "tide_level_m": round(tide, 3),
        "wave_height_m": round(wave, 3),
        "wind_speed_ms": round(wind, 3),
        "rain_mm": round(rain, 3),
        "turbidity_NTU": round(turb, 3),
        "chlorophyll_mg_m3": round(chl, 3),
        "oil_slick_score": round(min(oil, 1.0), 3),
    }

def maybe_inject_event(t):
    r = random.random()
    if r < 0.02:
        return "storm"
    if r < 0.03:
        return "bloom"
    if r < 0.035:
        return "oil"
    return None

def post_json(path, payload):
    if not BACKEND_URL or POST_MODE == "none":
        return None, 0
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.text, resp.status_code
    except Exception as e:
        return str(e), -1

def build_ingest_payload(station, metrics, zscores, if_score, labels):
    return {
        "source": "simulator",
        "timestamp": now_iso(),
        "station_id": station["station_id"],
        "lat": station["lat"],
        "lon": station["lon"],
        "metrics": metrics,
        "anomaly": {
            "is_anomaly": bool(labels),
            "detectors": {
                "rolling_z": {k: round(v, 3) for k, v in zscores.items()},
                "iforest": round(if_score, 3),
            },
            "labels": labels
        }
    }

def build_alert(station, ingest_payload, labels):
    sev = "low"
    if "possible_oil_spill" in labels or "possible_flooding" in labels:
        sev = "high"
    elif "possible_algal_bloom" in labels:
        sev = "medium"
    return {
        "timestamp": ingest_payload["timestamp"],
        "station_id": station["station_id"],
        "threat_type": labels[0] if labels else "generic_anomaly",
        "severity": sev,
        "confidence": max(0.5, ingest_payload["anomaly"]["detectors"]["iforest"]),
        "payload": {"latest_reading": ingest_payload}
    }

def main():
    print(f"Simulator starting. BACKEND_URL='{BACKEND_URL or 'OFFLINE'}', POST_MODE={POST_MODE}, cadence={CADENCE_SEC}s")
    t = 0
    rng = np.random.default_rng(123)

    with open("out/stream.jsonl", "a", encoding="utf-8") as fstream,              open("out/alerts.jsonl", "a", encoding="utf-8") as falerts:

        while True:
            for st in STATIONS:
                event = maybe_inject_event(t)
                metrics = gen_metrics(t, st, event=event, rng=rng)

                z = det_z.update(st["station_id"], metrics)
                if_score = det_if.score(metrics)
                labels = label_threat(metrics, z, if_score)

                ingest_payload = build_ingest_payload(st, metrics, z, if_score, labels)

                fstream.write(json.dumps(ingest_payload) + "\n")
                fstream.flush()

                if POST_MODE in ("ingest", "both"):
                    _, code = post_json("/ingest", ingest_payload)
                    if code <= 0:
                        print(f"[WARN] /ingest not reachable (code={code})")

                if labels:
                    alert = build_alert(st, ingest_payload, labels)
                    falerts.write(json.dumps(alert) + "\n")
                    falerts.flush()
                    if POST_MODE in ("alerts", "both"):
                        _, code = post_json("/alerts", alert)
                        if code <= 0:
                            print(f"[WARN] /alerts not reachable (code={code})")

            t += 1
            time.sleep(CADENCE_SEC)

if __name__ == "__main__":
    main()
