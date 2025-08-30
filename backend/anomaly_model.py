import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
from joblib import dump, load
from sklearn.ensemble import IsolationForest
import os

METRIC_ORDER = [
    "tide_level_m",
    "wave_height_m",
    "wind_speed_ms",
    "rain_mm",
    "turbidity_NTU",
    "chlorophyll_mg_m3",
    "oil_slick_score",
]

@dataclass
class RollingStats:
    window: int = 60

class RollingZScoreDetector:
    def __init__(self, window: int = 60, z_thresh: float = 3.0):
        self.window = window
        self.z_thresh = z_thresh
        self.buffers: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.window)))

    def update(self, station_id: str, metrics: Dict[str, float]) -> Dict[str, float]:
        zscores = {}
        for m in METRIC_ORDER:
            x = metrics.get(m, 0.0)
            buf = self.buffers[station_id][m]
            buf.append(x)
            if len(buf) < 10:  # warmup
                zscores[m] = 0.0
                continue
            arr = np.array(buf, dtype=float)
            mu = arr.mean()
            sd = arr.std(ddof=1) or 1e-6
            zscores[m] = float((x - mu) / sd)
        return zscores

    def is_anomaly(self, zscores: Dict[str, float]) -> bool:
        return any(abs(z) > self.z_thresh for z in zscores.values())

class IFDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.clf: Optional[IsolationForest] = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def fit(self, X: np.ndarray, **kwargs):
        self.clf = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
            **kwargs,
        ).fit(X)

    def score(self, metrics: Dict[str, float]) -> float:
        if self.clf is None:
            return 0.0
        x = np.array([[metrics.get(m, 0.0) for m in METRIC_ORDER]], dtype=float)
        s = self.clf.score_samples(x)[0]
        norm = (0.5 - s) / 2.0
        return float(np.clip(norm, 0.0, 1.0))

    def save(self, path: str):
        if self.clf is not None:
            dump(self.clf, path)

    def load(self, path: str):
        self.clf = load(path)

def label_threat(metrics: Dict[str, float], z: Dict[str, float], if_score: float) -> List[str]:
    labels = []
    if abs(z.get("tide_level_m", 0)) > 2.5 and abs(z.get("wave_height_m", 0)) > 2.5 and abs(z.get("wind_speed_ms", 0)) > 2.0:
        labels.append("possible_flooding")
    if abs(z.get("chlorophyll_mg_m3", 0)) > 3.0 and abs(z.get("turbidity_NTU", 0)) > 2.0:
        labels.append("possible_algal_bloom")
    if metrics.get("oil_slick_score", 0) > 0.4 or abs(z.get("oil_slick_score", 0)) > 3.5:
        labels.append("possible_oil_spill")
    if if_score > 0.75 and not labels:
        labels.append("generic_anomaly")
    return labels
