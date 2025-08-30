import numpy as np
from backend.anomaly_model import IFDetector

def gen_normal_samples(n=10000, seed=7):
    rng = np.random.default_rng(seed)
    tide = 1.2 + 0.3 * rng.normal(size=n)
    wave = 1.5 + 0.5 * rng.normal(size=n)
    wind = 6.0 + 2.0 * rng.normal(size=n)
    rain = np.clip(rng.normal(0, 0.5, size=n), 0, None)
    turb = np.clip(4.0 + 1.0 * rng.normal(size=n), 0, None)
    chl = np.clip(1.5 + 0.5 * rng.normal(size=n), 0, None)
    oil = np.clip(0.02 + 0.02 * rng.normal(size=n), 0, 0.15)
    X = np.stack([tide, wave, wind, rain, turb, chl, oil], axis=1)
    return X

if __name__ == "__main__":
    X = gen_normal_samples()
    det = IFDetector()
    det.fit(X)
    det.save("models/iforest.joblib")
    print("Saved models/iforest.joblib")
