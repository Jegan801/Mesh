import joblib
import numpy as np
import os

MODEL_PATH = "models/severity_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"ML model not found at {MODEL_PATH}. "
        "Run: python ml/train_severity_model.py"
    )

model = joblib.load(MODEL_PATH)

def predict_severity(features):
    features = np.array(features).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    cls = int(np.argmax(proba))
    return cls, proba.tolist()
