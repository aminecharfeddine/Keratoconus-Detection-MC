# =========================
# predict.py
# =========================
import joblib
import pandas as pd
from preprocessing import load_and_preprocess

MODEL_PATH = "models/keratoconus_multiclass_pipeline.pkl"

model = joblib.load(MODEL_PATH)

LABEL_MAP = {
    0: "Normal",
    1: "Fruste (suspect kératocône)",
    2: "Kératocône avéré"
}

def predict_from_file(uploaded_file):
    X = load_and_preprocess(uploaded_file)

    proba = model.predict_proba(X)[0]
    pred_class = int(proba.argmax())

    return {
        "class": pred_class,
        "label": LABEL_MAP[pred_class],
        "probabilities": proba.tolist()
    }
