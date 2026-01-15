# =========================
# predict.py
# =========================
import joblib
from preprocessing import load_and_preprocess

MODEL_PATH = "models/keratoconus_multiclass_pipeline.pkl"

model = joblib.load(MODEL_PATH)

LABEL_MAP = {
    0: "Normal",
    1: "Fruste",
    2: "Kératocône avéré"
}

def predict_from_file(uploaded_file):
    eyes_data = load_and_preprocess(uploaded_file)

    results = {}

    for eye, X in eyes_data.items():
        proba = model.predict_proba(X)[0]
        pred_class = int(proba.argmax())

        results[eye] = {
            "class": pred_class,
            "label": LABEL_MAP[pred_class],
            "probabilities": {
                "Normal": proba[0],
                "Fruste": proba[1],
                "Kératocône": proba[2]
            }
        }

    return results
