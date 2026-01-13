# =========================
# preprocessing.py
# =========================
import pandas as pd
import joblib

FEATURE_NAMES_PATH = "models/feature_names.pkl"

FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)

def load_and_preprocess(uploaded_file):
    # Détection du séparateur (tabulation ou virgule)
    df = pd.read_csv(uploaded_file, sep=None, engine="python")

    # Supprimer colonnes non numériques / administratives si présentes
    drop_cols = [
        "PatientID", "First Name", "Last Name", "DOB",
        "Age", "Gender", "Ethnicity", "Eye",
        "Scan Date", "Scan Time", "Scan Type"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Garder uniquement les colonnes vues à l'entraînement
    df = df.reindex(columns=FEATURE_NAMES)

    # Une ligne = un œil
    if len(df) != 1:
        raise ValueError("Le fichier doit contenir exactement un œil")

    return df
