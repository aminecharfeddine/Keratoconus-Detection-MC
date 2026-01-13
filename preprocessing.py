# =========================
# preprocessing.py
# =========================
import pandas as pd
import joblib

FEATURE_NAMES_PATH = "models/feature_names.pkl"
FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)

DROP_COLS = [
    "PatientID", "First Name", "Last Name", "DOB",
    "Age", "Gender", "Ethnicity",
    "Scan Date", "Scan Time", "Scan Type"
]

def load_and_preprocess(uploaded_file):
    # Lecture auto du séparateur (tab ou virgule)
    df = pd.read_csv(uploaded_file, sep=None, engine="python")

    if "Eye" not in df.columns:
        raise ValueError("Colonne 'Eye' absente du fichier")

    # Nettoyage colonnes non utilisées
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Séparation par œil
    eyes_data = {}

    for eye in df["Eye"].unique():
        df_eye = df[df["Eye"] == eye].copy()

        # Supprimer la colonne Eye après split
        df_eye = df_eye.drop(columns=["Eye"])

        # Alignement strict avec les features du modèle
        df_eye = df_eye.reindex(columns=FEATURE_NAMES)

        if df_eye.isnull().all(axis=1).any():
            raise ValueError(f"Données manquantes excessives pour l'œil {eye}")

        eyes_data[eye] = df_eye

    return eyes_data
