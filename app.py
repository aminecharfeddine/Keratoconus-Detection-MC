# =========================
# app.py
# =========================
import streamlit as st
import pandas as pd
from predict import predict_from_file

st.set_page_config(
    page_title="Keratoconus AI – Aide à la décision",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Détection du kératocône par IA")
st.markdown(
    """
    **Outil d’aide à la décision médicale**  
    Importez un fichier issu de la machine d’imagerie cornéenne pour obtenir
    une estimation du stade du kératocône.

    ⚠️ *Ce système ne remplace pas le diagnostic médical.*
    """
)

uploaded_file = st.file_uploader(
    "📂 Importer le fichier de l'œil (.txt ou .csv)",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    with st.spinner("Analyse en cours..."):
        result = predict_from_file(uploaded_file)

    st.success("Analyse terminée")

    st.subheader("🧪 Résultat du modèle")
    st.write(f"**Classe prédite :** {result['label']}")

    st.subheader("📊 Probabilités par classe")
    proba_df = pd.DataFrame(
        result["probabilities"],
        index=["Normal", "Fruste", "Kératocône"],
        columns=["Probabilité"]
    )
    st.bar_chart(proba_df)

    st.markdown("---")
    st.caption(
        "Modèle IA entraîné sur données tomographiques cornéennes – "
        "priorité à la détection précoce du kératocône fruste."
    )
