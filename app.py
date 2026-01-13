# =========================
# app.py
# =========================
import streamlit as st
import pandas as pd
from predict import predict_from_file

st.set_page_config(
    page_title="Keratoconus AI – Aide à la décision",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Détection du kératocône par IA")
st.markdown(
    """
    **Outil d’aide à la décision médicale**  
    Importez un fichier issu de la machine d’imagerie cornéenne (2 yeux).
    
    ⚠️ *Ce système ne constitue pas un dispositif médical autonome.*
    """
)

uploaded_file = st.file_uploader(
    "📂 Importer le fichier patient (.txt ou .csv)",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    with st.spinner("Analyse en cours..."):
        results = predict_from_file(uploaded_file)

    st.success("Analyse terminée")

    cols = st.columns(len(results))

    for col, (eye, res) in zip(cols, results.items()):
        with col:
            st.subheader(f"👁️ Œil {eye}")

            st.markdown(f"### 🧪 **{res['label']}**")

            proba_df = pd.DataFrame.from_dict(
                res["probabilities"],
                orient="index",
                columns=["Probabilité"]
            )

            st.bar_chart(proba_df)

            st.markdown("---")
            st.caption(
                "Interprétation clinique recommandée en contexte "
                "d’examen ophtalmologique complet."
            )

st.markdown("---")
st.caption(
    "Modèle LightGBM multiclasses – priorité à la détection du kératocône fruste."
)
