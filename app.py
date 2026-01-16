import streamlit as st
import pandas as pd
from predict import predict_from_file
import altair as alt


st.set_page_config(
    page_title="Keratoconus AI",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Détection du kératocône par IA - Bêta Test")

uploaded_file = st.file_uploader("📂 Importer le fichier patient (.txt ou .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        with st.spinner("Analyse des données cornéennes..."):
            # Appel de la fonction de votre script predict.py
            # Elle doit retourner un dictionnaire : { "OD": {...}, "OS": {...} }
            results = predict_from_file(uploaded_file)

        if not results:
            st.error("Aucune donnée n'a pu être extraite. Vérifiez le format du fichier.")
        else:
            st.success(f"Analyse terminée ({len(results)} œil/yeux détectés)")

            # Utilisation d'onglets pour un affichage propre par œil
            tab_list = st.tabs([f"👁️ Œil {eye}" for eye in results.keys()])

            for i, (eye, res) in enumerate(results.items()):
                with tab_list[i]:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric(label="Diagnostic", value=res['label'])
                        st.write("**Détails des probabilités :**")
                        # Transformation pour l'affichage
                        ORDER = ["Normal", "Fruste", "Kératocône"]
                        proba_df = pd.DataFrame.from_dict(
                            res["probabilities"], 
                            orient="index", 
                            columns=["Score"]
                        )
                        st.dataframe(proba_df.style.highlight_max(axis=0, color='lightgreen'))

                    with col2:
                        ORDER = ["Normal", "Fruste", "Kératocône"]
                        
                        chart_df = (
                            proba_df
                            .reset_index()
                            .rename(columns={"index": "Classe"})
                        )
                        
                        chart = (
                                alt.Chart(chart_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Classe:N", sort=ORDER, title="Classe diagnostique"),
                                    y=alt.Y("Score:Q", title="Probabilité"),
                                    color=alt.Color(
                                        "Classe:N",
                                        scale=alt.Scale(
                                            domain=["Normal", "Fruste", "Kératocône"],
                                            range=["#2ecc71", "#f39c12", "#e74c3c"]
                                        ),
                                        legend=None
                                    ),
                                    tooltip=[
                                        alt.Tooltip("Classe", title="Classe"),
                                        alt.Tooltip("Score", title="Probabilité", format=".2%")
                                    ]
                                )
                            )
                        
                        st.altair_chart(chart, use_container_width=True)

                    st.info("💡 Interprétation : Ce résultat doit être corrélé à l'examen clinique.")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

st.markdown("---")
st.caption("Modèle LightGBM – Aide au diagnostic du kératocône | © 2026 - Clinique Ophtalmologique de Tunis")
