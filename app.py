import streamlit as st
import pandas as pd
import plotly.express as px
from predict import predict_from_file

st.set_page_config(
    page_title="Keratoconus AI",
    page_icon="👁️",
    layout="wide"
)

# --- CONFIGURATION DES COULEURS ---
COLOR_MAP = {
    "Normal": "#2ecc71",       # Vert
    "Fruste": "#f1c40f",       # Jaune/Or
    "Kératocône": "#e74c3c"    # Rouge
}

st.title("👁️ Détection du kératocône par IA")

uploaded_file = st.file_uploader("📂 Importer le fichier patient (.txt ou .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        with st.spinner("Analyse des données cornéennes..."):
            results = predict_from_file(uploaded_file)

        if not results:
            st.warning("Aucune donnée détectée.")
        else:
            st.success(f"Analyse terminée ({len(results)} œil/yeux)")

            # Création des onglets
            tabs = st.tabs([f"👁️ Œil {eye}" for eye in results.keys()])

            for i, (eye, res) in enumerate(results.items()):
                with tabs[i]:
                    # Récupération de la couleur selon le label
                    # On simplifie le label pour matcher COLOR_MAP (ex: "Kératocône avéré" -> "Kératocône")
                    main_label = res['label']
                    bg_color = "#f0f2f6"
                    if "Normal" in main_label: bg_color = COLOR_MAP["Normal"]
                    elif "Fruste" in main_label: bg_color = COLOR_MAP["Fruste"]
                    else: bg_color = COLOR_MAP["Kératocône"]

                    # --- EN-TÊTE COLORÉ ---
                    st.markdown(
                        f"""
                        <div style="background-color:{bg_color}; padding:20px; border-radius:10px; text-align:center;">
                            <h2 style="color:white; margin:0;">{main_label.upper()}</h2>
                        </div>
                        """, 
                        unsafe_style_context=True, # Note: Utilisez unsafe_allow_html=True
                        unsafe_allow_html=True
                    )

                    st.write("##") # Espacement

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("📊 Probabilités")
                        df_proba = pd.DataFrame({
                            'Classe': list(res["probabilities"].keys()),
                            'Confiance': list(res["probabilities"].values())
                        })
                        
                        # Graphique Plotly personnalisé
                        fig = px.bar(
                            df_proba, 
                            x='Confiance', 
                            y='Classe', 
                            orientation='h',
                            color='Classe',
                            color_discrete_map={
                                "Normal": COLOR_MAP["Normal"],
                                "Fruste": COLOR_MAP["Fruste"],
                                "Kératocône": COLOR_MAP["Kératocône"]
                            },
                            range_x=[0, 1]
                        )
                        fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("📝 Note clinique")
                        if "Normal" in main_label:
                            st.write("L'analyse topographique ne présente pas de signe suspect de kératocône.")
                        elif "Fruste" in main_label:
                            st.warning("Prudence : Des irrégularités subtiles ont été détectées. Un suivi est recommandé avant toute chirurgie réfractive.")
                        else:
                            st.error("Signes clairs de kératocône détectés. Une prise en charge spécialisée est suggérée.")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

st.markdown("---")
st.caption("Aide au diagnostic basée sur LightGBM | © 2026 Clinique Ophtalmologique de Tunis")
