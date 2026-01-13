import streamlit as st
import pandas as pd
from predict import predict_from_file

# Configuration de la page
st.set_page_config(
    page_title="KeraCheck AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour le style
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822102.png", width=80)
    st.title("Navigation")
    st.info("💡 **Aide** : Importez le fichier d'export brut (.csv/.txt) de la topographie cornéenne.")
    
    st.divider()
    st.subheader("Paramètres du modèle")
    st.write("Modèle : `LightGBM v2.1`")
    st.write("Seuil de confiance : `85%` :white_check_mark:")

# --- HEADER ---
st.title("👁️ Keratoconus AI Decision Support")
st.caption("Analyse automatisée de la morphologie cornéenne par Intelligence Artificielle.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("", type=["txt", "csv"])

if uploaded_file:
    try:
        with st.spinner("Analyse biométrique en cours..."):
            results = predict_from_file(uploaded_file)
        
        st.success("Analyse terminée avec succès.")
        
        # --- RESULTS DISPLAY ---
        # On crée deux colonnes pour OD et OS si disponibles
        tabs = st.tabs([f"👁️ Œil {eye}" for eye in results.keys()])

        for i, (eye, res) in enumerate(results.items()):
            with tabs[i]:
                # Détermination de la couleur selon le label
                label = res['label']
                color = "#28a745" if "Normal" in label else "#fd7e14" if "Fruste" in label else "#dc3545"
                bg_light = "#e8f5e9" if "Normal" in label else "#fff3e0" if "Fruste" in label else "#ffebee"

                # Header de résultat
                st.markdown(f"""
                    <div style="background-color:{bg_light}; padding:20px; border-radius:10px; border-left: 8px solid {color};">
                        <h2 style="color:{color}; margin:0;">{label}</h2>
                        <p style="color:#444; margin:0;">Confiance du modèle : <b>{max(res['probabilities'].values()):.1%}</b></p>
                    </div>
                """, unsafe_allow_html=True)

                st.write(" ") # Spacer

                col_data, col_chart = st.columns([1, 1.5])

                with col_data:
                    st.subheader("Probabilités par classe")
                    # Création d'un DF propre pour l'affichage
                    df_res = pd.DataFrame({
                        "Diagnostic": list(res["probabilities"].keys()),
                        "Score (%)": [v * 100 for v in res["probabilities"].values()]
                    })
                    st.table(df_res.style.format({"Score (%)": "{:.1f}%"}))
                    
                    if "Kératocône" in label:
                        st.warning("⚠️ **Avis médical :** Signes de forte suspicion. Une tomographie d'élévation est recommandée.")

                with col_chart:
                    # Graphique à barres horizontal
                    st.subheader("Distribution du risque")
                    st.bar_chart(data=df_res.set_index("Diagnostic"), horizontal=True, height=200)

                st.divider()
                with st.expander("Voir les données brutes extraites"):
                    st.write("Ces données sont celles utilisées par le modèle après prétraitement.")
                    # Si vous voulez afficher le DataFrame d'origine
                    st.json(res["probabilities"])

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")
        st.info("Vérifiez que le format du fichier correspond bien à l'export standard de la machine.")

else:
    # État vide (Landing page)
    st.info("Veuillez importer un fichier pour lancer l'analyse.")


# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Keratoconus Dectection AI Project - Clinique Ophtalmologique de Tunis.")
