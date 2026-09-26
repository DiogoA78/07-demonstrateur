"""
Démonstrateur Data Science — Portfolio Diogo Almeida
====================================================
Application Streamlit multi-modules regroupant les démos interactives
des projets 3 à 6 du portfolio Data Science.
"""

import streamlit as st
from utils.style import apply_custom_css

# --- Configuration de la page ---
st.set_page_config(
    page_title="Démonstrateur Data Science",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS global ---
apply_custom_css()

# --- Définition des modules ---
MODULES = {
    "Accueil": {
        "icon": "🏠",
        "description": "Présentation du démonstrateur",
    },
    "Immobilier IDF": {
        "icon": "🏘️",
        "description": "Prédiction de prix immobilier en Île-de-France",
        "project": "Projet 3",
        "tags": ["Régression", "Géospatial", "Folium"],
    },
    "Anomalies capteurs": {
        "icon": "📡",
        "description": "Détection d'anomalies sur données de capteurs industriels",
        "project": "Projet 4",
        "tags": ["Anomaly Detection", "Time Series", "Unsupervised"],
    },
    "Scoring crédit": {
        "icon": "💳",
        "description": "Scoring de demandes de crédit avec explicabilité SHAP",
        "project": "Projet 5",
        "tags": ["Classification", "SHAP", "Fairlearn"],
    },
    "Sentiment NLP": {
        "icon": "💬",
        "description": "Analyse de sentiment sur critiques de films (FR)",
        "project": "Projet 6",
        "tags": ["NLP", "CamemBERT", "TF-IDF"],
    },
}


# --- Sidebar : navigation ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="font-size: 1.5rem; margin: 0;">🔬 Démonstrateur</h1>
        <p style="color: #B0B0B0; font-size: 0.85rem; margin: 5px 0 0 0;">
            Data Science Portfolio
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected = st.radio(
        "Navigation",
        list(MODULES.keys()),
        format_func=lambda x: f"{MODULES[x]['icon']}  {x}",
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="padding: 10px; font-size: 0.8rem; color: #888;">
        <p><strong>Diogo Almeida</strong></p>
        <p>Data Scientist</p>
        <p>
            <a href="https://github.com/DiogoA78" target="_blank" style="color: #4F8BF9;">GitHub</a> ·
            <a href="https://linkedin.com/in/diogo-almeida0" target="_blank" style="color: #4F8BF9;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- Page d'accueil ---
def render_home():
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 10px;">
            🔬 Démonstrateur Data Science
        </h1>
        <p style="color: #B0B0B0; font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
            Application interactive regroupant les algorithmes de mon portfolio.
            Explorez chaque module pour tester les modèles en temps réel.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    cols = st.columns(2)
    module_list = [k for k in MODULES if k != "Accueil"]

    for i, name in enumerate(module_list):
        mod = MODULES[name]
        with cols[i % 2]:
            tags_html = " ".join(
                f'<span class="badge badge-info">{t}</span>' for t in mod["tags"]
            )
            st.markdown(f"""
            <div class="metric-card" style="min-height: 160px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">{mod['icon']}</div>
                <h3 style="font-size: 1.1rem; color: #FAFAFA; font-weight: 600;">{name}</h3>
                <p style="color: #B0B0B0; font-size: 0.85rem; margin: 8px 0;">
                    {mod['description']}
                </p>
                <div style="margin-top: 10px;">
                    <span class="badge badge-success">{mod['project']}</span>
                    {tags_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

    st.markdown("")

    st.markdown("""
    <div style="text-align: center; padding: 30px 0; color: #666; font-size: 0.85rem;">
        <p>Sélectionnez un module dans la barre latérale pour commencer.</p>
        <p style="margin-top: 10px;">
            Diogo Almeida · Data Scientist ·
            <a href="https://github.com/DiogoA78" target="_blank" style="color: #4F8BF9;">GitHub</a> ·
            <a href="https://linkedin.com/in/diogo-almeida0" target="_blank" style="color: #4F8BF9;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- Routeur ---
if selected == "Accueil":
    render_home()
elif selected == "Immobilier IDF":
    from modules.immobilier import render
    render()
elif selected == "Anomalies capteurs":
    from modules.anomalies import render
    render()
elif selected == "Scoring crédit":
    from modules.credit import render
    render()
elif selected == "Sentiment NLP":
    from modules.sentiment import render
    render()
