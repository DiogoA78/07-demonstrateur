"""
Démonstrateur Data Science — Portfolio Diogo Almeida
====================================================
Application Streamlit multi-modules regroupant les démos interactives
des projets 3 à 6 du portfolio Data Science.

Bilingual: Français / English
"""

import streamlit as st
from utils.style import apply_custom_css
from utils.translations import get_text

# --- Configuration de la page ---
st.set_page_config(
    page_title="Data Science Demonstrator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS global ---
apply_custom_css()

# --- Module definitions (language-agnostic keys) ---
MODULE_KEYS = ["home", "immobilier", "anomalies", "credit", "sentiment"]

MODULE_ICONS = {
    "home": "🏠",
    "immobilier": "🏘️",
    "anomalies": "📡",
    "credit": "💳",
    "sentiment": "💬",
}

MODULE_TAGS = {
    "immobilier": ["Régression", "Géospatial", "Folium"],
    "anomalies": ["Anomaly Detection", "Time Series", "Unsupervised"],
    "credit": ["Classification", "SHAP", "Fairlearn"],
    "sentiment": ["NLP", "CamemBERT", "TF-IDF"],
}

# --- Sidebar ---
with st.sidebar:
    # Language selector
    lang_select = st.radio(
        "🌐",
        ["🇫🇷 FR", "🇬🇧 EN"],
        horizontal=True,
        label_visibility="collapsed",
        key="lang_selector",
    )
    lang = "en" if "EN" in lang_select else "fr"

    T = get_text("app", lang)

    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="font-size: 1.5rem; margin: 0;">{T['sidebar_title']}</h1>
        <p style="color: #B0B0B0; font-size: 0.85rem; margin: 5px 0 0 0;">
            {T['sidebar_subtitle']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Build display names from translations
    module_names = {
        "home": T["home_name"],
        "immobilier": T["immobilier_name"],
        "anomalies": T["anomalies_name"],
        "credit": T["credit_name"],
        "sentiment": T["sentiment_name"],
    }

    selected = st.radio(
        "Navigation",
        MODULE_KEYS,
        format_func=lambda x: f"{MODULE_ICONS[x]}  {module_names[x]}",
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


# --- Page d'accueil / Home page ---
def render_home():
    T = get_text("app", lang)

    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 10px;">
            {T['home_title']}
        </h1>
        <p style="color: #B0B0B0; font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
            {T['home_subtitle']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # Module descriptions
    module_descs = {
        "immobilier": T["immobilier_desc"],
        "anomalies": T["anomalies_desc"],
        "credit": T["credit_desc"],
        "sentiment": T["sentiment_desc"],
    }

    module_projects = {
        "immobilier": T["project_3"],
        "anomalies": T["project_4"],
        "credit": T["project_5"],
        "sentiment": T["project_6"],
    }

    cols = st.columns(2)
    card_keys = [k for k in MODULE_KEYS if k != "home"]

    for i, key in enumerate(card_keys):
        with cols[i % 2]:
            tags_html = " ".join(
                f'<span class="badge badge-info">{t}</span>' for t in MODULE_TAGS[key]
            )
            st.markdown(f"""
            <div class="metric-card" style="min-height: 160px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">{MODULE_ICONS[key]}</div>
                <h3 style="font-size: 1.1rem; color: #FAFAFA; font-weight: 600;">{module_names[key]}</h3>
                <p style="color: #B0B0B0; font-size: 0.85rem; margin: 8px 0;">
                    {module_descs[key]}
                </p>
                <div style="margin-top: 10px;">
                    <span class="badge badge-success">{module_projects[key]}</span>
                    {tags_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

    st.markdown("")

    st.markdown(f"""
    <div style="text-align: center; padding: 30px 0; color: #666; font-size: 0.85rem;">
        <p>{T['home_footer']}</p>
        <p style="margin-top: 10px;">
            Diogo Almeida · Data Scientist ·
            <a href="https://github.com/DiogoA78" target="_blank" style="color: #4F8BF9;">GitHub</a> ·
            <a href="https://linkedin.com/in/diogo-almeida0" target="_blank" style="color: #4F8BF9;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- Routeur ---
if selected == "home":
    render_home()
elif selected == "immobilier":
    from modules.immobilier import render
    render(lang)
elif selected == "anomalies":
    from modules.anomalies import render
    render(lang)
elif selected == "credit":
    from modules.credit import render
    render(lang)
elif selected == "sentiment":
    from modules.sentiment import render
    render(lang)
