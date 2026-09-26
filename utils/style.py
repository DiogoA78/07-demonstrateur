"""
Utilitaires de style partagés pour le Démonstrateur Data Science.
"""

import streamlit as st


# Palette de couleurs cohérente pour tous les modules
COLORS = {
    "primary": "#4F8BF9",
    "success": "#00C853",
    "danger": "#FF5252",
    "warning": "#FFD740",
    "info": "#40C4FF",
    "neutral": "#B0BEC5",
    "bg_card": "#1E1E2E",
    "bg_metric": "#262740",
    "text": "#FAFAFA",
    "text_secondary": "#B0B0B0",
}

# Palette pour les graphiques (séries multiples)
PLOT_COLORS = [
    "#4F8BF9",  # Bleu
    "#FF6B6B",  # Rouge
    "#51CF66",  # Vert
    "#FFD43B",  # Jaune
    "#CC5DE8",  # Violet
    "#FF922B",  # Orange
    "#20C997",  # Teal
    "#F06595",  # Rose
]


def apply_custom_css():
    """Applique le CSS personnalisé global."""
    st.markdown("""
    <style>
        /* Carte métrique */
        .metric-card {
            background: linear-gradient(135deg, #1E1E2E 0%, #262740 100%);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(79, 139, 249, 0.2);
            margin-bottom: 10px;
        }
        .metric-card h3 {
            color: #B0B0B0;
            font-size: 0.85rem;
            margin-bottom: 5px;
            font-weight: 400;
        }
        .metric-card .value {
            color: #FAFAFA;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .metric-card .delta {
            font-size: 0.8rem;
            margin-top: 5px;
        }
        .delta-positive { color: #00C853; }
        .delta-negative { color: #FF5252; }

        /* Section info */
        .info-box {
            background: rgba(79, 139, 249, 0.1);
            border-left: 4px solid #4F8BF9;
            border-radius: 0 8px 8px 0;
            padding: 15px 20px;
            margin: 15px 0;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-success { background: rgba(0, 200, 83, 0.2); color: #00C853; }
        .badge-danger { background: rgba(255, 82, 82, 0.2); color: #FF5252; }
        .badge-warning { background: rgba(255, 215, 64, 0.2); color: #FFD740; }
        .badge-info { background: rgba(64, 196, 255, 0.2); color: #40C4FF; }

        /* Header module */
        .module-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 25px;
            border: 1px solid rgba(79, 139, 249, 0.15);
        }
        .module-header h2 {
            margin: 0 0 8px 0;
            font-size: 1.6rem;
        }
        .module-header p {
            color: #B0B0B0;
            margin: 0;
            font-size: 0.95rem;
        }

        /* Séparateur discret */
        .separator {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(79,139,249,0.3), transparent);
            margin: 25px 0;
        }

        /* Sidebar améliorée */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
        }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label, value, delta=None, delta_type="positive"):
    """Affiche une carte métrique stylisée."""
    delta_html = ""
    if delta:
        css_class = f"delta-{delta_type}"
        arrow = "+" if delta_type == "positive" else ""  # removed arrows, just sign
        delta_html = f'<div class="delta {css_class}">{arrow}{delta}</div>'

    st.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def module_header(icon, title, description):
    """Affiche le header d'un module."""
    st.markdown(f"""
    <div class="module-header">
        <h2>{icon} {title}</h2>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def separator():
    """Affiche un séparateur discret."""
    st.markdown('<hr class="separator">', unsafe_allow_html=True)


def info_box(text):
    """Affiche une boîte d'information."""
    st.markdown(f"""
    <div class="info-box">
        {text}
    </div>
    """, unsafe_allow_html=True)
