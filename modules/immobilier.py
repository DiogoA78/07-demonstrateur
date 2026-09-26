"""
Module Immobilier IDF — Projet 3
================================
- Onglet 1 : Carte interactive des prix médians par commune
- Onglet 2 : Prédicteur de prix immobilier
- Onglet 3 : Statistiques du marché
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
import os

from utils.style import module_header, metric_card, separator, info_box, PLOT_COLORS

MODEL_PATH = os.path.join("models", "immobilier_model.pkl")
DATA_PATH = os.path.join("data", "immobilier_sample.csv")


def load_model():
    """Charge le modèle de prédiction immobilière."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def load_data():
    """Charge les données sample ou génère des données de démonstration."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return generate_demo_data()


def generate_demo_data():
    """Génère des données réalistes pour la démonstration."""
    np.random.seed(42)
    communes = {
        "Paris 1er": (48.8606, 2.3376, 12500),
        "Paris 6e": (48.8510, 2.3342, 14200),
        "Paris 8e": (48.8744, 2.3106, 13800),
        "Paris 16e": (48.8637, 2.2769, 11500),
        "Paris 11e": (48.8590, 2.3811, 10200),
        "Paris 18e": (48.8925, 2.3444, 8200),
        "Paris 19e": (48.8817, 2.3825, 7100),
        "Paris 20e": (48.8638, 2.3985, 7500),
        "Boulogne-Billancourt": (48.8397, 2.2399, 8900),
        "Neuilly-sur-Seine": (48.8847, 2.2688, 10800),
        "Saint-Germain-en-Laye": (48.8986, 2.0941, 6800),
        "Versailles": (48.8049, 2.1204, 7200),
        "Vincennes": (48.8474, 2.4386, 8100),
        "Saint-Denis": (48.9362, 2.3575, 4200),
        "Montreuil": (48.8634, 2.4484, 5800),
        "Créteil": (48.7909, 2.4551, 4500),
        "Nanterre": (48.8924, 2.2071, 5600),
        "Argenteuil": (48.9477, 2.2473, 3800),
        "Évry-Courcouronnes": (48.6249, 2.4299, 3200),
        "Cergy": (49.0364, 2.0608, 3500),
        "Massy": (48.7300, 2.2719, 4800),
        "Saint-Cloud": (48.8454, 2.2156, 8500),
        "Rueil-Malmaison": (48.8769, 2.1808, 7100),
        "Issy-les-Moulineaux": (48.8244, 2.2700, 8400),
        "Levallois-Perret": (48.8933, 2.2875, 9200),
    }

    rows = []
    for commune, (lat, lon, prix_m2_base) in communes.items():
        for _ in range(20):
            surface = np.random.randint(20, 150)
            nb_pieces = max(1, int(surface / 25) + np.random.randint(-1, 2))
            etage = np.random.randint(0, 8)
            annee_construction = np.random.randint(1850, 2024)
            prix_m2 = prix_m2_base * np.random.uniform(0.85, 1.15)
            prix = surface * prix_m2

            rows.append({
                "commune": commune,
                "latitude": lat + np.random.uniform(-0.005, 0.005),
                "longitude": lon + np.random.uniform(-0.005, 0.005),
                "surface": surface,
                "nb_pieces": nb_pieces,
                "etage": etage,
                "annee_construction": annee_construction,
                "prix_m2": round(prix_m2),
                "prix": round(prix),
            })

    return pd.DataFrame(rows)


def render():
    """Point d'entrée du module Immobilier IDF."""
    module_header(
        "🏘️",
        "Immobilier Île-de-France",
        "Exploration des prix immobiliers et prédiction par Machine Learning"
    )

    # Charger les données
    df = load_data()
    model = load_model()

    # Onglets
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Carte des prix",
        "🔮 Prédicteur de prix",
        "📊 Statistiques marché"
    ])

    # --- Onglet 1 : Carte interactive ---
    with tab1:
        render_map(df)

    # --- Onglet 2 : Prédicteur ---
    with tab2:
        render_predictor(df, model)

    # --- Onglet 3 : Statistiques ---
    with tab3:
        render_stats(df)


def render_map(df):
    """Carte interactive des prix médians par commune."""
    st.subheader("Prix médians par commune")

    # Agréger par commune
    agg = df.groupby("commune").agg(
        prix_m2_median=("prix_m2", "median"),
        prix_median=("prix", "median"),
        nb_transactions=("prix", "count"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()

    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Communes analysées", f"{len(agg)}")
    with col2:
        metric_card("Prix médian / m²", f"{int(agg['prix_m2_median'].median()):,} €".replace(",", " "))
    with col3:
        metric_card("Transactions", f"{len(df):,}".replace(",", " "))
    with col4:
        metric_card(
            "Écart min-max",
            f"{int(agg['prix_m2_median'].min()):,} - {int(agg['prix_m2_median'].max()):,} €/m²".replace(",", " ")
        )

    st.markdown("")

    # Carte interactive Folium
    center_lat = agg["latitude"].mean()
    center_lon = agg["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    # Échelle de couleur (vert = pas cher → rouge = cher)
    prix_min = agg["prix_m2_median"].min()
    prix_max = agg["prix_m2_median"].max()

    def get_color(prix):
        ratio = (prix - prix_min) / (prix_max - prix_min) if prix_max > prix_min else 0.5
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f"#{r:02x}{g:02x}40"

    for _, row in agg.iterrows():
        radius = max(8, min(25, row["nb_transactions"] / 2))
        color = get_color(row["prix_m2_median"])

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 13px; min-width: 180px;">
            <b>{row['commune']}</b><br>
            Prix médian / m² : <b>{int(row['prix_m2_median']):,} €</b><br>
            Prix médian : <b>{int(row['prix_median']):,} €</b><br>
            Transactions : {int(row['nb_transactions'])}
        </div>
        """.replace(",", " ")

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['commune']} — {int(row['prix_m2_median']):,} €/m²".replace(",", " "),
        ).add_to(m)

    st_folium(m, width="100%", height=550, returned_objects=[])

    # Classement
    separator()
    st.subheader("Classement des communes")

    sort_order = st.radio(
        "Trier par prix au m²",
        ["Plus cher d'abord", "Moins cher d'abord"],
        horizontal=True,
    )
    ascending = sort_order == "Moins cher d'abord"
    ranking = agg.sort_values("prix_m2_median", ascending=ascending)[
        ["commune", "prix_m2_median", "prix_median", "nb_transactions"]
    ].reset_index(drop=True)
    ranking.index += 1
    ranking.columns = ["Commune", "Prix médian / m² (€)", "Prix médian (€)", "Transactions"]

    st.dataframe(ranking, width="stretch", height=400)


def render_predictor(df, model):
    """Formulaire de prédiction de prix."""
    st.subheader("Estimer le prix d'un bien")

    communes = sorted(df["commune"].unique())

    if model is None:
        info_box(
            "⚠️ <strong>Mode démonstration</strong> — Le modèle entraîné n'est pas chargé. "
            "L'estimation utilise les statistiques du dataset. "
            "Pour activer le vrai modèle, placez le fichier <code>immobilier_model.pkl</code> "
            "dans le dossier <code>models/</code>."
        )

    col1, col2 = st.columns(2)

    with col1:
        commune = st.selectbox("Commune", communes, index=communes.index("Paris 6e") if "Paris 6e" in communes else 0)
        surface = st.slider("Surface (m²)", 10, 250, 65)
        nb_pieces = st.slider("Nombre de pièces", 1, 8, 3)

    with col2:
        etage = st.slider("Étage", 0, 15, 3)
        annee_construction = st.slider("Année de construction", 1800, 2025, 1970)

    separator()

    if st.button("🔮 Estimer le prix", type="primary", width="stretch"):
        if model is not None:
            # Prédiction avec le vrai modèle
            features = pd.DataFrame([{
                "surface": surface,
                "nb_pieces": nb_pieces,
                "etage": etage,
                "annee_construction": annee_construction,
            }])
            # Ajouter l'encodage de la commune si nécessaire
            try:
                prediction = model.predict(features)[0]
            except Exception:
                prediction = _estimate_from_data(df, commune, surface)
        else:
            prediction = _estimate_from_data(df, commune, surface)

        prix_m2 = prediction / surface

        st.markdown("")
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Prix estimé", f"{int(prediction):,} €".replace(",", " "))
        with col2:
            metric_card("Prix au m²", f"{int(prix_m2):,} €/m²".replace(",", " "))
        with col3:
            commune_median = df[df["commune"] == commune]["prix_m2"].median()
            diff_pct = ((prix_m2 - commune_median) / commune_median) * 100
            delta_type = "negative" if diff_pct > 0 else "positive"
            metric_card(
                "vs. médiane commune",
                f"{diff_pct:+.1f}%",
                delta=f"Médiane : {int(commune_median):,} €/m²".replace(",", " "),
                delta_type=delta_type,
            )

        # Contexte : distribution de la commune
        st.markdown("")
        commune_data = df[df["commune"] == commune]
        fig = px.histogram(
            commune_data, x="prix_m2", nbins=20,
            title=f"Distribution des prix au m² — {commune}",
            labels={"prix_m2": "Prix / m² (€)", "count": "Nombre de biens"},
            color_discrete_sequence=[PLOT_COLORS[0]],
        )
        fig.add_vline(x=prix_m2, line_dash="dash", line_color=PLOT_COLORS[1],
                      annotation_text=f"Votre estimation : {int(prix_m2):,} €/m²")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")


def _estimate_from_data(df, commune, surface):
    """Estimation simple basée sur le prix/m² médian de la commune."""
    commune_data = df[df["commune"] == commune]
    if len(commune_data) > 0:
        prix_m2 = commune_data["prix_m2"].median()
    else:
        prix_m2 = df["prix_m2"].median()
    return prix_m2 * surface


def render_stats(df):
    """Statistiques du marché immobilier."""
    st.subheader("Vue d'ensemble du marché")

    # Métriques globales
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Surface moyenne", f"{df['surface'].mean():.0f} m²")
    with col2:
        metric_card("Prix moyen", f"{int(df['prix'].mean()):,} €".replace(",", " "))
    with col3:
        metric_card("Pièces (moyenne)", f"{df['nb_pieces'].mean():.1f}")

    separator()

    # Distribution des prix par commune (box plot)
    top_communes = df.groupby("commune")["prix_m2"].median().nlargest(15).index
    df_top = df[df["commune"].isin(top_communes)]

    fig1 = px.box(
        df_top, x="commune", y="prix_m2",
        title="Distribution des prix au m² — Top 15 communes",
        labels={"commune": "", "prix_m2": "Prix / m² (€)"},
        color_discrete_sequence=[PLOT_COLORS[0]],
    )
    fig1.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig1, width="stretch")

    # Surface vs Prix (scatter)
    fig2 = px.scatter(
        df, x="surface", y="prix",
        color="commune",
        title="Surface vs. Prix de vente",
        labels={"surface": "Surface (m²)", "prix": "Prix (€)"},
        opacity=0.6,
        hover_data=["commune", "nb_pieces", "prix_m2"],
    )
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig2, width="stretch")

    # Répartition par nombre de pièces
    col1, col2 = st.columns(2)
    with col1:
        pieces_count = df["nb_pieces"].value_counts().sort_index()
        fig3 = px.bar(
            x=pieces_count.index, y=pieces_count.values,
            title="Répartition par nombre de pièces",
            labels={"x": "Nombre de pièces", "y": "Nombre de biens"},
            color_discrete_sequence=[PLOT_COLORS[2]],
        )
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, width="stretch")

    with col2:
        fig4 = px.scatter(
            df, x="annee_construction", y="prix_m2",
            title="Année de construction vs. Prix / m²",
            labels={"annee_construction": "Année", "prix_m2": "Prix / m² (€)"},
            opacity=0.5,
            color_discrete_sequence=[PLOT_COLORS[4]],
        )
        fig4.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, width="stretch")
