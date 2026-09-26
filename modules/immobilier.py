"""
Module Immobilier IDF — Projet 3
================================
- Onglet 1 : Carte interactive des prix médians par commune
- Onglet 2 : Prédicteur de prix immobilier
- Onglet 3 : Statistiques du marché
"""

import os

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from utils.style import PLOT_COLORS, metric_card, module_header, separator
from utils.translations import get_text

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

            rows.append(
                {
                    "commune": commune,
                    "latitude": lat + np.random.uniform(-0.005, 0.005),
                    "longitude": lon + np.random.uniform(-0.005, 0.005),
                    "surface": surface,
                    "nb_pieces": nb_pieces,
                    "etage": etage,
                    "annee_construction": annee_construction,
                    "prix_m2": round(prix_m2),
                    "prix": round(prix),
                }
            )

    return pd.DataFrame(rows)


def render(lang="fr"):
    """Point d'entrée du module Immobilier IDF."""
    T = get_text("immobilier", lang)

    module_header("🏘️", T["header_title"], T["header_desc"])

    # Charger les données
    df = load_data()
    model = load_model()

    # Onglets
    tab1, tab2, tab3 = st.tabs(
        [
            T["tab_map"],
            T["tab_predictor"],
            T["tab_stats"],
        ]
    )

    with tab1:
        render_map(df, T)

    with tab2:
        render_predictor(df, model, T)

    with tab3:
        render_stats(df, T)


def render_map(df, T):
    """Carte interactive des prix médians par commune."""
    st.subheader(T["map_title"])

    # Agréger par commune
    agg = (
        df.groupby("commune")
        .agg(
            prix_m2_median=("prix_m2", "median"),
            prix_median=("prix", "median"),
            nb_transactions=("prix", "count"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )

    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(T["metric_communes"], f"{len(agg)}")
    with col2:
        metric_card(
            T["metric_median_m2"],
            f"{int(agg['prix_m2_median'].median()):,} €".replace(",", " "),
        )
    with col3:
        metric_card(T["metric_transactions"], f"{len(df):,}".replace(",", " "))
    with col4:
        metric_card(
            T["metric_spread"],
            f"{int(agg['prix_m2_median'].min()):,} - {int(agg['prix_m2_median'].max()):,} €/m²".replace(
                ",", " "
            ),
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
        ratio = (
            (prix - prix_min) / (prix_max - prix_min) if prix_max > prix_min else 0.5
        )
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f"#{r:02x}{g:02x}40"

    for _, row in agg.iterrows():
        radius = max(8, min(25, row["nb_transactions"] / 2))
        color = get_color(row["prix_m2_median"])

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 13px; min-width: 180px;">
            <b>{row["commune"]}</b><br>
            {T["popup_median_m2"]} : <b>{int(row["prix_m2_median"]):,} €</b><br>
            {T["popup_median_price"]} : <b>{int(row["prix_median"]):,} €</b><br>
            {T["popup_transactions"]} : {int(row["nb_transactions"])}
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
            tooltip=f"{row['commune']} — {int(row['prix_m2_median']):,} €/m²".replace(
                ",", " "
            ),
        ).add_to(m)

    st_folium(m, width="100%", height=550, returned_objects=[])

    # Classement
    separator()
    st.subheader(T["ranking_title"])

    sort_order = st.radio(
        T["sort_label"],
        [T["sort_expensive"], T["sort_cheapest"]],
        horizontal=True,
    )
    ascending = sort_order == T["sort_cheapest"]
    ranking = agg.sort_values("prix_m2_median", ascending=ascending)[
        ["commune", "prix_m2_median", "prix_median", "nb_transactions"]
    ].reset_index(drop=True)
    ranking.index += 1
    ranking.columns = [
        T["col_commune"],
        T["col_median_m2"],
        T["col_median_price"],
        T["col_transactions"],
    ]

    st.dataframe(ranking, width="stretch", height=400)


def render_predictor(df, model, T):
    """Formulaire de prédiction de prix."""
    st.subheader(T["predictor_title"])

    communes = sorted(df["commune"].unique())

    col1, col2 = st.columns(2)

    with col1:
        commune = st.selectbox(
            T["label_commune"],
            communes,
            index=communes.index("Paris 6e") if "Paris 6e" in communes else 0,
        )
        surface = st.slider(T["label_surface"], 10, 250, 65)
        nb_pieces = st.slider(T["label_rooms"], 1, 8, 3)

    with col2:
        etage = st.slider(T["label_floor"], 0, 15, 3)
        annee_construction = st.slider(T["label_year"], 1800, 2025, 1970)

    separator()

    if st.button(T["btn_estimate"], type="primary", width="stretch"):
        if model is not None:
            features = pd.DataFrame(
                [
                    {
                        "surface": surface,
                        "nb_pieces": nb_pieces,
                        "etage": etage,
                        "annee_construction": annee_construction,
                    }
                ]
            )
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
            metric_card(T["result_price"], f"{int(prediction):,} €".replace(",", " "))
        with col2:
            metric_card(T["result_m2"], f"{int(prix_m2):,} €/m²".replace(",", " "))
        with col3:
            commune_median = df[df["commune"] == commune]["prix_m2"].median()
            diff_pct = ((prix_m2 - commune_median) / commune_median) * 100
            delta_type = "negative" if diff_pct > 0 else "positive"
            metric_card(
                T["result_vs_median"],
                f"{diff_pct:+.1f}%",
                delta=T["result_median_label"].format(
                    value=f"{int(commune_median):,}".replace(",", " ")
                ),
                delta_type=delta_type,
            )

        # Contexte : distribution de la commune
        st.markdown("")
        commune_data = df[df["commune"] == commune]
        fig = px.histogram(
            commune_data,
            x="prix_m2",
            nbins=20,
            title=T["chart_distribution"].format(commune=commune),
            labels={"prix_m2": T["axis_price_m2"], "count": T["axis_num_properties"]},
            color_discrete_sequence=[PLOT_COLORS[0]],
        )
        fig.add_vline(
            x=prix_m2,
            line_dash="dash",
            line_color=PLOT_COLORS[1],
            annotation_text=T["annotation_estimate"].format(
                value=f"{int(prix_m2):,}".replace(",", " ")
            ),
        )
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


def render_stats(df, T):
    """Statistiques du marché immobilier."""
    st.subheader(T["stats_title"])

    # Métriques globales
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(T["metric_avg_surface"], f"{df['surface'].mean():.0f} m²")
    with col2:
        metric_card(
            T["metric_avg_price"], f"{int(df['prix'].mean()):,} €".replace(",", " ")
        )
    with col3:
        metric_card(T["metric_avg_rooms"], f"{df['nb_pieces'].mean():.1f}")

    separator()

    # Distribution des prix par commune (box plot)
    top_communes = df.groupby("commune")["prix_m2"].median().nlargest(15).index
    df_top = df[df["commune"].isin(top_communes)]

    fig1 = px.box(
        df_top,
        x="commune",
        y="prix_m2",
        title=T["chart_box_top15"],
        labels={"commune": "", "prix_m2": T["axis_price_m2"]},
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
        df,
        x="surface",
        y="prix",
        color="commune",
        title=T["chart_scatter_title"],
        labels={"surface": T["axis_surface"], "prix": T["axis_price"]},
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
            x=pieces_count.index,
            y=pieces_count.values,
            title=T["chart_rooms_dist"],
            labels={"x": T["axis_rooms"], "y": T["axis_num_props"]},
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
            df,
            x="annee_construction",
            y="prix_m2",
            title=T["chart_year_price"],
            labels={
                "annee_construction": T["axis_year"],
                "prix_m2": T["axis_price_m2"],
            },
            opacity=0.5,
            color_discrete_sequence=[PLOT_COLORS[4]],
        )
        fig4.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, width="stretch")
