"""
Module Anomalies capteurs — Projet 4
=====================================
- Onglet 1 : Visualisation des capteurs avec anomalies colorées
- Onglet 2 : Comparaison des 4 méthodes de détection
- Onglet 3 : Métriques de performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

from utils.style import module_header, metric_card, separator, info_box, PLOT_COLORS

MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "anomalies_sample.csv")

METHOD_NAMES = {
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "ocsvm": "One-Class SVM",
    "autoencoder": "Autoencoder",
}

METHOD_COLORS = {
    "Isolation Forest": PLOT_COLORS[0],
    "Local Outlier Factor": PLOT_COLORS[1],
    "One-Class SVM": PLOT_COLORS[2],
    "Autoencoder": PLOT_COLORS[3],
}


def load_data():
    """Charge les données de capteurs ou génère des données de démonstration."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return generate_demo_data()


def generate_demo_data():
    """Génère des données de capteurs réalistes avec anomalies injectées."""
    np.random.seed(42)
    n_points = 2000

    # Séries temporelles de capteurs
    t = np.arange(n_points)
    sensor1 = 50 + 10 * np.sin(2 * np.pi * t / 200) + np.random.normal(0, 2, n_points)
    sensor2 = 30 + 5 * np.cos(2 * np.pi * t / 150) + np.random.normal(0, 1.5, n_points)
    sensor3 = 70 + 8 * np.sin(2 * np.pi * t / 300) + np.random.normal(0, 3, n_points)

    # Injecter des anomalies
    anomaly_true = np.zeros(n_points, dtype=int)
    anomaly_category = np.array(["normal"] * n_points)

    # Anomalies de type valve1 (pics soudains)
    valve1_idx = np.random.choice(range(200, 500), 25, replace=False)
    sensor1[valve1_idx] += np.random.uniform(25, 45, len(valve1_idx))
    anomaly_true[valve1_idx] = 1
    anomaly_category[valve1_idx] = "valve1"

    # Anomalies de type valve2 (dérive progressive)
    valve2_start = 800
    valve2_idx = np.arange(valve2_start, valve2_start + 60)
    sensor2[valve2_idx] += np.linspace(0, 30, 60)
    anomaly_true[valve2_idx] = 1
    anomaly_category[valve2_idx] = "valve2"

    # Anomalies "other" (bruit simultané multi-capteurs)
    other_idx = np.random.choice(range(1200, 1600), 30, replace=False)
    sensor1[other_idx] += np.random.uniform(-20, -10, len(other_idx))
    sensor3[other_idx] += np.random.uniform(20, 35, len(other_idx))
    anomaly_true[other_idx] = 1
    anomaly_category[other_idx] = "other"

    # Prédictions simulées des 4 méthodes (avec des profils de performance distincts)
    pred_if = anomaly_true.copy()
    pred_lof = anomaly_true.copy()
    pred_ocsvm = anomaly_true.copy()
    pred_ae = anomaly_true.copy()

    # Isolation Forest : bon sur valve1, quelques faux positifs
    fp_if = np.random.choice(np.where(anomaly_true == 0)[0], 15, replace=False)
    pred_if[fp_if] = 1
    fn_if = np.random.choice(valve2_idx[:10], 5, replace=False)
    pred_if[fn_if] = 0

    # LOF : bon sur valve2, rate quelques valve1
    fn_lof = np.random.choice(valve1_idx[:8], 4, replace=False)
    pred_lof[fn_lof] = 0
    fp_lof = np.random.choice(np.where(anomaly_true == 0)[0], 20, replace=False)
    pred_lof[fp_lof] = 1

    # OCSVM : beaucoup de faux positifs
    fp_ocsvm = np.random.choice(np.where(anomaly_true == 0)[0], 40, replace=False)
    pred_ocsvm[fp_ocsvm] = 1
    fn_ocsvm = np.random.choice(other_idx[:10], 3, replace=False)
    pred_ocsvm[fn_ocsvm] = 0

    # Autoencoder : le meilleur globalement
    fp_ae = np.random.choice(np.where(anomaly_true == 0)[0], 8, replace=False)
    pred_ae[fp_ae] = 1
    fn_ae = np.random.choice(valve2_idx[:5], 2, replace=False)
    pred_ae[fn_ae] = 0

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_points, freq="h"),
        "sensor1_temperature": np.round(sensor1, 2),
        "sensor2_pressure": np.round(sensor2, 2),
        "sensor3_vibration": np.round(sensor3, 2),
        "anomaly_true": anomaly_true,
        "anomaly_category": anomaly_category,
        "pred_isolation_forest": pred_if,
        "pred_lof": pred_lof,
        "pred_ocsvm": pred_ocsvm,
        "pred_autoencoder": pred_ae,
    })

    return df


def compute_metrics(y_true, y_pred):
    """Calcule Precision, Recall, F1 sans dépendance sklearn."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def render():
    """Point d'entrée du module Anomalies capteurs."""
    module_header(
        "📡",
        "Anomalies capteurs industriels",
        "Détection d'anomalies sur données de capteurs — comparaison de 4 méthodes"
    )

    df = load_data()

    tab1, tab2, tab3 = st.tabs([
        "📈 Visualisation capteurs",
        "🔍 Comparaison méthodes",
        "📊 Métriques de performance"
    ])

    with tab1:
        render_sensors(df)
    with tab2:
        render_comparison(df)
    with tab3:
        render_metrics(df)


def render_sensors(df):
    """Visualisation des séries temporelles avec anomalies colorées."""
    st.subheader("Séries temporelles des capteurs")

    # Métriques
    n_anomalies = int(df["anomaly_true"].sum())
    n_total = len(df)
    pct_anomalies = n_anomalies / n_total * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Points de mesure", f"{n_total:,}".replace(",", " "))
    with col2:
        metric_card("Anomalies détectées", f"{n_anomalies}")
    with col3:
        metric_card("Taux d'anomalies", f"{pct_anomalies:.1f}%")
    with col4:
        categories = df[df["anomaly_true"] == 1]["anomaly_category"].nunique()
        metric_card("Types d'anomalies", f"{categories}")

    st.markdown("")

    # Filtre par catégorie
    categories_list = ["Toutes"] + sorted(df[df["anomaly_category"] != "normal"]["anomaly_category"].unique().tolist())
    cat_filter = st.selectbox("Filtrer par catégorie d'anomalie", categories_list)

    # Sélection du capteur
    sensors = {
        "sensor1_temperature": "Capteur 1 — Température (°C)",
        "sensor2_pressure": "Capteur 2 — Pression (bar)",
        "sensor3_vibration": "Capteur 3 — Vibration (mm/s)",
    }
    sensor_key = st.selectbox(
        "Capteur",
        list(sensors.keys()),
        format_func=lambda x: sensors[x],
    )

    # Préparer les données
    df_plot = df.copy()
    if cat_filter != "Toutes":
        mask_anomaly = (df_plot["anomaly_category"] == cat_filter)
    else:
        mask_anomaly = df_plot["anomaly_true"] == 1

    # Graphique
    fig = go.Figure()

    # Ligne principale
    fig.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot[sensor_key],
        mode="lines",
        name="Signal normal",
        line=dict(color=PLOT_COLORS[0], width=1),
        opacity=0.7,
    ))

    # Points d'anomalie
    df_anom = df_plot[mask_anomaly]
    if len(df_anom) > 0:
        color_map = {"valve1": PLOT_COLORS[1], "valve2": PLOT_COLORS[3], "other": PLOT_COLORS[4]}
        for cat in df_anom["anomaly_category"].unique():
            df_cat = df_anom[df_anom["anomaly_category"] == cat]
            fig.add_trace(go.Scatter(
                x=df_cat["timestamp"], y=df_cat[sensor_key],
                mode="markers",
                name=f"Anomalie — {cat}",
                marker=dict(
                    color=color_map.get(cat, PLOT_COLORS[1]),
                    size=8,
                    symbol="x",
                    line=dict(width=1, color="white"),
                ),
            ))

    fig.update_layout(
        title=sensors[sensor_key],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        legend=dict(orientation="h", y=-0.15),
        xaxis_title="",
        yaxis_title=sensor_key.split("_")[-1].capitalize(),
    )
    st.plotly_chart(fig, width="stretch")

    # Distribution des anomalies par catégorie
    separator()
    col1, col2 = st.columns(2)
    with col1:
        cat_counts = df[df["anomaly_true"] == 1]["anomaly_category"].value_counts()
        fig_pie = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title="Répartition par type d'anomalie",
            color_discrete_sequence=PLOT_COLORS[1:],
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col2:
        # Timeline des anomalies
        df_anom_all = df[df["anomaly_true"] == 1].copy()
        df_anom_all["hour"] = df_anom_all["timestamp"].dt.hour if hasattr(df_anom_all["timestamp"].dt, "hour") else pd.to_datetime(df_anom_all["timestamp"]).dt.hour
        hour_counts = df_anom_all["hour"].value_counts().sort_index()
        fig_bar = px.bar(
            x=hour_counts.index, y=hour_counts.values,
            title="Anomalies par heure de la journée",
            labels={"x": "Heure", "y": "Nombre d'anomalies"},
            color_discrete_sequence=[PLOT_COLORS[1]],
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, width="stretch")


def render_comparison(df):
    """Comparaison côte à côte des 4 méthodes de détection."""
    st.subheader("Comparaison des méthodes de détection")

    info_box(
        "Chaque graphique montre la même série temporelle avec les anomalies détectées "
        "par chaque méthode. Les <strong>faux positifs</strong> (détections incorrectes) "
        "sont en <span style='color:#FFD740'>jaune</span>, les <strong>vrais positifs</strong> "
        "en <span style='color:#FF5252'>rouge</span>."
    )

    sensor_key = "sensor1_temperature"

    methods = ["pred_isolation_forest", "pred_lof", "pred_ocsvm", "pred_autoencoder"]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[METHOD_NAMES[m.replace("pred_", "")] for m in methods],
        vertical_spacing=0.06,
    )

    for i, method in enumerate(methods, 1):
        # Signal
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[sensor_key],
            mode="lines",
            line=dict(color=PLOT_COLORS[0], width=0.8),
            opacity=0.5,
            showlegend=(i == 1),
            name="Signal",
        ), row=i, col=1)

        # Vrais positifs
        tp_mask = (df["anomaly_true"] == 1) & (df[method] == 1)
        fig.add_trace(go.Scatter(
            x=df.loc[tp_mask, "timestamp"], y=df.loc[tp_mask, sensor_key],
            mode="markers",
            marker=dict(color=PLOT_COLORS[1], size=5, symbol="circle"),
            showlegend=(i == 1),
            name="Vrai positif",
        ), row=i, col=1)

        # Faux positifs
        fp_mask = (df["anomaly_true"] == 0) & (df[method] == 1)
        fig.add_trace(go.Scatter(
            x=df.loc[fp_mask, "timestamp"], y=df.loc[fp_mask, sensor_key],
            mode="markers",
            marker=dict(color=PLOT_COLORS[3], size=5, symbol="diamond"),
            showlegend=(i == 1),
            name="Faux positif",
        ), row=i, col=1)

        # Faux négatifs
        fn_mask = (df["anomaly_true"] == 1) & (df[method] == 0)
        fig.add_trace(go.Scatter(
            x=df.loc[fn_mask, "timestamp"], y=df.loc[fn_mask, sensor_key],
            mode="markers",
            marker=dict(color=PLOT_COLORS[2], size=5, symbol="triangle-up"),
            showlegend=(i == 1),
            name="Faux négatif (raté)",
        ), row=i, col=1)

    fig.update_layout(
        height=900,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")


def render_metrics(df):
    """Tableau comparatif des métriques de performance."""
    st.subheader("Métriques de performance")

    methods = ["pred_isolation_forest", "pred_lof", "pred_ocsvm", "pred_autoencoder"]
    y_true = df["anomaly_true"].values

    results = []
    for method in methods:
        name = METHOD_NAMES[method.replace("pred_", "")]
        y_pred = df[method].values
        m = compute_metrics(y_true, y_pred)
        m["method"] = name
        results.append(m)

    df_metrics = pd.DataFrame(results)

    # Identifier le meilleur F1
    best_f1_idx = df_metrics["f1"].idxmax()
    best_method = df_metrics.loc[best_f1_idx, "method"]

    # Cards
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    for i, row in df_metrics.iterrows():
        with cols[i]:
            is_best = row["method"] == best_method
            badge = " 🏆" if is_best else ""
            metric_card(
                row["method"] + badge,
                f"F1 = {row['f1']:.3f}",
                delta=f"P={row['precision']:.3f} | R={row['recall']:.3f}",
                delta_type="positive" if is_best else "neutral",
            )

    separator()

    # Graphique radar
    categories = ["Precision", "Recall", "F1-Score"]
    fig = go.Figure()

    for i, row in df_metrics.iterrows():
        values = [row["precision"], row["recall"], row["f1"]]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["method"],
            line=dict(color=METHOD_COLORS.get(row["method"], PLOT_COLORS[i])),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="rgba(0,0,0,0)",
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title="Comparaison Radar — Precision / Recall / F1",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

    # Tableau détaillé
    separator()
    st.subheader("Tableau détaillé")

    df_display = df_metrics[["method", "precision", "recall", "f1", "tp", "fp", "fn"]].copy()
    df_display.columns = ["Méthode", "Precision", "Recall", "F1-Score", "Vrais Positifs", "Faux Positifs", "Faux Négatifs"]
    df_display["Precision"] = df_display["Precision"].apply(lambda x: f"{x:.3f}")
    df_display["Recall"] = df_display["Recall"].apply(lambda x: f"{x:.3f}")
    df_display["F1-Score"] = df_display["F1-Score"].apply(lambda x: f"{x:.3f}")

    st.dataframe(df_display, width="stretch", hide_index=True)

    # Matrice de confusion par méthode
    separator()
    st.subheader("Matrices de confusion")

    cols = st.columns(4)
    for i, method in enumerate(methods):
        name = METHOD_NAMES[method.replace("pred_", "")]
        y_pred = df[method].values
        m = compute_metrics(y_true, y_pred)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        with cols[i]:
            st.markdown(f"**{name}**")
            conf_df = pd.DataFrame(
                [[tn, m["fp"]], [m["fn"], m["tp"]]],
                index=["Prédit Normal", "Prédit Anomalie"],
                columns=["Vrai Normal", "Vrai Anomalie"],
            )
            # Correction : inversion pour l'affichage standard
            conf_df = pd.DataFrame(
                [[m["tp"], m["fn"]], [m["fp"], tn]],
                index=["Prédit Anomalie", "Prédit Normal"],
                columns=["Vrai Anomalie", "Vrai Normal"],
            )
            st.dataframe(conf_df, width="stretch")
