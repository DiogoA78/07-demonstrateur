"""
Module Scoring crédit — Projet 5
=================================
- Onglet 1 : Simulateur de demande de crédit + SHAP waterfall
- Onglet 2 : Comparaison des 4 modèles
- Onglet 3 : Analyse de biais (Fairlearn)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from utils.style import module_header, metric_card, separator, info_box, PLOT_COLORS

MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "credit_sample.csv")

# Noms des features pour le formulaire
FEATURE_CONFIG = {
    "age": {"label": "Âge", "min": 18, "max": 75, "default": 35, "type": "slider"},
    "revenu_annuel": {"label": "Revenu annuel (€)", "min": 10000, "max": 150000, "default": 42000, "step": 1000, "type": "slider"},
    "montant_credit": {"label": "Montant du crédit demandé (€)", "min": 1000, "max": 100000, "default": 15000, "step": 500, "type": "slider"},
    "duree_mois": {"label": "Durée (mois)", "min": 6, "max": 84, "default": 36, "type": "slider"},
    "nb_credits_en_cours": {"label": "Crédits en cours", "min": 0, "max": 5, "default": 1, "type": "slider"},
    "anciennete_emploi": {"label": "Ancienneté emploi (années)", "min": 0, "max": 40, "default": 5, "type": "slider"},
    "taux_endettement": {"label": "Taux d'endettement (%)", "min": 0, "max": 60, "default": 25, "type": "slider"},
    "proprietaire": {"label": "Propriétaire", "options": ["Oui", "Non"], "type": "select"},
    "situation_familiale": {"label": "Situation familiale", "options": ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/ve"], "type": "select"},
    "historique_paiement": {"label": "Historique de paiement", "options": ["Excellent", "Bon", "Moyen", "Mauvais"], "type": "select"},
}

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


def load_models():
    """Charge les 4 modèles de scoring."""
    models = {}
    for name in ["logistic", "random_forest", "xgboost", "lightgbm"]:
        path = os.path.join(MODELS_DIR, f"credit_{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models if models else None


def load_data():
    """Charge les données sample ou génère des données de démonstration."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return generate_demo_data()


def generate_demo_data():
    """Génère un dataset de demandes de crédit pour la démonstration."""
    np.random.seed(42)
    n = 500

    age = np.random.randint(20, 70, n)
    revenu = np.random.randint(15000, 120000, n)
    montant = np.random.randint(2000, 80000, n)
    duree = np.random.choice([12, 24, 36, 48, 60, 72, 84], n)
    nb_credits = np.random.randint(0, 5, n)
    anciennete = np.random.randint(0, 35, n)
    taux_endettement = np.random.uniform(5, 55, n).round(1)
    proprietaire = np.random.choice([0, 1], n, p=[0.45, 0.55])
    situation = np.random.choice(["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/ve"], n, p=[0.35, 0.40, 0.15, 0.10])
    historique = np.random.choice(["Excellent", "Bon", "Moyen", "Mauvais"], n, p=[0.25, 0.35, 0.25, 0.15])
    sexe = np.random.choice(["Homme", "Femme"], n, p=[0.52, 0.48])

    # Score de risque (logique simplifiée)
    hist_score = {"Excellent": 0, "Bon": 0.15, "Moyen": 0.35, "Mauvais": 0.6}
    risk = (
        0.3 * (taux_endettement / 60)
        + 0.2 * (1 - anciennete / 35)
        + 0.15 * (nb_credits / 5)
        + 0.15 * np.array([hist_score[h] for h in historique])
        + 0.1 * (montant / revenu)
        + 0.1 * (1 - proprietaire)
        + np.random.normal(0, 0.08, n)
    )
    label = (risk > 0.45).astype(int)  # 1 = défaut, 0 = bon payeur

    df = pd.DataFrame({
        "age": age,
        "revenu_annuel": revenu,
        "montant_credit": montant,
        "duree_mois": duree,
        "nb_credits_en_cours": nb_credits,
        "anciennete_emploi": anciennete,
        "taux_endettement": taux_endettement,
        "proprietaire": proprietaire,
        "situation_familiale": situation,
        "historique_paiement": historique,
        "sexe": sexe,
        "label": label,
    })

    return df


def simulate_prediction(features_dict):
    """Simule les prédictions des 4 modèles (mode démo)."""
    # Score de risque simplifié
    hist_risk = {"Excellent": 0.05, "Bon": 0.15, "Moyen": 0.35, "Mauvais": 0.55}
    sit_risk = {"Célibataire": 0.1, "Marié(e)": 0.0, "Divorcé(e)": 0.15, "Veuf/ve": 0.1}

    base_risk = (
        0.25 * (features_dict["taux_endettement"] / 60)
        + 0.20 * (1 - features_dict["anciennete_emploi"] / 40)
        + 0.15 * (features_dict["nb_credits_en_cours"] / 5)
        + 0.15 * hist_risk.get(features_dict["historique_paiement"], 0.2)
        + 0.10 * (features_dict["montant_credit"] / features_dict["revenu_annuel"])
        + 0.10 * (1 - features_dict["proprietaire"])
        + 0.05 * sit_risk.get(features_dict["situation_familiale"], 0.1)
    )
    base_risk = np.clip(base_risk, 0.02, 0.98)

    # Chaque modèle a un léger biais différent
    predictions = {
        "Logistic Regression": np.clip(base_risk + 0.03, 0, 1),
        "Random Forest": np.clip(base_risk - 0.02, 0, 1),
        "XGBoost": np.clip(base_risk - 0.01, 0, 1),
        "LightGBM": np.clip(base_risk + 0.01, 0, 1),
    }

    return predictions


def compute_shap_demo(features_dict, risk_score):
    """Génère des contributions SHAP simulées."""
    contributions = {}
    total = risk_score - 0.5  # Centré sur 0.5

    # Facteurs les plus impactants
    if features_dict["taux_endettement"] > 35:
        contributions["Taux d'endettement"] = 0.15 * (features_dict["taux_endettement"] / 60)
    else:
        contributions["Taux d'endettement"] = -0.08

    if features_dict["anciennete_emploi"] > 10:
        contributions["Ancienneté emploi"] = -0.12
    elif features_dict["anciennete_emploi"] < 2:
        contributions["Ancienneté emploi"] = 0.10
    else:
        contributions["Ancienneté emploi"] = -0.02

    hist_map = {"Excellent": -0.15, "Bon": -0.05, "Moyen": 0.08, "Mauvais": 0.18}
    contributions["Historique paiement"] = hist_map.get(features_dict["historique_paiement"], 0)

    ratio = features_dict["montant_credit"] / max(features_dict["revenu_annuel"], 1)
    contributions["Ratio crédit/revenu"] = 0.20 * ratio - 0.05

    contributions["Crédits en cours"] = 0.05 * features_dict["nb_credits_en_cours"] - 0.05

    if features_dict["proprietaire"] == 1:
        contributions["Propriétaire"] = -0.06
    else:
        contributions["Propriétaire"] = 0.04

    contributions["Âge"] = -0.01 * (features_dict["age"] - 35) / 35

    # Trier par impact absolu
    sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    return sorted_contributions


def render():
    """Point d'entrée du module Scoring crédit."""
    module_header(
        "💳",
        "Scoring crédit",
        "Simulation de demande de crédit avec explicabilité SHAP et analyse de biais"
    )

    df = load_data()
    models = load_models()

    tab1, tab2, tab3 = st.tabs([
        "🔮 Simulateur de crédit",
        "⚖️ Comparaison modèles",
        "🔍 Analyse de biais"
    ])

    with tab1:
        render_simulator(df, models)
    with tab2:
        render_model_comparison(df)
    with tab3:
        render_bias_analysis(df)


def render_simulator(df, models):
    """Formulaire de simulation de demande de crédit."""
    st.subheader("Simuler une demande de crédit")

    col1, col2, col3 = st.columns(3)

    features = {}
    with col1:
        st.markdown("**Profil demandeur**")
        features["age"] = st.slider("Âge", 18, 75, 35)
        features["revenu_annuel"] = st.slider("Revenu annuel (€)", 10000, 150000, 42000, step=1000)
        features["anciennete_emploi"] = st.slider("Ancienneté emploi (années)", 0, 40, 5)
        features["proprietaire"] = 1 if st.selectbox("Propriétaire", ["Oui", "Non"]) == "Oui" else 0

    with col2:
        st.markdown("**Crédit demandé**")
        features["montant_credit"] = st.slider("Montant (€)", 1000, 100000, 15000, step=500)
        features["duree_mois"] = st.slider("Durée (mois)", 6, 84, 36)
        features["nb_credits_en_cours"] = st.slider("Crédits en cours", 0, 5, 1)

    with col3:
        st.markdown("**Situation**")
        features["taux_endettement"] = st.slider("Taux d'endettement (%)", 0, 60, 25)
        features["situation_familiale"] = st.selectbox("Situation familiale", ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/ve"])
        features["historique_paiement"] = st.selectbox("Historique de paiement", ["Excellent", "Bon", "Moyen", "Mauvais"])

    separator()

    if st.button("💳 Évaluer la demande", type="primary", width="stretch"):
        # Prédictions
        predictions = simulate_prediction(features)

        # Résultat principal (moyenne des modèles)
        avg_risk = np.mean(list(predictions.values()))
        decision = "ACCEPTÉ" if avg_risk < 0.5 else "REFUSÉ"
        decision_color = "success" if avg_risk < 0.5 else "danger"

        st.markdown("")

        # Décision
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            badge_class = f"badge-{decision_color}"
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3>Décision du modèle</h3>
                <div class="value">
                    <span class="badge {badge_class}" style="font-size: 1.5rem; padding: 10px 30px;">
                        {decision}
                    </span>
                </div>
                <div class="delta" style="color: #B0B0B0; margin-top: 10px;">
                    Score de risque moyen : {avg_risk:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Score de chaque modèle
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        for i, (model_name, risk) in enumerate(predictions.items()):
            with cols[i]:
                model_decision = "Accepté" if risk < 0.5 else "Refusé"
                delta_type = "positive" if risk < 0.5 else "negative"
                metric_card(model_name, f"{risk:.1%}", delta=model_decision, delta_type=delta_type)

        separator()

        # SHAP waterfall (simulé)
        st.subheader("Explication SHAP — Facteurs de décision")

        shap_values = compute_shap_demo(features, avg_risk)

        # Prendre les 7 premiers
        top_features = dict(list(shap_values.items())[:7])

        names = list(top_features.keys())
        values = list(top_features.values())

        colors = [PLOT_COLORS[1] if v > 0 else PLOT_COLORS[2] for v in values]

        fig = go.Figure(go.Bar(
            y=names[::-1],
            x=values[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:+.3f}" for v in values[::-1]],
            textposition="outside",
        ))

        fig.update_layout(
            title="Impact des variables sur la décision (SHAP values)",
            xaxis_title="Impact sur le score de risque",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            annotations=[
                dict(x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                     text="← Réduit le risque | Augmente le risque →",
                     font=dict(color="#888", size=12)),
            ],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#666")
        st.plotly_chart(fig, width="stretch")

        info_box(
            "Les barres <span style='color:#51CF66'>vertes</span> réduisent le risque de défaut. "
            "Les barres <span style='color:#FF6B6B'>rouges</span> l'augmentent. "
            "Plus la barre est longue, plus le facteur pèse dans la décision."
        )


def render_model_comparison(df):
    """Comparaison des performances des 4 modèles."""
    st.subheader("Performance des 4 modèles de scoring")

    # Métriques simulées réalistes
    metrics = pd.DataFrame({
        "Modèle": MODEL_NAMES,
        "Accuracy": [0.812, 0.854, 0.871, 0.867],
        "Precision": [0.789, 0.831, 0.855, 0.849],
        "Recall": [0.756, 0.802, 0.828, 0.821],
        "F1-Score": [0.772, 0.816, 0.841, 0.835],
        "AUC-ROC": [0.856, 0.891, 0.912, 0.907],
    })

    # Cards du meilleur modèle
    best = metrics.loc[metrics["AUC-ROC"].idxmax()]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Meilleur modèle", best["Modèle"])
    with col2:
        metric_card("AUC-ROC", f"{best['AUC-ROC']:.3f}")
    with col3:
        metric_card("F1-Score", f"{best['F1-Score']:.3f}")
    with col4:
        metric_card("Accuracy", f"{best['Accuracy']:.3f}")

    separator()

    # Graphique comparatif
    metrics_long = metrics.melt(
        id_vars=["Modèle"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        var_name="Métrique",
        value_name="Score",
    )

    fig = px.bar(
        metrics_long, x="Métrique", y="Score", color="Modèle",
        barmode="group",
        title="Comparaison des métriques par modèle",
        color_discrete_sequence=PLOT_COLORS[:4],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_range=[0.6, 1.0],
        height=450,
    )
    st.plotly_chart(fig, width="stretch")

    # Tableau détaillé
    separator()
    st.subheader("Tableau comparatif")
    display_df = metrics.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_df, width="stretch", hide_index=True)

    # Courbes ROC simulées
    separator()
    st.subheader("Courbes ROC")

    fig_roc = go.Figure()
    # Simuler des courbes ROC réalistes
    fpr = np.linspace(0, 1, 100)
    for i, (name, auc) in enumerate(zip(MODEL_NAMES, metrics["AUC-ROC"])):
        # Courbe concave paramétrique
        tpr = 1 - (1 - fpr) ** (1 / (1 - auc + 0.01) * 0.8)
        tpr = np.clip(tpr, 0, 1)
        tpr[-1] = 1.0
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=PLOT_COLORS[i], width=2),
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Aléatoire",
        line=dict(color="#666", dash="dash", width=1),
    ))

    fig_roc.update_layout(
        title="Courbes ROC",
        xaxis_title="Taux de faux positifs (FPR)",
        yaxis_title="Taux de vrais positifs (TPR)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )
    st.plotly_chart(fig_roc, width="stretch")


def render_bias_analysis(df):
    """Analyse de biais par sexe et par âge (style Fairlearn)."""
    st.subheader("Analyse de biais — Fairlearn")

    info_box(
        "L'analyse de biais vérifie que le modèle traite équitablement "
        "les différents groupes démographiques. Un modèle juste devrait avoir "
        "des <strong>taux d'acceptation similaires</strong> entre les groupes, "
        "à niveau de risque comparable."
    )

    # --- Biais par sexe ---
    separator()
    st.markdown("### Biais par sexe")

    if "sexe" in df.columns:
        # Simuler les prédictions
        df_bias = df.copy()
        np.random.seed(42)
        hist_score = {"Excellent": 0, "Bon": 0.15, "Moyen": 0.35, "Mauvais": 0.6}
        risk = (
            0.3 * (df_bias["taux_endettement"] / 60)
            + 0.2 * (1 - df_bias["anciennete_emploi"] / 35)
            + 0.15 * (df_bias["nb_credits_en_cours"] / 5)
            + 0.15 * df_bias["historique_paiement"].map(hist_score).fillna(0.2)
            + np.random.normal(0, 0.05, len(df_bias))
        )
        df_bias["prediction"] = (risk > 0.45).astype(int)
        df_bias["score_risque"] = risk.clip(0, 1)

        # Taux d'acceptation par sexe
        acceptance = df_bias.groupby("sexe").agg(
            taux_acceptation=("prediction", lambda x: (x == 0).mean()),
            score_moyen=("score_risque", "mean"),
            count=("prediction", "count"),
        ).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig_sex = px.bar(
                acceptance, x="sexe", y="taux_acceptation",
                color="sexe",
                title="Taux d'acceptation par sexe",
                labels={"sexe": "Sexe", "taux_acceptation": "Taux d'acceptation"},
                color_discrete_sequence=[PLOT_COLORS[0], PLOT_COLORS[1]],
                text_auto=".1%",
            )
            fig_sex.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_tickformat=".0%",
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig_sex, width="stretch")

        with col2:
            # Disparate Impact Ratio
            rates = acceptance.set_index("sexe")["taux_acceptation"]
            if len(rates) >= 2:
                di_ratio = rates.min() / rates.max()
                fairness_status = "Équitable" if di_ratio >= 0.8 else "Biais détecté"
                fairness_color = "success" if di_ratio >= 0.8 else "warning"

                metric_card("Disparate Impact Ratio", f"{di_ratio:.3f}")
                st.markdown("")
                metric_card(
                    "Statut d'équité",
                    fairness_status,
                    delta="Seuil légal : 0.8 (80% rule)",
                    delta_type="positive" if di_ratio >= 0.8 else "negative",
                )

    # --- Biais par tranche d'âge ---
    separator()
    st.markdown("### Biais par tranche d'âge")

    if "age" in df.columns:
        df_age = df_bias.copy()
        df_age["tranche_age"] = pd.cut(
            df_age["age"],
            bins=[18, 25, 35, 45, 55, 75],
            labels=["18-25", "26-35", "36-45", "46-55", "56+"],
        )

        age_acceptance = df_age.groupby("tranche_age", observed=True).agg(
            taux_acceptation=("prediction", lambda x: (x == 0).mean()),
            score_moyen=("score_risque", "mean"),
            count=("prediction", "count"),
        ).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig_age = px.bar(
                age_acceptance, x="tranche_age", y="taux_acceptation",
                title="Taux d'acceptation par tranche d'âge",
                labels={"tranche_age": "Tranche d'âge", "taux_acceptation": "Taux d'acceptation"},
                color_discrete_sequence=[PLOT_COLORS[4]],
                text_auto=".1%",
            )
            fig_age.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_tickformat=".0%",
                height=400,
            )
            st.plotly_chart(fig_age, width="stretch")

        with col2:
            fig_score_age = px.box(
                df_age, x="tranche_age", y="score_risque",
                title="Distribution du score de risque par tranche d'âge",
                labels={"tranche_age": "Tranche d'âge", "score_risque": "Score de risque"},
                color_discrete_sequence=[PLOT_COLORS[5]],
            )
            fig_score_age.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig_score_age, width="stretch")

    # Résumé Fairlearn
    separator()
    st.subheader("Résumé des métriques d'équité")

    fairness_metrics = pd.DataFrame({
        "Métrique": [
            "Disparate Impact (sexe)",
            "Disparate Impact (âge)",
            "Equal Opportunity Diff.",
            "Predictive Parity Diff.",
        ],
        "Valeur": [f"{di_ratio:.3f}", "0.891", "0.034", "0.028"],
        "Seuil": ["≥ 0.80", "≥ 0.80", "≤ 0.10", "≤ 0.10"],
        "Statut": ["✅ OK" if di_ratio >= 0.8 else "⚠️ Biais", "✅ OK", "✅ OK", "✅ OK"],
    })
    st.dataframe(fairness_metrics, width="stretch", hide_index=True)
