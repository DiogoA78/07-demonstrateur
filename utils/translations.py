"""
Translations for the Data Science Demonstrator.
Supports French (fr) and English (en).
"""

TRANSLATIONS = {
    # =========================================================================
    # APP — Main application (app.py)
    # =========================================================================
    "app": {
        "fr": {
            "page_title": "Démonstrateur Data Science",
            "sidebar_title": "🔬 Démonstrateur",
            "sidebar_subtitle": "Data Science Portfolio",
            "home_title": "🔬 Démonstrateur Data Science",
            "home_subtitle": (
                "Application interactive regroupant les algorithmes de mon portfolio. "
                "Explorez chaque module pour tester les modèles en temps réel."
            ),
            "home_footer": "Sélectionnez un module dans la barre latérale pour commencer.",
            # Module names
            "home_name": "Accueil",
            "immobilier_name": "Immobilier IDF",
            "anomalies_name": "Anomalies capteurs",
            "credit_name": "Scoring crédit",
            "sentiment_name": "Sentiment NLP",
            # Module descriptions
            "home_desc": "Présentation du démonstrateur",
            "immobilier_desc": "Prédiction de prix immobilier en Île-de-France",
            "anomalies_desc": "Détection d'anomalies sur données de capteurs industriels",
            "credit_desc": "Scoring de demandes de crédit avec explicabilité SHAP",
            "sentiment_desc": "Analyse de sentiment sur critiques de films (FR)",
            # Project labels
            "project_3": "Projet 3",
            "project_4": "Projet 4",
            "project_5": "Projet 5",
            "project_6": "Projet 6",
        },
        "en": {
            "page_title": "Data Science Demonstrator",
            "sidebar_title": "🔬 Demonstrator",
            "sidebar_subtitle": "Data Science Portfolio",
            "home_title": "🔬 Data Science Demonstrator",
            "home_subtitle": (
                "Interactive application showcasing the algorithms from my portfolio. "
                "Explore each module to test models in real time."
            ),
            "home_footer": "Select a module in the sidebar to get started.",
            "home_name": "Home",
            "immobilier_name": "Real Estate IDF",
            "anomalies_name": "Sensor Anomalies",
            "credit_name": "Credit Scoring",
            "sentiment_name": "Sentiment NLP",
            "home_desc": "Demonstrator overview",
            "immobilier_desc": "Real estate price prediction in Île-de-France",
            "anomalies_desc": "Anomaly detection on industrial sensor data",
            "credit_desc": "Credit scoring with SHAP explainability",
            "sentiment_desc": "Sentiment analysis on French movie reviews",
            "project_3": "Project 3",
            "project_4": "Project 4",
            "project_5": "Project 5",
            "project_6": "Project 6",
        },
    },
    # =========================================================================
    # IMMOBILIER — Real Estate IDF (modules/immobilier.py)
    # =========================================================================
    "immobilier": {
        "fr": {
            "header_title": "Immobilier Île-de-France",
            "header_desc": "Exploration des prix immobiliers et prédiction par Machine Learning",
            "tab_map": "🗺️ Carte des prix",
            "tab_predictor": "🔮 Prédicteur de prix",
            "tab_stats": "📊 Statistiques marché",
            # Map tab
            "map_title": "Prix médians par commune",
            "metric_communes": "Communes analysées",
            "metric_median_m2": "Prix médian / m²",
            "metric_transactions": "Transactions",
            "metric_spread": "Écart min-max",
            "popup_median_m2": "Prix médian / m²",
            "popup_median_price": "Prix médian",
            "popup_transactions": "Transactions",
            "ranking_title": "Classement des communes",
            "sort_label": "Trier par prix au m²",
            "sort_expensive": "Plus cher d'abord",
            "sort_cheapest": "Moins cher d'abord",
            "col_commune": "Commune",
            "col_median_m2": "Prix médian / m² (€)",
            "col_median_price": "Prix médian (€)",
            "col_transactions": "Transactions",
            # Predictor tab
            "predictor_title": "Estimer le prix d'un bien",
            "label_commune": "Commune",
            "label_surface": "Surface (m²)",
            "label_rooms": "Nombre de pièces",
            "label_floor": "Étage",
            "label_year": "Année de construction",
            "btn_estimate": "🔮 Estimer le prix",
            "result_price": "Prix estimé",
            "result_m2": "Prix au m²",
            "result_vs_median": "vs. médiane commune",
            "result_median_label": "Médiane : {value} €/m²",
            "chart_distribution": "Distribution des prix au m² — {commune}",
            "axis_price_m2": "Prix / m² (€)",
            "axis_num_properties": "Nombre de biens",
            "annotation_estimate": "Votre estimation : {value} €/m²",
            # Stats tab
            "stats_title": "Vue d'ensemble du marché",
            "metric_avg_surface": "Surface moyenne",
            "metric_avg_price": "Prix moyen",
            "metric_avg_rooms": "Pièces (moyenne)",
            "chart_box_top15": "Distribution des prix au m² — Top 15 communes",
            "chart_scatter_title": "Surface vs. Prix de vente",
            "axis_surface": "Surface (m²)",
            "axis_price": "Prix (€)",
            "chart_rooms_dist": "Répartition par nombre de pièces",
            "axis_rooms": "Nombre de pièces",
            "axis_num_props": "Nombre de biens",
            "chart_year_price": "Année de construction vs. Prix / m²",
            "axis_year": "Année",
        },
        "en": {
            "header_title": "Île-de-France Real Estate",
            "header_desc": "Real estate price exploration and Machine Learning prediction",
            "tab_map": "🗺️ Price Map",
            "tab_predictor": "🔮 Price Predictor",
            "tab_stats": "📊 Market Statistics",
            "map_title": "Median prices by municipality",
            "metric_communes": "Municipalities analyzed",
            "metric_median_m2": "Median price / m²",
            "metric_transactions": "Transactions",
            "metric_spread": "Min-max spread",
            "popup_median_m2": "Median price / m²",
            "popup_median_price": "Median price",
            "popup_transactions": "Transactions",
            "ranking_title": "Municipality ranking",
            "sort_label": "Sort by price per m²",
            "sort_expensive": "Most expensive first",
            "sort_cheapest": "Least expensive first",
            "col_commune": "Municipality",
            "col_median_m2": "Median price / m² (€)",
            "col_median_price": "Median price (€)",
            "col_transactions": "Transactions",
            "predictor_title": "Estimate a property's price",
            "label_commune": "Municipality",
            "label_surface": "Area (m²)",
            "label_rooms": "Number of rooms",
            "label_floor": "Floor",
            "label_year": "Year built",
            "btn_estimate": "🔮 Estimate price",
            "result_price": "Estimated price",
            "result_m2": "Price per m²",
            "result_vs_median": "vs. municipality median",
            "result_median_label": "Median: {value} €/m²",
            "chart_distribution": "Price per m² distribution — {commune}",
            "axis_price_m2": "Price / m² (€)",
            "axis_num_properties": "Number of properties",
            "annotation_estimate": "Your estimate: {value} €/m²",
            "stats_title": "Market overview",
            "metric_avg_surface": "Average area",
            "metric_avg_price": "Average price",
            "metric_avg_rooms": "Rooms (average)",
            "chart_box_top15": "Price per m² distribution — Top 15 municipalities",
            "chart_scatter_title": "Area vs. Sale price",
            "axis_surface": "Area (m²)",
            "axis_price": "Price (€)",
            "chart_rooms_dist": "Distribution by number of rooms",
            "axis_rooms": "Number of rooms",
            "axis_num_props": "Number of properties",
            "chart_year_price": "Year built vs. Price / m²",
            "axis_year": "Year",
        },
    },
    # =========================================================================
    # ANOMALIES — Sensor Anomalies (modules/anomalies.py)
    # =========================================================================
    "anomalies": {
        "fr": {
            "header_title": "Anomalies capteurs industriels",
            "header_desc": "Détection d'anomalies sur données de capteurs — comparaison de 4 méthodes",
            "tab_sensors": "📈 Visualisation capteurs",
            "tab_comparison": "🔍 Comparaison méthodes",
            "tab_metrics": "📊 Métriques de performance",
            # Sensors tab
            "sensors_title": "Séries temporelles des capteurs",
            "metric_datapoints": "Points de mesure",
            "metric_anomalies": "Anomalies détectées",
            "metric_anomaly_rate": "Taux d'anomalies",
            "metric_anomaly_types": "Types d'anomalies",
            "filter_all": "Toutes",
            "filter_label": "Filtrer par catégorie d'anomalie",
            "sensor_label": "Capteur",
            "sensor1": "Capteur 1 — Température (°C)",
            "sensor2": "Capteur 2 — Pression (bar)",
            "sensor3": "Capteur 3 — Vibration (mm/s)",
            "legend_normal": "Signal normal",
            "legend_anomaly": "Anomalie — {cat}",
            "chart_pie_title": "Répartition par type d'anomalie",
            "chart_hour_title": "Anomalies par heure de la journée",
            "axis_hour": "Heure",
            "axis_num_anomalies": "Nombre d'anomalies",
            # Comparison tab
            "comparison_title": "Comparaison des méthodes de détection",
            "comparison_info": (
                "Chaque graphique montre la même série temporelle avec les anomalies détectées "
                "par chaque méthode. Les <strong>faux positifs</strong> (détections incorrectes) "
                "sont en <span style='color:#FFD740'>jaune</span>, les <strong>vrais positifs</strong> "
                "en <span style='color:#FF5252'>rouge</span>."
            ),
            "legend_signal": "Signal",
            "legend_true_pos": "Vrai positif",
            "legend_false_pos": "Faux positif",
            "legend_false_neg": "Faux négatif (raté)",
            # Metrics tab
            "metrics_title": "Métriques de performance",
            "radar_title": "Comparaison Radar — Precision / Recall / F1",
            "table_title": "Tableau détaillé",
            "col_method": "Méthode",
            "col_tp": "Vrais Positifs",
            "col_fp": "Faux Positifs",
            "col_fn": "Faux Négatifs",
            "confusion_title": "Matrices de confusion",
            "pred_anomaly": "Prédit Anomalie",
            "pred_normal": "Prédit Normal",
            "true_anomaly": "Vrai Anomalie",
            "true_normal": "Vrai Normal",
        },
        "en": {
            "header_title": "Industrial Sensor Anomalies",
            "header_desc": "Anomaly detection on sensor data — comparison of 4 methods",
            "tab_sensors": "📈 Sensor Visualization",
            "tab_comparison": "🔍 Method Comparison",
            "tab_metrics": "📊 Performance Metrics",
            "sensors_title": "Sensor Time Series",
            "metric_datapoints": "Data points",
            "metric_anomalies": "Anomalies detected",
            "metric_anomaly_rate": "Anomaly rate",
            "metric_anomaly_types": "Anomaly types",
            "filter_all": "All",
            "filter_label": "Filter by anomaly category",
            "sensor_label": "Sensor",
            "sensor1": "Sensor 1 — Temperature (°C)",
            "sensor2": "Sensor 2 — Pressure (bar)",
            "sensor3": "Sensor 3 — Vibration (mm/s)",
            "legend_normal": "Normal signal",
            "legend_anomaly": "Anomaly — {cat}",
            "chart_pie_title": "Distribution by anomaly type",
            "chart_hour_title": "Anomalies by hour of day",
            "axis_hour": "Hour",
            "axis_num_anomalies": "Number of anomalies",
            "comparison_title": "Comparison of detection methods",
            "comparison_info": (
                "Each chart shows the same time series with anomalies detected "
                "by each method. <strong>False positives</strong> (incorrect detections) "
                "are in <span style='color:#FFD740'>yellow</span>, <strong>true positives</strong> "
                "in <span style='color:#FF5252'>red</span>."
            ),
            "legend_signal": "Signal",
            "legend_true_pos": "True positive",
            "legend_false_pos": "False positive",
            "legend_false_neg": "False negative (missed)",
            "metrics_title": "Performance Metrics",
            "radar_title": "Radar Comparison — Precision / Recall / F1",
            "table_title": "Detailed Table",
            "col_method": "Method",
            "col_tp": "True Positives",
            "col_fp": "False Positives",
            "col_fn": "False Negatives",
            "confusion_title": "Confusion Matrices",
            "pred_anomaly": "Predicted Anomaly",
            "pred_normal": "Predicted Normal",
            "true_anomaly": "True Anomaly",
            "true_normal": "True Normal",
        },
    },
    # =========================================================================
    # CREDIT — Credit Scoring (modules/credit.py)
    # =========================================================================
    "credit": {
        "fr": {
            "header_title": "Scoring crédit",
            "header_desc": "Simulation de demande de crédit avec explicabilité SHAP et analyse de biais",
            "tab_simulator": "🔮 Simulateur de crédit",
            "tab_models": "⚖️ Comparaison modèles",
            "tab_bias": "🔍 Analyse de biais",
            # Simulator
            "simulator_title": "Simuler une demande de crédit",
            "section_profile": "**Profil demandeur**",
            "section_credit": "**Crédit demandé**",
            "section_situation": "**Situation**",
            "label_age": "Âge",
            "label_income": "Revenu annuel (€)",
            "label_seniority": "Ancienneté emploi (années)",
            "label_owner": "Propriétaire",
            "label_amount": "Montant (€)",
            "label_duration": "Durée (mois)",
            "label_current_credits": "Crédits en cours",
            "label_debt_ratio": "Taux d'endettement (%)",
            "label_family_status": "Situation familiale",
            "label_payment_history": "Historique de paiement",
            "btn_evaluate": "💳 Évaluer la demande",
            # Options display (for format_func)
            "opt_yes": "Oui",
            "opt_no": "Non",
            "opt_single": "Célibataire",
            "opt_married": "Marié(e)",
            "opt_divorced": "Divorcé(e)",
            "opt_widowed": "Veuf/ve",
            "opt_excellent": "Excellent",
            "opt_good": "Bon",
            "opt_average": "Moyen",
            "opt_poor": "Mauvais",
            # Decision
            "decision_title": "Décision du modèle",
            "decision_accepted": "ACCEPTÉ",
            "decision_refused": "REFUSÉ",
            "decision_accepted_short": "Accepté",
            "decision_refused_short": "Refusé",
            "avg_risk_score": "Score de risque moyen : {value}",
            # SHAP
            "shap_title": "Explication SHAP — Facteurs de décision",
            "shap_chart_title": "Impact des variables sur la décision (SHAP values)",
            "shap_axis": "Impact sur le score de risque",
            "shap_annotation": "← Réduit le risque | Augmente le risque →",
            "shap_info": (
                "Les barres <span style='color:#51CF66'>vertes</span> réduisent le risque de défaut. "
                "Les barres <span style='color:#FF6B6B'>rouges</span> l'augmentent. "
                "Plus la barre est longue, plus le facteur pèse dans la décision."
            ),
            # SHAP feature names
            "shap_debt_ratio": "Taux d'endettement",
            "shap_seniority": "Ancienneté emploi",
            "shap_payment_hist": "Historique paiement",
            "shap_credit_ratio": "Ratio crédit/revenu",
            "shap_current_credits": "Crédits en cours",
            "shap_owner": "Propriétaire",
            "shap_age": "Âge",
            # Model comparison
            "models_title": "Performance des 4 modèles de scoring",
            "best_model": "Meilleur modèle",
            "chart_comparison": "Comparaison des métriques par modèle",
            "col_model": "Modèle",
            "col_metric": "Métrique",
            "col_score": "Score",
            "comparison_table_title": "Tableau comparatif",
            "roc_title": "Courbes ROC",
            "roc_axis_fpr": "Taux de faux positifs (FPR)",
            "roc_axis_tpr": "Taux de vrais positifs (TPR)",
            "roc_random": "Aléatoire",
            # Bias analysis
            "bias_title": "Analyse de biais — Fairlearn",
            "bias_info": (
                "L'analyse de biais vérifie que le modèle traite équitablement "
                "les différents groupes démographiques. Un modèle juste devrait avoir "
                "des <strong>taux d'acceptation similaires</strong> entre les groupes, "
                "à niveau de risque comparable."
            ),
            "bias_sex_title": "### Biais par sexe",
            "chart_acceptance_sex": "Taux d'acceptation par sexe",
            "axis_sex": "Sexe",
            "axis_acceptance": "Taux d'acceptation",
            "metric_di_ratio": "Disparate Impact Ratio",
            "fairness_status_label": "Statut d'équité",
            "fairness_fair": "Équitable",
            "fairness_bias": "Biais détecté",
            "fairness_threshold": "Seuil légal : 0.8 (80% rule)",
            "bias_age_title": "### Biais par tranche d'âge",
            "chart_acceptance_age": "Taux d'acceptation par tranche d'âge",
            "axis_age_group": "Tranche d'âge",
            "chart_risk_age": "Distribution du score de risque par tranche d'âge",
            "axis_risk_score": "Score de risque",
            "fairness_summary_title": "Résumé des métriques d'équité",
            "fairness_col_metric": "Métrique",
            "fairness_col_value": "Valeur",
            "fairness_col_threshold": "Seuil",
            "fairness_col_status": "Statut",
            "fairness_di_sex": "Disparate Impact (sexe)",
            "fairness_di_age": "Disparate Impact (âge)",
            "fairness_eod": "Equal Opportunity Diff.",
            "fairness_ppd": "Predictive Parity Diff.",
        },
        "en": {
            "header_title": "Credit Scoring",
            "header_desc": "Credit application simulation with SHAP explainability and bias analysis",
            "tab_simulator": "🔮 Credit Simulator",
            "tab_models": "⚖️ Model Comparison",
            "tab_bias": "🔍 Bias Analysis",
            "simulator_title": "Simulate a credit application",
            "section_profile": "**Applicant profile**",
            "section_credit": "**Credit requested**",
            "section_situation": "**Situation**",
            "label_age": "Age",
            "label_income": "Annual income (€)",
            "label_seniority": "Employment seniority (years)",
            "label_owner": "Homeowner",
            "label_amount": "Amount (€)",
            "label_duration": "Duration (months)",
            "label_current_credits": "Current credits",
            "label_debt_ratio": "Debt-to-income ratio (%)",
            "label_family_status": "Family status",
            "label_payment_history": "Payment history",
            "btn_evaluate": "💳 Evaluate application",
            "opt_yes": "Yes",
            "opt_no": "No",
            "opt_single": "Single",
            "opt_married": "Married",
            "opt_divorced": "Divorced",
            "opt_widowed": "Widowed",
            "opt_excellent": "Excellent",
            "opt_good": "Good",
            "opt_average": "Average",
            "opt_poor": "Poor",
            "decision_title": "Model decision",
            "decision_accepted": "ACCEPTED",
            "decision_refused": "REFUSED",
            "decision_accepted_short": "Accepted",
            "decision_refused_short": "Refused",
            "avg_risk_score": "Average risk score: {value}",
            "shap_title": "SHAP Explanation — Decision Factors",
            "shap_chart_title": "Variable impact on the decision (SHAP values)",
            "shap_axis": "Impact on risk score",
            "shap_annotation": "← Reduces risk | Increases risk →",
            "shap_info": (
                "<span style='color:#51CF66'>Green</span> bars reduce the default risk. "
                "<span style='color:#FF6B6B'>Red</span> bars increase it. "
                "The longer the bar, the more weight the factor has in the decision."
            ),
            "shap_debt_ratio": "Debt-to-income ratio",
            "shap_seniority": "Employment seniority",
            "shap_payment_hist": "Payment history",
            "shap_credit_ratio": "Credit/income ratio",
            "shap_current_credits": "Current credits",
            "shap_owner": "Homeowner",
            "shap_age": "Age",
            "models_title": "Performance of 4 scoring models",
            "best_model": "Best model",
            "chart_comparison": "Metric comparison by model",
            "col_model": "Model",
            "col_metric": "Metric",
            "col_score": "Score",
            "comparison_table_title": "Comparison table",
            "roc_title": "ROC Curves",
            "roc_axis_fpr": "False Positive Rate (FPR)",
            "roc_axis_tpr": "True Positive Rate (TPR)",
            "roc_random": "Random",
            "bias_title": "Bias Analysis — Fairlearn",
            "bias_info": (
                "Bias analysis checks that the model treats all demographic groups "
                "fairly. A fair model should have "
                "<strong>similar acceptance rates</strong> across groups, "
                "at comparable risk levels."
            ),
            "bias_sex_title": "### Bias by sex",
            "chart_acceptance_sex": "Acceptance rate by sex",
            "axis_sex": "Sex",
            "axis_acceptance": "Acceptance rate",
            "metric_di_ratio": "Disparate Impact Ratio",
            "fairness_status_label": "Fairness status",
            "fairness_fair": "Fair",
            "fairness_bias": "Bias detected",
            "fairness_threshold": "Legal threshold: 0.8 (80% rule)",
            "bias_age_title": "### Bias by age group",
            "chart_acceptance_age": "Acceptance rate by age group",
            "axis_age_group": "Age group",
            "chart_risk_age": "Risk score distribution by age group",
            "axis_risk_score": "Risk score",
            "fairness_summary_title": "Fairness metrics summary",
            "fairness_col_metric": "Metric",
            "fairness_col_value": "Value",
            "fairness_col_threshold": "Threshold",
            "fairness_col_status": "Status",
            "fairness_di_sex": "Disparate Impact (sex)",
            "fairness_di_age": "Disparate Impact (age)",
            "fairness_eod": "Equal Opportunity Diff.",
            "fairness_ppd": "Predictive Parity Diff.",
        },
    },
    # =========================================================================
    # SENTIMENT — Sentiment NLP (modules/sentiment.py)
    # =========================================================================
    "sentiment": {
        "fr": {
            "header_title": "Analyse de sentiment — NLP",
            "header_desc": "Comparez un baseline TF-IDF et CamemBERT sur des critiques de films en français",
            "examples_title": "Exemples à tester",
            "example_enthusiastic": "🌟 Enthousiaste",
            "example_nuanced": "🤔 Nuancé",
            "example_ironic": "😏 Ironique",
            "example_negative": "👎 Négatif",
            "example_mixed": "😐 Mitigé",
            "french_only_notice": (
                "🇫🇷 **Ce modèle fonctionne uniquement avec du texte en français.** "
                "Il a été entraîné sur un corpus de critiques de films francophones (Allociné)."
            ),
            "input_label": "Votre critique de film :",
            "btn_analyze": "🔍 Analyser le sentiment",
            "warning_empty": "Veuillez saisir une critique.",
            "positive": "Positif",
            "negative": "Négatif",
            "confidence": "Confiance",
            "top_words_title": "**Mots les plus influents :**",
            "axis_tfidf_weight": "Poids TF-IDF",
            "camembert_info": (
                "🤖 <strong>CamemBERT</strong> est un modèle de langage pré-entraîné "
                "sur un large corpus de textes français, puis fine-tuné sur notre dataset "
                "de critiques de films Allociné. Il capte mieux le contexte, l'ironie "
                "et les nuances que le baseline TF-IDF."
            ),
            "agree_msg": "✅ Les deux modèles sont d'accord : **{label}**",
            "disagree_msg": (
                "⚠️ Désaccord : TF-IDF dit **{label1}** "
                "({conf1}), CamemBERT dit **{label2}** "
                "({conf2})"
            ),
            "why_title": "Pourquoi deux modèles ?",
            "tfidf_title": "**TF-IDF + ML classique**",
            "tfidf_points": (
                "- Rapide et léger\n"
                "- Interprétable (mots influents)\n"
                "- Baseline de référence\n"
                "- Faiblesses : ironie, contexte"
            ),
            "camembert_title": "**CamemBERT (Transformer)**",
            "camembert_points": (
                "- Comprend le contexte\n"
                "- Capte l'ironie et les nuances\n"
                "- State-of-the-art NLP français\n"
                "- Plus lourd en ressources"
            ),
            "complementarity_title": "**Complémentarité**",
            "complementarity_points": (
                "- Le baseline sert de référence\n"
                "- CamemBERT améliore les cas difficiles\n"
                "- Les désaccords révèlent les ambiguïtés\n"
                "- Portfolio : maîtrise ML + Deep Learning"
            ),
        },
        "en": {
            "header_title": "Sentiment Analysis — NLP",
            "header_desc": "Compare a TF-IDF baseline and CamemBERT on French movie reviews",
            "examples_title": "Examples to test",
            "example_enthusiastic": "🌟 Enthusiastic",
            "example_nuanced": "🤔 Nuanced",
            "example_ironic": "😏 Ironic",
            "example_negative": "👎 Negative",
            "example_mixed": "😐 Mixed",
            "french_only_notice": (
                "🇫🇷 **This model only works with French text.** "
                "It was trained on a French-language movie review corpus (Allociné)."
            ),
            "input_label": "Your movie review:",
            "btn_analyze": "🔍 Analyze sentiment",
            "warning_empty": "Please enter a review.",
            "positive": "Positive",
            "negative": "Negative",
            "confidence": "Confidence",
            "top_words_title": "**Most influential words:**",
            "axis_tfidf_weight": "TF-IDF Weight",
            "camembert_info": (
                "🤖 <strong>CamemBERT</strong> is a language model pre-trained "
                "on a large corpus of French texts, then fine-tuned on our "
                "Allociné movie reviews dataset. It captures context, irony "
                "and nuances better than the TF-IDF baseline."
            ),
            "agree_msg": "✅ Both models agree: **{label}**",
            "disagree_msg": (
                "⚠️ Disagreement: TF-IDF says **{label1}** "
                "({conf1}), CamemBERT says **{label2}** "
                "({conf2})"
            ),
            "why_title": "Why two models?",
            "tfidf_title": "**TF-IDF + Classical ML**",
            "tfidf_points": (
                "- Fast and lightweight\n"
                "- Interpretable (influential words)\n"
                "- Reference baseline\n"
                "- Weaknesses: irony, context"
            ),
            "camembert_title": "**CamemBERT (Transformer)**",
            "camembert_points": (
                "- Understands context\n"
                "- Captures irony and nuances\n"
                "- French NLP state-of-the-art\n"
                "- More resource-intensive"
            ),
            "complementarity_title": "**Complementarity**",
            "complementarity_points": (
                "- Baseline serves as reference\n"
                "- CamemBERT improves on hard cases\n"
                "- Disagreements reveal ambiguities\n"
                "- Portfolio: ML + Deep Learning mastery"
            ),
        },
    },
}


def get_text(module, lang="fr"):
    """
    Get all translations for a module in the specified language.

    Parameters
    ----------
    module : str
        Module identifier (e.g. "app", "immobilier", "credit").
    lang : str
        Language code ("fr" or "en"). Defaults to "fr".

    Returns
    -------
    dict
        Flat dictionary {key: translated_string}.
    """
    mod = TRANSLATIONS.get(module, {})
    return mod.get(lang, mod.get("fr", {}))
