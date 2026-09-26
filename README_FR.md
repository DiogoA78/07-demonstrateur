🇬🇧 [English version](README.md)

# Démonstrateur Data Science

Application Streamlit multi-modules regroupant les démos interactives du portfolio Data Science de Diogo Almeida.

## Modules

| Module | Projet | Description |
|--------|--------|-------------|
| Immobilier IDF | P3 | Carte interactive des prix + prédicteur ML + stats marché |
| Anomalies capteurs | P4 | Visualisation capteurs + comparaison 4 méthodes de détection |
| Scoring crédit | P5 | Simulateur de crédit + SHAP waterfall + analyse biais Fairlearn |
| Sentiment NLP | P6 | Baseline TF-IDF vs CamemBERT sur critiques de films français |

## Installation

```bash
# Cloner le repo
git clone https://github.com/DiogoA78/07-demonstrateur.git
cd 07-demonstrateur

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou : venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## Mode démonstration

L'application fonctionne en **mode démo** sans modèles entraînés : les prédictions sont simulées à partir d'heuristiques réalistes. Pour activer les vrais modèles, placez les fichiers `.pkl` issus des notebooks dans le dossier `models/`.

### Fichiers modèles attendus

| Module | Fichiers | Source |
|--------|----------|--------|
| Immobilier | `immobilier_model.pkl` | Notebook P3 |
| Anomalies | (données intégrées) | Notebook P4 |
| Crédit | `credit_logistic.pkl`, `credit_random_forest.pkl`, `credit_xgboost.pkl`, `credit_lightgbm.pkl` | Notebook P5 |
| Sentiment | `tfidf_vectorizer.pkl`, `baseline_model.pkl`, `camembert_sentiment/` | Notebook P6 |

## Stack technique

- **Frontend** : Streamlit
- **Visualisation** : Plotly
- **ML** : scikit-learn, XGBoost, LightGBM, Transformers (CamemBERT)
- **Équité** : Fairlearn (analyse de biais)

## Déploiement

> [🔗 Voir l'app sur Streamlit Cloud](https://diogoa78-07-demonstrateur-hsrfdt8xqoiobofkvqmaqd.streamlit.app/)

## Auteur

**Diogo Almeida** — Data Scientist
- [GitHub](https://github.com/DiogoA78)
- [LinkedIn](https://linkedin.com/in/diogo-almeida0)
