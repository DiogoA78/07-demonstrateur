# Demonstrateur Data Science

Application Streamlit multi-modules regroupant les demos interactives du portfolio Data Science de Diogo Almeida.

## Modules

| Module | Projet | Description |
|--------|--------|-------------|
| Immobilier IDF | P3 | Carte interactive des prix + predicteur ML + stats marche |
| Anomalies capteurs | P4 | Visualisation capteurs + comparaison 4 methodes de detection |
| Scoring credit | P5 | Simulateur de credit + SHAP waterfall + analyse biais Fairlearn |
| Sentiment NLP | P6 | Baseline TF-IDF vs CamemBERT sur critiques de films francais |

## Installation

```bash
# Cloner le repo
git clone https://github.com/DiogoA78/07-demonstrateur.git
cd 07-demonstrateur

# Creer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou : venv\Scripts\activate  # Windows

# Installer les dependances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## Mode demonstration

L'application fonctionne en **mode demo** sans modeles entraines : les predictions sont simulees a partir d'heuristiques realistes. Pour activer les vrais modeles, placez les fichiers `.pkl` issus des notebooks dans le dossier `models/`.

### Fichiers modeles attendus

| Module | Fichiers | Source |
|--------|----------|--------|
| Immobilier | `immobilier_model.pkl` | Notebook P3 |
| Anomalies | (donnees integrees) | Notebook P4 |
| Credit | `credit_logistic.pkl`, `credit_random_forest.pkl`, `credit_xgboost.pkl`, `credit_lightgbm.pkl` | Notebook P5 |
| Sentiment | `tfidf_vectorizer.pkl`, `baseline_model.pkl`, `camembert_sentiment/` | Notebook P6 |

## Stack technique

- **Frontend** : Streamlit
- **Visualisation** : Plotly
- **ML** : scikit-learn, XGBoost, LightGBM, Transformers (CamemBERT)
- **Equite** : Fairlearn (analyse de biais)

## Deploiement

> [🔗 Voir l'app sur Streamlit Cloud](https://diogoa78-07-demonstrateur-hsrfdt8xqoiobofkvqmaqd.streamlit.app/)

## Auteur

**Diogo Almeida** - Data Scientist
- [GitHub](https://github.com/DiogoA78)
- [LinkedIn](https://linkedin.com/in/diogo-almeida0)
