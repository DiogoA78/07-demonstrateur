🇫🇷 [Version française](README_FR.md)

# Data Science Demonstrator

Multi-module Streamlit application bringing together interactive demos from Diogo Almeida's Data Science portfolio.

## Modules

| Module | Project | Description |
|--------|---------|-------------|
| Real Estate IDF | P3 | Interactive price map + ML predictor + market stats |
| Sensor Anomalies | P4 | Sensor visualization + comparison of 4 detection methods |
| Credit Scoring | P5 | Credit simulator + SHAP waterfall + Fairlearn bias analysis |
| Sentiment NLP | P6 | TF-IDF baseline vs CamemBERT on French movie reviews |

## Installation

```bash
# Clone the repo
git clone https://github.com/DiogoA78/07-demonstrateur.git
cd 07-demonstrateur

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

## Demo Mode

The application runs in **demo mode** without trained models: predictions are simulated using realistic heuristics. To activate the real models, place the `.pkl` files generated from the notebooks into the `models/` folder.

### Expected Model Files

| Module | Files | Source |
|--------|-------|--------|
| Real Estate | `immobilier_model.pkl` | Notebook P3 |
| Anomalies | (data built-in) | Notebook P4 |
| Credit | `credit_logistic.pkl`, `credit_random_forest.pkl`, `credit_xgboost.pkl`, `credit_lightgbm.pkl` | Notebook P5 |
| Sentiment | `tfidf_vectorizer.pkl`, `baseline_model.pkl`, `camembert_sentiment/` | Notebook P6 |

## Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **ML**: scikit-learn, XGBoost, LightGBM, Transformers (CamemBERT)
- **Fairness**: Fairlearn (bias analysis)

## Deployment

> [🔗 View the app on Streamlit Cloud](https://diogoa78-07-demonstrateur-hsrfdt8xqoiobofkvqmaqd.streamlit.app/)

## Author

**Diogo Almeida** — Data Scientist
- [GitHub](https://github.com/DiogoA78)
- [LinkedIn](https://linkedin.com/in/diogo-almeida0)
