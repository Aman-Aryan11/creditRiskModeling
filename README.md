# Credit Risk Modeling

An end-to-end loan default risk project covering data preparation, exploratory analysis, feature engineering, feature selection, imbalanced classification, model tuning, and credit risk evaluation. A Streamlit app turns the saved model into an interactive score and rating.

## Project contents

- `credit_risk_model.ipynb` — full modeling workflow and evaluation.
- `dataset/` — customer, loan, and bureau input data used by the notebook.
- `artifacts/model_data.joblib` — model bundle exported by the notebook.
- `app/` — Streamlit interface and prediction helper, with the model bundle in `app/artifacts/`.
- `requirements.txt` — pinned Python dependencies.

## Setup

Python 3.9 is supported. From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the notebook

Open `credit_risk_model.ipynb` in Jupyter and run all cells from the repository root so the relative `dataset/` and `artifacts/` paths resolve. The notebook writes the fitted model bundle to `artifacts/model_data.joblib`.

## Run the app

From the repository root:

```bash
streamlit run app/main.py
```

The app loads its model bundle from `app/artifacts/model_data.joblib`.

## Modeling workflow

1. Join customer, loan, and bureau data, then split into stratified train and test sets.
2. Clean missing values and invalid records, inspect distributions, and engineer loan-to-income and delinquency features.
3. Select and encode features, compare logistic regression, random forest, and XGBoost, and address class imbalance with undersampling and SMOTE-Tomek.
4. Tune candidate models, then evaluate discrimination and rank ordering with classification metrics, ROC-AUC, decile tables, KS, and Gini.
5. Export the final model and preprocessing data for scoring.

## Notes

- The notebook uses a fixed random seed for its train/test split and sampling steps.
- Credit scores and ratings shown by the app are derived from estimated default probability; they are project outputs and should be validated before any lending decision.
