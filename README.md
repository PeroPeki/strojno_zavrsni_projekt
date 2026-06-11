# House Price Prediction — Završni projekt

Academic ML/DL course project. Predicts real estate sale prices using three progressively complex models and explains predictions with SHAP.

## Dataset

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Place `train.csv` and `test.csv` in `house_price_prediction/data/`.

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow shap jupyter
cd house_price_prediction
jupyter notebook
```

## Structure

```
house_price_prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_EDA_preprocessing.ipynb
│   └── 02_models_evaluation.ipynb
├── output/
│   ├── figures/
│   └── results/
│       └── model_results.csv
├── seminar/
│   └── seminar_outline.md
└── CLAUDE.md
```

## Models

1. **Linear Regression + Ridge** — baseline
2. **XGBoost** — gradient boosting
3. **MLP Neural Network** — Keras/TensorFlow

## Explainability

SHAP TreeExplainer applied to the XGBoost model.

## Evaluation

RMSE, MAE, R² on log-transformed SalePrice (log1p/expm1).
