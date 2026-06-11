# House Price Prediction — CLAUDE.md

## Project

Academic ML/DL course project. Three models, SHAP explainability, Croatian academic submission.

## Stack

Python 3.10+, numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, tensorflow/keras, shap, jupyter

## Coding Conventions

- Notebook markdown cells: **Croatian (HR)**
- Code comments: English
- All plots: `plt.savefig('../output/figures/<name>.png', bbox_inches='tight')` before `plt.show()`
- `random_state=42` everywhere
- No `%matplotlib inline`
- Target: always `np.log1p()` before training, `np.expm1()` after predicting

## evaluate_model() — exact signature, use everywhere

```python
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
```

## Preprocessing Order

1. Concat train (drop SalePrice) + test
2. Fill missing: numeric → median, categorical → mode
3. LabelEncoder on all object columns
4. Add 4 engineered features (TotalSF, TotalBath, HouseAge, Remodeled)
5. Split back: X = all_data[:n_train], X_test_final = all_data[n_train:]
6. y = np.log1p(train['SalePrice'])
7. train_test_split(X, y, test_size=0.2, random_state=42)

## Required Output Files

- output/figures/saleprice_dist.png
- output/figures/missing_values.png
- output/figures/correlation_heatmap.png
- output/figures/xgb_feature_importance.png
- output/figures/mlp_learning_curve.png
- output/figures/shap_summary_bar.png
- output/figures/shap_beeswarm.png
- output/figures/model_comparison.png
- output/results/model_results.csv
