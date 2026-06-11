# Seminarski rad — Predviđanje cijena nekretnina

**Kolegij:** Strojno i duboko učenje  
**Student:** [Ime i prezime]  
**Akademska godina:** 2025/2026

---

## 1. Uvod i definicija problema

- Motivacija: praktična primjena ML u procjeni vrijednosti nekretnina
- Cilj: predviđanje prodajne cijene kuće (`SalePrice`) na temelju 79 obilježja
- Metrika uspješnosti: RMSE na log-transformiranoj ciljnoj varijabli
- Opis dataseta: Kaggle — House Prices: Advanced Regression Techniques
  - 1460 primjeraka za treniranje, 79 obilježja + cijena
  - 1459 primjeraka za testiranje, bez ciljne varijable

---

## 2. Podaci i istraživačka analiza (EDA)

### 2.1 Opis dataseta
- Numeričke vs. kategoričke varijable
- Raspon i statistike ciljne varijable

### 2.2 Ciljna varijabla
- Distribucija sirove `SalePrice` — desna asimetrija
- Log-transformacija: `np.log1p(SalePrice)` → normalnija distribucija
- Vizualizacija: `saleprice_dist.png`

### 2.3 Nedostajuće vrijednosti
- Top 20 obilježja po udjelu nedostajućih vrijednosti
- Strategija popunjavanja: medijan (numeričke), mod (kategoričke)
- Vizualizacija: `missing_values.png`

### 2.4 Korelacijska analiza
- Top 10 obilježja s najvećom korelacijom s `SalePrice`
- Ključna obilježja: `OverallQual`, `GrLivArea`, `GarageCars`, ...
- Vizualizacija: `correlation_heatmap.png`

---

## 3. Predprocesiranje i inženjering obilježja

### 3.1 Pipeline
1. Konkatenacija train i test skupova
2. Popunjavanje nedostajućih vrijednosti
3. Label encoding kategoričkih obilježja
4. Inženjering novih obilježja
5. Razdvajanje nazad i train/val split

### 3.2 Nova obilježja
| Obilježje | Formula | Motivacija |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Ukupna kvadratura |
| `TotalBath` | `FullBath + 0.5 * HalfBath` | Težinska suma kupaonica |
| `HouseAge` | `YrSold - YearBuilt` | Starost kuće pri prodaji |
| `Remodeled` | `YearRemodAdd != YearBuilt` | Binarna — je li renovirana |

---

## 4. Modeli

### 4.1 Linearna regresija (baseline)
- Opis metode: minimizacija kvadratnih grešaka
- Regularizacija: Ridge (L2, α=10) sprečava preučenje
- Normalizacija ulaza: `StandardScaler`
- Prednosti: interpretabilnost, brzina treniranja
- Ograničenja: pretpostavka linearnosti

### 4.2 XGBoost
- Opis: ensemble metoda — gradient boosting stabala odlučivanja
- Parametri: n_estimators=500, lr=0.05, max_depth=4, subsample=0.8
- Prednosti: robustnost, automatska selekcija obilježja
- Vizualizacija: `xgb_feature_importance.png`

### 4.3 MLP Neuronska mreža
- Arhitektura: Input → 256 → Dropout(0.3) → 128 → Dropout(0.2) → 64 → 1
- Optimizer: Adam (lr=0.001), Loss: MSE
- Regularizacija: Dropout + EarlyStopping (patience=15)
- Vizualizacija krivulja učenja: `mlp_learning_curve.png`

---

## 5. Objašnjivost — SHAP analiza

- Metoda: SHAP TreeExplainer primijenjen na XGBoost model
- Globalna važnost obilježja: `shap_summary_bar.png`
- Distribucija SHAP vrijednosti (beeswarm): `shap_beeswarm.png`
- Ključni nalaz: [popuniti nakon eksperimenta]

---

## 6. Evaluacija i usporedba

### 6.1 Metrike

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Ridge (α=10) | — | — | — |
| XGBoost | — | — | — |
| MLP | — | — | — |

*(popuniti nakon pokretanja notebooka)*

### 6.2 Usporedni graf
- Vizualizacija: `model_comparison.png`

### 6.3 Zaključak
- Koji model postiže najbolje rezultate i zašto
- Kompromis između interpretabilnosti i performansi
- Prijedlozi za poboljšanje (hyperparameter tuning, feature selection, ensembling)

---

## 7. Zaključak

- Sažetak postignuća projekta
- Praktična primjenljivost modela
- Potencijalni pravci daljnjeg istraživanja

---

## Literatura

1. Kaggle. *House Prices: Advanced Regression Techniques.* https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16.*
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS.*
4. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* O'Reilly.
