# 🏠 House Price Prediction API (FastAPI + XGBoost + Docker + Render)

🔗 **Live Demo:** (https://prediction-for-house-price.onrender.com)  
A fully deployed web API where users can modify house features and get instant price predictions through my trained **XGBoost regression model**.  
Built end-to-end — from model training → API → Docker containerization → Render cloud deployment (CI-ready).

---

## 🚀 Overview

This project demonstrates the complete **Machine Learning Engineering workflow**:
- **Modeling:** Trained an XGBoost regression model with Optuna hyperparameter tuning.
- **Feature Engineering:** Applied winsorization, Box–Cox target transformation, and multiple engineered features.
- **Preprocessing:** Designed a full sklearn `Pipeline` handling categorical, numerical, and ordinal mappings.
- **Serving:** Exposed the model through a REST API built with **FastAPI**.
- **Containerization:** Environment isolated with **Docker**.
- **Deployment:** Automatically built and deployed on **Render**, connected to GitHub for auto-deploy.

---

## 🧩 Tech Stack

**Machine Learning:** `XGBoost`, `scikit-learn`, `Optuna`, `Feature-engine`  
**Serving:** `FastAPI`, `Uvicorn`  
**Containerization:** `Docker`  
**Deployment:** `Render (PaaS)`  
**Visualization:** `Seaborn`, `Matplotlib`  

---


---

## 🧠 Core Modeling Details

### 1. **Feature Engineering (FE)**
Implemented in `add_top_features()`:
- Created composite metrics such as `TotalSF`, `TotalBath`, and `TotalPorchSF`
- Derived `HouseAge` and `RemodAge` (renovation gap)
- Added `log1p_` transformations for key skewed numeric features  
  (`GrLivArea`, `LotArea`, `TotalBsmtSF`, etc.)

### 2. **Ordinal Mapping**
Used explicit ordinal encodings for ordered categorical features:
- `ExterQual`, `BsmtCond`, `KitchenQual`, `GarageQual`, etc.  
  → mapped via dictionaries like `{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}`  
- Implemented through a custom transformer `apply_ordinal_maps()` wrapped in `FunctionTransformer`.

### 3. **Outlier Winsorization**
- Applied **quantile winsorization** (`fold=0.005`) to suppress right-tail outliers for `GrLivArea` and `TotalBsmtSF`.
- Non-winsorized numerical features handled via separate imputation pipeline.

### 4. **Box–Cox Transformation**
- Target variable `SalePrice` transformed with `PowerTransformer(method='box-cox')` for variance stabilization.
- Used `TransformedTargetRegressor` wrapping the model for smooth inverse transformation at prediction time.

### 5. **Hyperparameter Optimization**
Used **Optuna + TPESampler + MedianPruner** to tune:
- `learning_rate`, `max_depth`, `n_estimators`, `min_child_weight`, `subsample`, etc.
- Objective: minimize RMSE via 5-fold CV.

---

## 🧱 FastAPI Serving Layer

The `main.py` file:
- Loads the serialized pipeline `house_price_xgb_pipe.pkl`
- Provides:
  - `GET /` → simple HTML front-end for demo  
  - `GET /docs` → interactive Swagger API  
  - `POST /predict` → returns `{"predicted_price": float}`
 
🐳 Docker & Deployment

Dockerfile (final CMD line):

CMD ["bash", "-c", "uvicorn house_price_api.app.main:app --host 0.0.0.0 --port ${PORT:-8003}"]


🧩 Engineering Highlights

Designed end-to-end reproducible ML pipeline (feature engineering → model → inference).

Encapsulated preprocessing logic (ordinal mapping, winsorization, Box–Cox) as reusable sklearn transformers.

Automated hyperparameter tuning with Optuna (parallel + pruning).

Built a FastAPI microservice for model inference, fully Dockerized.

Deployed to Render with live public endpoint and minimal HTML interface.

🚀 How to Use
Clone the repository:
git clone https://github.com/Eriq7/ML-House-Price-API-Render
cd ML-House-Price-API-Render



