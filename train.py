"""
train.py
--------
Generates synthetic salary data, trains a RandomForestRegressor,
and saves the trained model + scaler to disk using joblib.

Run once before starting the API:
    python train.py
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ──────────────────────────────────────────────
# 1. Reproducible synthetic dataset
# ──────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_SAMPLES = 1_000

years_experience = np.random.uniform(0, 30, N_SAMPLES)

# Salary formula:
#   base = 30,000
#   +2,500 per year of experience
#   +Gaussian noise (std ≈ 4,000)
salary = (
    30_000
    + 2_500 * years_experience
    + np.random.normal(0, 4_000, N_SAMPLES)
).clip(min=20_000)          # floor at $20k

X = years_experience.reshape(-1, 1)   # shape: (N, 1)
y = salary

# ──────────────────────────────────────────────
# 2. Train / test split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ──────────────────────────────────────────────
# 3. Build a scikit-learn Pipeline
#    StandardScaler → RandomForestRegressor
# ──────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 4. Evaluate
# ──────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"✅ Training complete")
print(f"   MAE : ${mae:,.2f}")
print(f"   R²  : {r2:.4f}")

# ──────────────────────────────────────────────
# 5. Persist model
# ──────────────────────────────────────────────
MODEL_DIR  = "app/model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"   Model saved → {MODEL_PATH}")
