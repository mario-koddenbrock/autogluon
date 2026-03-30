"""Minimal smoke test for TabICL regression support."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from autogluon.tabular import TabularPredictor

np.random.seed(42)
X, y = make_regression(n_samples=200, n_features=20, noise=0.1, random_state=42)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y

train = df.iloc[:160]
test = df.iloc[160:]

print(f"Train: {train.shape}, Test: {test.shape}")

predictor = TabularPredictor(
    label="target",
    problem_type="regression",
).fit(
    train_data=train,
    hyperparameters={"TABICL": {}},
    time_limit=120,
    verbosity=2,
)

preds = predictor.predict(test.drop(columns=["target"]))
r2 = r2_score(test["target"].values, preds.values)
print(f"\nR² on test set: {r2:.4f}")
assert r2 > 0.5, f"R² too low: {r2:.4f} — model may not be working correctly"
print("OK")
