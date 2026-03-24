# ============================================================
# UHPC Compressive Strength Prediction
# Q1 Journal–Ready Full Pipeline
# ============================================================

# ----------------------------
# 1. Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import shap

# ----------------------------
# 2. Load Dataset
# ----------------------------
data = pd.read_csv("UHPC_data.csv")

# ----------------------------
# 3. AGE-BASED REGIME IDENTIFICATION FIGURE
# ----------------------------
early = data[data["Age"] <= 28]
late = data[data["Age"] > 28]

plt.figure(figsize=(7,5))
plt.scatter(early["Age"], early["CS"], alpha=0.7, label="Age ≤ 28 days")
plt.scatter(late["Age"], late["CS"], alpha=0.7, label="Age > 28 days")
plt.axvline(28, color="red", linestyle="--", linewidth=2)
plt.xlabel("Curing Age (days)")
plt.ylabel("Compressive Strength (MPa)")
plt.title("UHPC Strength Development Regime Classification")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig_1_Age_Regime.png", dpi=600)
plt.show()

# ----------------------------
# 4. CORRELATION HEATMAP
# ----------------------------
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of UHPC Parameters")
plt.tight_layout()
plt.savefig("Fig_2_Correlation_Heatmap.png", dpi=600)
plt.show()

# ----------------------------
# 5. FEATURES & TARGET
# ----------------------------
X = data.drop("CS", axis=1)
y = data["CS"]

# ----------------------------
# 6. TRAIN–TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 7. SCALING
# ----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ----------------------------
# 8. DEFINE MODELS
# ----------------------------
models = {
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
    "SVR": SVR(kernel="rbf", C=100, gamma=0.1),
    "ANN": MLPRegressor(hidden_layer_sizes=(50,50), max_iter=2000, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42
    )
}

# ----------------------------
# 9. TRAIN & EVALUATE ALL MODELS
# ----------------------------
results = {}
metrics = []

for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    results[name] = pred

    metrics.append([
        name,
        r2_score(y_test, pred),
        np.sqrt(mean_squared_error(y_test, pred)),
        mean_absolute_error(y_test, pred),
        np.mean(y_test - pred),
        np.std(y_test - pred)
    ])

perf_df = pd.DataFrame(
    metrics,
    columns=["Model", "R2", "RMSE", "MAE", "Mean_Error", "Std_Error"]
)

print(perf_df)
perf_df.to_csv("Table_Model_Performance.csv", index=False)

# ----------------------------
# 10. ALL MODEL GRAPHS
# ----------------------------
for name, pred in results.items():

    # Actual vs Predicted
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual CS")
    plt.ylabel("Predicted CS")
    plt.title(f"Actual vs Predicted – {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Fig_AVP_{name}.png", dpi=600)
    plt.show()

    # Residual distribution
    residuals = y_test - pred
    plt.figure(figsize=(6,5))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residual Distribution – {name}")
    plt.tight_layout()
    plt.savefig(f"Fig_Residual_{name}.png", dpi=600)
    plt.show()

# ----------------------------
# 11. CROSS-VALIDATION (XGBOOST)
# ----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
xgb = models["XGBoost"]
cv_r2 = cross_val_score(xgb, X_train_s, y_train, cv=cv, scoring="r2")

print("XGBoost CV R2 Mean:", cv_r2.mean())
print("XGBoost CV R2 Std :", cv_r2.std())

# ----------------------------
# 12. SHAP EXPLAINABILITY (XGBOOST)
# ----------------------------
explainer = shap.Explainer(xgb, X_train_s)
shap_values = explainer(X_test_s)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("Fig_SHAP_Global.png", dpi=600)
plt.show()

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("Fig_SHAP_Detailed.png", dpi=600)
plt.show()

shap.waterfall_plot(shap_values[0], show=False)
plt.savefig("Fig_SHAP_Local.png", dpi=600)
plt.show()

# ============================================================
# END OF Q1-READY PIPELINE
# ============================================================
