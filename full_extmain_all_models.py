# ============================================================
# UHPC Compressive Strength Prediction
# All Models + All Graphs + XAI (FINAL)
# ============================================================

# ----------------------------
# 1. Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
# Columns:
# C S SF LP QP FA NS A W Fi SP T Age CS

data = pd.read_csv("UHPC_data.csv")

# ----------------------------
# 3. Correlation Heatmap
# ----------------------------
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of UHPC Parameters")
plt.show()

# ----------------------------
# 4. Define Features & Target
# ----------------------------
X = data.drop("CS", axis=1)
y = data["CS"]

# ----------------------------
# 5. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 6. Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 7. Define Models
# ----------------------------
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=15, random_state=42
    ),
    "SVR": SVR(kernel="rbf", C=100, gamma=0.1),
    "ANN": MLPRegressor(
        hidden_layer_sizes=(50, 50),
        max_iter=2000,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
}

# ----------------------------
# 8. Train Models & Predict
# ----------------------------
results = {}
performance = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    results[name] = pred

    performance.append([
        name,
        r2_score(y_test, pred),
        np.sqrt(mean_squared_error(y_test, pred)),
        mean_absolute_error(y_test, pred)
    ])

# ----------------------------
# 9. Performance Comparison Table
# ----------------------------
performance_df = pd.DataFrame(
    performance, columns=["Model", "R2", "RMSE", "MAE"]
)

print("\nMODEL PERFORMANCE COMPARISON")
print(performance_df)

# ----------------------------
# 10. GRAPHS FOR ALL MODELS
# ----------------------------

for name, pred in results.items():

    # ---- Actual vs Predicted ----
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, pred, alpha=0.7)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )
    plt.xlabel("Actual CS")
    plt.ylabel("Predicted CS")
    plt.title(f"Actual vs Predicted – {name}")
    plt.grid(True)
    plt.show()

    # ---- Residual Plot ----
    residuals = y_test - pred
    plt.figure(figsize=(6,5))
    plt.scatter(pred, residuals)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted CS")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot – {name}")
    plt.show()

    # ---- Error Distribution ----
    plt.figure(figsize=(6,5))
    sns.histplot(residuals, kde=True)
    plt.title(f"Error Distribution – {name}")
    plt.show()

# ----------------------------
# 11. SHAP Explainable AI (Best Tree Model: XGBoost)
# ----------------------------
best_model = models["XGBoost"]

explainer = shap.Explainer(best_model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Global Importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Detailed Impact
shap.summary_plot(shap_values, X_test)

# Local Explanation
shap.waterfall_plot(shap_values[0])

# ----------------------------
# 12. SHAP Importance Table
# ----------------------------
shap_mean = np.abs(shap_values.values).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_SHAP": shap_mean
}).sort_values(by="Mean_SHAP", ascending=False)

print("\nSHAP FEATURE IMPORTANCE (XGBoost)")
print(shap_df)

# ============================================================
# END OF FULL CODE
# ============================================================
