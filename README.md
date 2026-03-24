# Interpretable-XGBoost-SHAP-Model-for-UHPC-Compressive-Strength
A comprehensive machine learning framework for predicting UHPC compressive strength incorporating multi-model evaluation, cross-validation, and SHAP-based interpretability.

Overview

This project presents a robust, end-to-end machine learning pipeline for predicting the compressive strength of Ultra-High Performance Concrete (UHPC).

It integrates:

Multiple ML models
Advanced visualization techniques
Explainable AI (XAI)
Statistical validation

Designed for both research (Q1-level) and practical engineering applications.

Key Highlights
-Multi-model comparison (RF, SVR, ANN, XGBoost)
-Publication-quality figures (600 DPI)
-SHAP-based explainability (global + local)
-Cross-validation for robustness
-Residual & error distribution analysis
-Engineering-focused feature interpretation

Models Used

Model	Description
-Random Forest	Ensemble learning for robust predictions
-XGBoost	Gradient boosting with high accuracy
-ANN	Neural network for nonlinear relationships
-SVR	Kernel-based regression

Features (Input Variables)
Cement (C)
Sand (S)
Silica Fume (SF)
Limestone Powder (LP)
Quartz Powder (QP)
Fly Ash (FA)
Nano Silica (NS)
Aggregate (A)
Water (W)
Fiber (Fi)
Superplasticizer (SP)
Temperature (T)
Age (days)

Target: Compressive Strength (CS)

Workflow
Data Loading & Preprocessing
Feature Scaling
Train-Test Split
Model Training
Performance Evaluation
Visualization & Error Analysis
SHAP Explainability
Cross-Validation

Outputs
Generated Figures
Age-based regime classification
Correlation heatmap
Actual vs Predicted plots
Residual scatter plots
Error distributions
SHAP plots (global + local)

Generated Table
Model performance comparison (R², RMSE, MAE, error stats)
Explainable AI (SHAP)
Identifies most influential mix parameters
Explains model decisions transparently
Supports engineering interpretability

Installation
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
