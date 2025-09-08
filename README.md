# Bank Marketing Prediction (LightGBM + Logistic Regression)

This project builds a machine learning pipeline to predict whether a customer will subscribe to a term deposit based on the **[Bank Marketing Dataset on Kaggle](https://www.kaggle.com/competitions/playground-series-s5e8/data)**.  

It combines **feature engineering (FE)** and a **LGBMClassifier** with hyperparameter optimization using **HalvingRandomSearchCV**.

---

## Project Workflow

1. **Data loading & preprocessing**
   - Splits train/test
   - Handles missing values & "unknown" placeholders
   - Feature engineering via custom `FE` transformer

2. **Model Benchmarking**
   - Compared Logistic Regression, SGDClassifier, RandomForest, and LightGBM
   - Cross-validation with stratified folds
   - Selected **LightGBM** based on PR-AUC and overall robustness

3. **Modeling**
   - Final model: `LGBMClassifier` wrapped in pipeline with `FE`

4. **Hyperparameter Optimization**
   - Optimization with **HalvingRandomSearchCV**
   - Search space over learning rate, tree complexity, regularization, and sampling

5. **Evaluation**
   - Metrics: PR-AUC, ROC-AUC, LogLoss, Brier
   - Threshold selection: Max-F1 and budget-based (top 10% scoring clients)
   - Curves: Precision–Recall, ROC, and Calibration

6. **Persistence & Deployment**
   - Best model saved with `joblib` → `model.joblib`
   - Hyperparameters saved as `params.json`

---
