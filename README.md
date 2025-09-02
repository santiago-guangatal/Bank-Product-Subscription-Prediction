# Bank Marketing Prediction (LightGBM + Logistic Regression)

This project builds a machine learning pipeline to predict whether a customer will subscribe to a term deposit based on the **Bank Marketing dataset**.  

It combines **feature engineering (FE)**, **LightGBM leaf encoding**, and a **logistic regression classifier**, with hyperparameter optimization using **Optuna**.

---

## Project Workflow

1. **Data loading & preprocessing**
   - Removes IDs
   - Splits train/test
   - Handles missing values & "unknown" placeholders
   - Feature engineering via custom `FE` transformer

2. **Modeling**
   - Base model: `LGBMClassifier`
   - Leaf encoding with `LeafOneHotEncoder`
   - Final classifier: `LogisticRegression`

3. **Hyperparameter Optimization**
   - Search space defined for LightGBM + Logistic Regression
   - Cross-validation with `StratifiedKFold (n=3)`
   - Optimization with **Optuna (TPESampler)**

4. **Evaluation**
   - ROC AUC
   - Average Precision (AP)
   - Precision–Recall & ROC Curves
   - Classification Report & Confusion Matrix

5. **Persistence**
   - Best model is saved with `cloudpickle` → `model.pkl`

---
