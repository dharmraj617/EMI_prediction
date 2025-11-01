# 💰 EMI Eligibility Prediction - An End-to-End MLOps Project

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-#0194E2?style=for-the-badge&logo=mlflow)](https://mlflow.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F89939?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-316CCC?style=for-the-badge&logo=xgboost)](https://xgboost.ai/)

This repository contains the complete code for an end-to-end machine learning project designed to predict loan EMI eligibility and the maximum affordable EMI for an applicant. The project uses a dataset of 400,000 financial records and follows a full MLOps workflow, from data ingestion to a live, multi-page web application.



---

## 🚀 Features

* **Multi-Page Streamlit App:** A user-friendly web interface for predictions and data exploration.
* **Dual-Model System:**
    1.  **Classification Model:** Predicts EMI eligibility (`Eligible`, `High_Risk`, `Not_Eligible`).
    2.  **Regression Model:** Predicts the maximum affordable monthly EMI (`max_monthly_emi`).
* **MLflow Experiment Tracking:** All model training runs, parameters, metrics, and artifacts are logged and viewable.
* **Interactive Data Explorer:** A dashboard to visualize the raw 400k-record dataset.
* **Deployment:** Deployed and hosted on Streamlit Cloud.

---

## 🛠️ Tech Stack

* **Data Processing:** `pandas`, `numpy`
* **ML Modeling:** `scikit-learn`, `xgboost`
* **MLOps & Experiment Tracking:** `mlflow`
* **Web Application:** `streamlit`
* **Version Control:** `git`, `GitHub`
* **Deployment:** `Streamlit Cloud`

---

## 🤖 MLOps Pipeline

This project is structured as a sequential pipeline. The output of one script becomes the input for the next, ensuring reproducibility and modularity.



1.  **`1_preprocess.py`**: Loads the raw 400,000-record CSV, cleans it (handles missing values, duplicates, and type errors), and splits it into `train`, `validation`, and `test` sets.
2.  **`2_eda.py`**: (Optional) Performs exploratory data analysis on the training set to find correlations and patterns.
3.  **`3_feature_engineering.py`**:
    * Creates new features (e.g., `debt_to_income_ratio`, `disposable_income`, `credit_score_bin`).
    * Creates a `ColumnTransformer` (preprocessor) to scale numeric and encode categorical features.
    * Saves the processed datasets and the vital `model_preprocessor.pkl`.
4.  **`4_run_classification.py`**:
    * Loads the processed data.
    * Trains three classification models (Logistic Regression, Random Forest, XGBoost).
    * Logs all parameters, metrics (Accuracy, F1, ROC-AUC), and the models to **MLflow**.
    * Saves the `classification_label_encoder.pkl`.
5.  **`5_run_regression.py`**:
    * Loads the processed data.
    * Trains three regression models (Linear Regression, Random Forest, XGBoost).
    * Logs all parameters, metrics (R2, RMSE, MAE), and the models to **MLflow**.
    * Automatically queries the MLflow server to find the **best** classification and regression models and saves them as `best_classification_model.pkl` and `best_regression_model.pkl`.
6.  **Streamlit App (`1_🏠_Home.py`, `pages/*.py`)**:
    * Loads all `.pkl` artifacts.
    * Presents a form to the user.
    * Applies the *exact same* feature engineering and preprocessing to the user's input.
    * Feeds the processed data to the best-saved models to deliver a real-time prediction.

---

## 📁 Project Structure
