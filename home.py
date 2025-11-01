import streamlit as st

st.set_page_config(
    page_title="EMI Eligibility Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 EMI Eligibility Prediction & Analysis")

st.markdown("""
Welcome to the EMI Eligibility analysis tool. This platform uses machine learning to
predict loan eligibility and maximum affordable EMI.

### 📋 Project Overview
This application is the final product of a complete MLOps pipeline:
* **Data Preprocessing:** Cleaned and validated 400,000 financial records.
* **Feature Engineering:** Created key ratios like Debt-to-Income and Disposable Income.
* **Model Training:** Developed separate models for:
    1.  **Classification:** (Eligible, High_Risk, Not_Eligible)
    2.  **Regression:** (Max Affordable EMI)
* **Experiment Tracking:** Used MLflow to log and compare all model experiments.
* **Deployment:** This Streamlit app serves the best models for real-time predictions.

### 🚀 How to Use This App
* **New Prediction:** Navigate to this page to fill out a form with financial details and get an instant eligibility prediction.
* **Data Explorer:** Interactively explore the raw dataset used for training.
* **MLflow Dashboard:** View the performance metrics of all trained models from the MLflow experiment runs.
""")