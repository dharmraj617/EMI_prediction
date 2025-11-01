import streamlit as st
import pandas as pd
import numpy as np
import pickle


# --- Helper Function (Copied from Script 3) ---
# We MUST apply the same feature engineering to the user's input
def create_financial_features(df):
    """
    Creates derived financial ratios based on the provided schema.
    This is the CORRECTED version.
    """
    expense_cols = [
        'monthly_rent', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses'
    ]
    df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)
    df['total_savings'] = df['bank_balance'] + df['emergency_fund']

    # Replace 0 with NaN for safe division
    df['monthly_salary_safe'] = df['monthly_salary'].replace(0, np.nan)

    df['disposable_income'] = df['monthly_salary'] - df['total_monthly_expenses'] - df['current_emi_amount']
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary_safe']
    df['expense_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary_safe']
    df['savings_to_salary_ratio'] = df['total_savings'] / df['monthly_salary_safe']
    df['loan_to_income_ratio'] = df['requested_amount'] / df['monthly_salary_safe']

    bins = [299, 579, 669, 739, 799, 850]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    df['credit_score_bin'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- START FIX ---
    # Be specific about which columns to fill.

    # 1. Fill NaNs in new numerical columns with 0
    num_cols_to_fill = [
        'disposable_income', 'debt_to_income_ratio', 'expense_to_income_ratio',
        'savings_to_salary_ratio', 'loan_to_income_ratio', 'monthly_salary_safe'
    ]
    for col in num_cols_to_fill:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    # 2. Fill NaNs in the categorical bin column with a valid category
    # (This handles edge cases, e.g., if credit score was NaN or outside bins)
    if 'credit_score_bin' in df.columns:
        # Add 'Unknown' as a valid category if it's not already there
        if 'Unknown' not in df['credit_score_bin'].cat.categories:
            df['credit_score_bin'] = df['credit_score_bin'].cat.add_categories('Unknown')

        # Fill any NaNs with our new 'Unknown' category
        df['credit_score_bin'].fillna('Unknown', inplace=True)
    # --- END FIX ---

    # This global line is now replaced by the specific logic above
    # df.fillna(0, inplace=True) # <-- OLD PROBLEMATIC LINE

    return df


# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        preprocessor = pickle.load(open('model_preprocessor.pkl', 'rb'))
        model_class = pickle.load(open('best_classification_model.pkl', 'rb'))
        model_reg = pickle.load(open('best_regression_model.pkl', 'rb'))
        label_encoder = pickle.load(open('classification_label_encoder.pkl', 'rb'))
        return preprocessor, model_class, model_reg, label_encoder
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run the training scripts first.")
        return None, None, None, None


preprocessor, model_class, model_reg, label_encoder = load_artifacts()

# --- Page Setup ---
st.set_page_config(page_title="New Prediction", page_icon="🔮", layout="wide")
st.title("🔮 New Eligibility Prediction")
st.write("Enter the applicant's details below to get a prediction.")

if preprocessor:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.header("Applicant Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 21, 60, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        with c2:
            education = st.selectbox("Education", ['High School', 'Graduate', 'Post Graduate', 'Professional'])
            employment_type = st.selectbox("Employment Type", ['Private', 'Government', 'Self-employed'])
            years_of_employment = st.number_input("Years of Employment", 0, 40, 5)
        with c3:
            company_type = st.selectbox("Company Type", ["Startup", "MNC", "Small-Scale", "Public"])  # Example
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            family_size = st.number_input("Family Size", 1, 10, 2)
            dependents = st.number_input("Dependents", 0, 5, 1)

        st.header("Financial Details (Monthly INR)")
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_salary = st.number_input("Monthly Salary", 15000, 200000, 50000, 1000)
            monthly_rent = st.number_input("Monthly Rent", 0, 50000, 10000, 500)
            school_fees = st.number_input("School Fees", 0, 20000, 0, 500)
            college_fees = st.number_input("College Fees", 0, 30000, 0, 500)
        with c2:
            travel_expenses = st.number_input("Travel Expenses", 0, 10000, 2000, 100)
            groceries_utilities = st.number_input("Groceries/Utilities", 0, 20000, 5000, 100)
            other_monthly_expenses = st.number_input("Other Expenses", 0, 15000, 1000, 100)
            existing_loans = st.selectbox("Existing Loans?", ["Yes", "No"])
            current_emi_amount = st.number_input("Current EMI Amount", 0, 50000, 0, 500)
        with c3:
            credit_score = st.number_input("Credit Score", 300, 850, 750)
            bank_balance = st.number_input("Bank Balance", 0, 1000000, 50000, 1000)
            emergency_fund = st.number_input("Emergency Fund", 0, 1000000, 20000, 1000)

        st.header("Loan Request Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            emi_scenario = st.selectbox("EMI Scenario", ["Scenario_1", "Scenario_2", "Scenario_3", "Scenario_4",
                                                         "Scenario_5"])  # Example
        with c2:
            requested_amount = st.number_input("Requested Loan Amount", 10000, 5000000, 100000, 1000)
        with c3:
            requested_tenure = st.number_input("Requested Tenure (Months)", 6, 60, 24)

        submit_button = st.form_submit_button(label='Predict Eligibility')

    # --- Prediction Logic ---
    if submit_button:
        # 1. Create a DataFrame from the inputs
        input_data = pd.DataFrame([
            {
                'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
                'monthly_salary': monthly_salary, 'employment_type': employment_type,
                'years_of_employment': years_of_employment, 'company_type': company_type,
                'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
                'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
                'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
                'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
                'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
                'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
                'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
                'requested_tenure': requested_tenure
            }
        ])

        # 2. Apply Feature Engineering
        input_featured = create_financial_features(input_data)

        # 3. Preprocess the data (using the loaded preprocessor)
        # Ensure columns match the ones preprocessor was trained on
        preprocessor_cols = preprocessor.feature_names_in_
        input_processed = preprocessor.transform(input_featured[preprocessor_cols])

        # 4. Make Predictions
        # Classification
        pred_class_idx = model_class.predict(input_processed)[0]
        pred_class_label = label_encoder.inverse_transform([pred_class_idx])[0]
        pred_class_proba = model_class.predict_proba(input_processed)[0]

        # Regression
        pred_reg_amount = model_reg.predict(input_processed)[0]

        # 5. Display Results
        st.header("Prediction Results")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Loan Eligibility")
            if pred_class_label == "Eligible":
                st.success(f"**Status: {pred_class_label}**")
                st.write("This applicant is a low risk and has good EMI affordability.")
            elif pred_class_label == "High_Risk":
                st.warning(f"**Status: {pred_class_label}**")
                st.write("This applicant is a marginal case. Approve with caution or at a higher interest rate.")
            else:
                st.error(f"**Status: {pred_class_label}**")
                st.write("This applicant is a high risk and the loan is not recommended.")

        with c2:
            st.subheader("Affordable EMI")
            st.metric("Max Recommended Monthly EMI", f"₹ {pred_reg_amount:,.2f}")
            st.write("This is the maximum EMI the applicant can safely afford based on their finances.")

        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame(
            [pred_class_proba],
            columns=label_encoder.classes_
        )
        st.dataframe(proba_df.style.format("{:.2%}"))