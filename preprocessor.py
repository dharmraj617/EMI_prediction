import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import sys
import os


def create_financial_features(df):
    """
    Creates derived financial ratios based on the provided schema.
    """
    print("--- Starting Feature Creation ---")

    cols_to_convert = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
        'family_size', 'dependents', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
        'requested_amount', 'requested_tenure'
    ]

    print(f"[Clean] Forcing {len(cols_to_convert)} columns to numeric...")
    for col in cols_to_convert:
        if col in df.columns:
            # errors='coerce' turns non-numeric text (like '$50,000' or 'N/A') into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"[Warning] Expected numeric column '{col}' not found. Skipping.")

    # Now, impute NaNs created by 'coerce' before calculations.
    # We'll fill financial columns with 0, but salary/credit score with median.

    # Fill with median (to avoid skewing ratios)
    if 'monthly_salary' in df.columns:
        df['monthly_salary'].fillna(df['monthly_salary'].median(), inplace=True)
    if 'credit_score' in df.columns:
        df['credit_score'].fillna(df['credit_score'].median(), inplace=True)

    # Fill with 0 (assuming 0 for missing expenses/balances is safest)
    expense_balance_cols = [
        'monthly_rent', 'school_fees', 'college_fees', 'travel_expenses',
        'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount',
        'bank_balance', 'emergency_fund'
    ]
    for col in expense_balance_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    df['monthly_salary'] = df['monthly_salary'].replace(0, np.nan)

    # 1. Total Monthly Expenses
    expense_cols = [
        'monthly_rent', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses'
    ]
    df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)

    # 2. Total Savings
    df['total_savings'] = df['bank_balance'] + df['emergency_fund']

    # 3. Disposable Income (Income after all known expenses and EMIs)
    df['disposable_income'] = df['monthly_salary'] - df['total_monthly_expenses'] - df['current_emi_amount']

    # 4. Debt-to-Income Ratio (DTI)
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']

    # 5. Expense-to-Income Ratio
    df['expense_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary']

    # 6. Savings-to-Salary Ratio
    df['savings_to_salary_ratio'] = df['total_savings'] / df['monthly_salary']

    # 7. Loan-to-Income Ratio (based on requested loan)
    df['loan_to_income_ratio'] = df['requested_amount'] / df['monthly_salary']

    # 8. Credit Score Bins (Example of binning)
    bins = [299, 579, 669, 739, 799, 850]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    df['credit_score_bin'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=True)

    # Replace infinite values (from division by zero if any slipped) with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute NaNs created by division by zero or from original data.
    # For ratios, a median of 0 or a specific value might be appropriate.
    # Here, we'll fill with 0 for simplicity, but median imputation is also good.
    ratio_cols = ['debt_to_income_ratio', 'expense_to_income_ratio',
                  'savings_to_salary_ratio', 'loan_to_income_ratio']
    df[ratio_cols] = df[ratio_cols].fillna(0)
    df['disposable_income'] = df['disposable_income'].fillna(df['disposable_income'].median())

    print(f"Created {len(expense_cols) + 7} new features.")

    return df


def define_preprocessor(df):
    """
    Defines the ColumnTransformer for preprocessing.
    """
    print("--- Defining Preprocessor ---")

    # 1. Define lists of columns based on data dictionary

    # Targets (to be excluded from features)
    targets = ['emi_eligibility', 'max_monthly_emi']

    # Original numerical features
    numeric_features_orig = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
        'family_size', 'dependents', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
        'requested_amount', 'requested_tenure'
    ]

    # New derived numerical features
    numeric_features_new = [
        'total_monthly_expenses', 'total_savings', 'disposable_income',
        'debt_to_income_ratio', 'expense_to_income_ratio',
        'savings_to_salary_ratio', 'loan_to_income_ratio'
    ]

    numeric_features = numeric_features_orig + numeric_features_new

    # Nominal categorical features (to be One-Hot Encoded)
    nominal_features = [
        'gender', 'marital_status', 'employment_type', 'company_type',
        'house_type', 'existing_loans', 'emi_scenario'
    ]

    # Ordinal categorical features (to be Ordinal Encoded)
    ordinal_features = ['education', 'credit_score_bin']

    # 2. Define the specific order for ordinal features
    education_order = ['High School', 'Graduate', 'Post Graduate', 'Professional']
    credit_bin_order = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

    # 3. Create the transformers

    # Numeric Transformer: Scale data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Nominal Transformer: One-Hot Encode
    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Ordinal Transformer: Ordinal Encode
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=[education_order, credit_bin_order],
                                   handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 4. Create the ColumnTransformer
    # This will apply the correct transformer to each column list

    # Filter lists to only include columns present in the DataFrame
    numeric_features = [col for col in numeric_features if col in df.columns]
    nominal_features = [col for col in nominal_features if col in df.columns]
    ordinal_features = [col for col in ordinal_features if col in df.columns]

    print(f"Found {len(numeric_features)} numerical features to scale.")
    print(f"Found {len(nominal_features)} nominal features to one-hot encode.")
    print(f"Found {len(ordinal_features)} ordinal features to ordinal encode.")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features)
        ],
        remainder='drop'  # Drop any columns not specified (like the original targets)
    )

    return preprocessor, numeric_features + nominal_features + ordinal_features, targets


def process_datasets(preprocessor, feature_list, target_list):
    """
    Loads train/val/test, applies feature engineering,
    fits preprocessor on train, and transforms all.
    Saves processed data.
    """

    datasets = {}
    processed_data = {}

    # 1. Load and apply feature engineering
    for split in ['train', 'validation', 'test']:
        filepath = f'{split}_dataset.csv'
        if not os.path.exists(filepath):
            print(f"[Error] File not found: {filepath}")
            return None

        print(f"\n--- Processing {split} data ---")
        df = pd.read_csv(filepath)
        df_featured = create_financial_features(df)
        datasets[split] = df_featured

    # 2. Fit the preprocessor on the TRAINING data
    print("\n--- Fitting Preprocessor on Training Data ---")
    # Separate features (X) and targets (y) for training data
    X_train = datasets['train'][feature_list]
    # Handle NaNs in targets if any (e.g., using median for regression target)
    y_train_class = datasets['train'][target_list[0]].fillna('Not_Eligible')
    y_train_reg = datasets['train'][target_list[1]].fillna(datasets['train'][target_list[1]].median())

    preprocessor.fit(X_train)
    print("Preprocessor fitted successfully.")

    # 3. Transform all datasets (train, val, test)
    for split in ['train', 'validation', 'test']:
        print(f"--- Transforming {split} data ---")
        df = datasets[split]

        # Separate X and y
        X = df[feature_list]
        y_class = df[target_list[0]].fillna('Not_Eligible')
        y_reg = df[target_list[1]].fillna(df[target_list[1]].median())

        # Transform features
        X_processed = preprocessor.transform(X)

        # Get feature names after transformation (important for MLflow)
        feature_names = preprocessor.get_feature_names_out()

        # Convert processed data back to DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

        # Combine processed X and y
        processed_df = pd.concat([
            X_processed_df,
            y_class.reset_index(drop=True),
            y_reg.reset_index(drop=True)
        ], axis=1)

        # Save processed data
        output_path = f'processed_{split}_dataset.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"Saved processed {split} data to {output_path}")
        processed_data[split] = processed_df

    # 4. Save the fitted preprocessor object
    preprocessor_path = 'model_preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"\n--- Preprocessor saved to {preprocessor_path} ---")

    return processed_data


if __name__ == "__main__":

    print("--- Starting Step 3: Feature Engineering & Preprocessing ---")

    # Load train_dataset to define the preprocessor
    try:
        train_df = pd.read_csv('train_dataset.csv')
    except FileNotFoundError:
        print("[Error] 'train_dataset.csv' not found.")
        print("Please run Step 1 script first.")
        sys.exit(1)

    # Apply feature creation to the sample df
    train_df_featured = create_financial_features(train_df.copy())

    # Define the preprocessor based on the engineered training data
    preprocessor, feature_list, target_list = define_preprocessor(train_df_featured)

    # Run the full processing pipeline on train, val, and test data
    processed_datasets = process_datasets(preprocessor, feature_list, target_list)

    if processed_datasets:
        print("\n--- Step 3 Complete ---")
        print("Processed datasets (train, val, test) and 'model_preprocessor.pkl' are saved.")
        print("\nProcessed Training Data Head:")
        print(processed_datasets['train'].head())