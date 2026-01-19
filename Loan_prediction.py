# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 12:37:04 2025

@author: aksha
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
# --------------------------------------------------
# App Layout
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Default Risk Assessment",
    layout="centered"
)

# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("catboost_model_v1.joblib")

model = load_model()

EXPECTED_COLS = ['Age', 'LoanAmount',
                 'CreditScore', 'MonthsEmployed', 'NumCreditLines',
                 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education', 
                 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                 'HasDependents', 'LoanPurpose', 'HasCoSigner', 
                 'Monthly_Income', 'EMI']



st.title("🏦 Loan Default Risk Assessment")
st.markdown(
    "This tool estimates the **risk of loan default** to support informed lending decisions."
)

# --------------------------------------------------
# Input Sections
# --------------------------------------------------
st.sidebar.header("👤 Applicant Profile")

# Loan ID
loan_id = st.sidebar.text_input('Enter your Loan ID')

# Name of the applicant/customer
name = st.sidebar.text_input('Enter your full name:')

# Age
age = st.sidebar.slider("Age", 18, 70, 35)

# Education level
education = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Post Graduate"])

# Employment status
employment = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])

# Marital status
marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

months_employed, income, monthly_income, emi = 0, 0, 0, 0

st.sidebar.header("💰 Financial Information & Credit history")

# Applying conditions based on employment status
if employment == "Salaried" or employment == 'Self-employed':
    # Number of Months being employed till now
    months_employed = st.sidebar.slider('Months employed', 0, 480, 60)
    # Annual Income
    income = st.sidebar.text_input("Annual Income", "10,000")
    # type conversion from string to int
    income = int(income.replace(',', ''))
    # Monthly Income
    monthly_income = round(income//12, 2)

# Debt-To-Income ratio
dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
# Credit Score
credit_score = st.sidebar.slider("Credit Score", 300, 850, 720)
# Number of Credit lines
num_credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 15, 4)

st.sidebar.header("📊 Household Details")

has_mortgage = st.sidebar.selectbox("Has Mortgage?", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
cosigner = st.sidebar.selectbox("Has Co-Signer?", ["Yes", "No"])

st.sidebar.header("📌 Loan Details")

# Name of the Bank
name_bank = st.sidebar.text_input(placeholder="Please enter bank name")

# Type of Loan
loan_purpose = st.sidebar.selectbox(
    "Purpose of Loan",
    ["Home", "Education", "Personal", "Auto", "Business"]
)
# Loan amount required/taken from the bank
loan_amount = st.sidebar.text_input("Loan Amount", "1,00,000")
loan_amount = int(loan_amount.replace(',', ''))
# Interest Rate
interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.5)
# Loan Tenure
loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

# Equated Monthly Installments(EMI)
emi = round(((loan_amount*interest_rate)+loan_amount)/loan_term, 2)

# Input data which will be passed to the model
user_data = {
        "Age": age,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti,
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": cosigner,
        "Monthly_Income": monthly_income,
        "EMI": emi
}
# --------------------------------------------------
# Prediction
# --------------------------------------------------

col1, col2 = st.columns(2)

if st.button("🔍 Assess Risk"):
    with col1:

        # Converting input data into DataFrame
        df = pd.DataFrame([user_data])

        # Enforce schema
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[EXPECTED_COLS]

        # Predict estimated probability of Loan default
        prob = model.predict_proba(df)[0, 1]

        st.subheader("📈 Risk Assessment Result")
        
        # Setting up the thresholds
        if prob >= 0.6:
            st.error(f"❌ Estimated Risk of Default ({prob:.2%})")
            st.markdown("**Suggested Action:** Reject or apply stricter loan terms")
        elif prob > 0.3 and prob < 0.6:
            st.warning(f"⚠️ Estimated Risk of Default ({prob:.2%})")
            st.markdown("**Suggested Action:** Manual review recommended")
        else:
            st.success(f"✅ Estimated Risk of Default ({prob:.2%})")
            st.markdown("**Suggested Action:** Future Loan can be approved if applied")

        """
        - Low → <30%
        - Medium → 30%-60%
        - High → >60%
        """

    with col2
        df = pd.DataFrame([user_data])
    
        # Enforce schema
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = np.nan
    
        df = df[EXPECTED_COLS]
    
        # estimating the probability of Loan default
        prob = model.predict_proba(df)[0, 1]

        # SHAPley explanations to check the contribution of each feature to decide the future outcome
        st.subheader('SHAPLEY explanations')
        st.text('Features contribution in deciding the final outcome')
        exp = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = exp(df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(fig, use_container_width=True, width='stretch')

st.caption("This system provides risk estimation only. Final decisions must follow business policies.")






















