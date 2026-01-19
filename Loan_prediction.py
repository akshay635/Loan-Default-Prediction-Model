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

loan_id = st.sidebar.text_input('Enter your Loan ID')
name = st.sidebar.text_input('Enter your full name:')
age = st.sidebar.slider("Age", 18, 70, 35)
education = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Post Graduate"])
employment = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

months_employed = 0
income = 0

st.sidebar.header("💰 Financial Information & Credit history")

if employment == "Salaried" or employment == 'Self-employed':
    months_employed = st.sidebar.slider('Months employed', 0, 480, 60)
    income = st.sidebar.text_input("Annual Income")
dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 720)
num_credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 15, 4)

st.sidebar.header("📊 Household Details")

has_mortgage = st.sidebar.selectbox("Has Mortgage?", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
cosigner = st.sidebar.selectbox("Has Co-Signer?", ["Yes", "No"])

st.sidebar.header("📌 Loan Details")

loan_purpose = st.sidebar.selectbox(
    "Purpose of Loan",
    ["Home", "Education", "Personal", "Auto", "Business"]
)
loan_amount = st.sidebar.text_input("Loan Amount")
interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.5)
loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

income = int(income.replace(',', '')
loan_amount = int(loan_amount.replace(',', '')
monthly_income = round(income//12, 2)
emi = round(((loan_amount*interest_rate)+loan_amount)/loan_term, 2)
# --------------------------------------------------
# Prediction
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Assess Risk"):
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

        df = pd.DataFrame([user_data])

        # Enforce schema
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[EXPECTED_COLS]

        # Predict probability
        prob = model.predict_proba(df)[0, 1]

        st.subheader("📈 Risk Assessment Result")

        if prob >= 0.6:
            st.error(f"⚠️ High Risk of Default ({prob:.2%})")
            st.markdown("**Suggested Action:** Reject or apply stricter loan terms")
        elif prob > 0.3 and prob < 0.6:
            st.warning(f"⚠️ Medium Risk of Default ({prob:.2%})")
            st.markdown("**Suggested Action:** Manual review recommended")
        else:
            st.success(f"✅ Low Risk of Default ({prob:.2%})")
            st.markdown("**Suggested Action:** Future Loan can be approved if applied")

with col2:
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

    df = pd.DataFrame([user_data])

    # Enforce schema
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[EXPECTED_COLS]

    # Predict probability
    prob = model.predict_proba(df)[0, 1]
    st.header('SHAPLEY explanations')
    st.text('Features impacting over the outcome')
    exp = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = exp(df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig, use_container_width=True)

st.caption("This system provides risk estimation only. Final decisions must follow business policies.")






