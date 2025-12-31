# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 12:37:04 2025

@author: aksha
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    return joblib.load("C:/Users/aksha/Documents/credit_risk_pipeline_v1.joblib")

model = load_model()

EXPECTED_COLS = model.feature_names_in_

st.title("🏦 Loan Default Risk Assessment")
st.markdown(
    "This tool estimates the **risk of loan default** to support informed lending decisions."
)

# --------------------------------------------------
# Input Sections
# --------------------------------------------------
st.header("👤 Applicant Profile")

name = st.text_input('Enter your full name:')
age = st.slider("Age", 18, 70, 35)
education = st.selectbox("Education Level", ["High School", "Graduate", "Post Graduate"])
employment = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
months_employed = st.slider("Months Employed", 0, 480, 60)

st.header("💰 Financial Information")

income = st.number_input("Annual Income", min_value=0, value=10000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 10.5)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)

st.header("📊 Credit & Household Details")

credit_score = st.slider("Credit Score", 300, 850, 720)
num_credit_lines = st.slider("Number of Credit Lines", 0, 15, 4)
has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

st.header("📌 Loan Purpose")

loan_purpose = st.selectbox(
    "Purpose of Loan",
    ["Home", "Education", "Personal", "Auto", "Business"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

monthly_income = round(income/12, 2)
emi = round(((loan_amount*interest_rate)+loan_amount)/loan_term, 2)
# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Assess Risk"):

    user_data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        'monthly_income': monthly_income,
        'EMI': emi,
        "DTIRatio": dti,
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": cosigner
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

    if prob > 0.45:
        st.error(f"⚠️ High Risk of Default ({prob:.2%})")
        st.markdown("**Suggested Action:** Reject or apply stricter loan terms")
    elif prob >= 0.30 and prob <= 0.45:
        st.warning(f"⚠️ Medium Risk of Default ({prob:.2%})")
        st.markdown("**Suggested Action:** Manual review recommended")
    else:
        st.success(f"✅ Low Risk of Default ({prob:.2%})")
        st.markdown("**Suggested Action:** Loan can be approved")

    st.caption(
        "This system provides risk estimation only. Final decisions must follow business policies."
    )
