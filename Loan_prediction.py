# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 08:47:04 2025
@author: aksha
"""
# importing necessary libraries
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
    return joblib.load("catboost_model_v1.joblib")

model = load_model()

# expected no of cols in historical data through which new user data will come as an input
EXPECTED_COLS = ['Age', 'LoanAmount',
                 'CreditScore', 'MonthsEmployed', 'NumCreditLines',
                 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education', 
                 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                 'HasDependents', 'LoanPurpose', 'HasCoSigner', 
                 'Monthly_Income', 'EMI']


# title of the page
st.title("🏦 Loan Default Risk Assessment")
st.markdown(
    "This tool estimates the **risk of loan default** to support informed lending decisions."
)

# --------------------------------------------------
# Input Sections
# --------------------------------------------------
st.header("👤 Applicant Profile")

# Loan ID of the customer
loan_id = st.text_input('Enter your Loan ID')

# Name of the customer
name = st.text_input('Enter your full name:')

# Age of the customer
age = st.slider("Age", 18, 70, 35)

# Education status of the customer
education = st.selectbox("Education Level", ["High School", "Graduate", "Post Graduate"])

# Employment status of the customer
employment = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])

# marital status of the customer
marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

# set to zero for Unemployed category
months_employed = 0
income = 0

st.header("💰 Financial Information & Credit history")

# changes the month employed and income if customer is employed or self-employed
if employment == "Salaried" or employment == 'Self-employed':
    months_employed = st.slider('Months employed', 0, 480, 60)
    income = st.number_input("Annual Income", min_value=0, value=10000)

# Debt-to-Income Ratio
dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)

# Credit Score 
credit_score = st.slider("Credit Score", 300, 850, 720)

# Number of credit lines
num_credit_lines = st.slider("Number of Credit Lines", 0, 15, 4)

st.header("📊 Household Details")

# mortgage
has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])

# dependents
has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

# co-signer
cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

st.header("📌 Loan Details")

# Loan purpose
loan_purpose = st.selectbox(
    "Purpose of Loan",
    ["Home", "Education", "Personal", "Auto", "Business"]
)

# Loan amount
loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)

# Interest rate
interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 10.5) 

# Loan term
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

# Monthly Income
monthly_income = round(income//12, 2)

# Equated Monthly Installments
emi = round(((loan_amount*interest_rate)+loan_amount)/loan_term, 2)
# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Assess Risk"):

    # user inputs into dictonary (key-value pair) format
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

    # DataFrame
    df = pd.DataFrame([user_data])

    # Enforce schema to check whether columns are missing
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[EXPECTED_COLS]

    # Predict probability
    prob = model.predict_proba(df)[0, 1]

    st.subheader("📈 Default Risk Assessment Result")

    # probability threshold settings for Default Risk assessment
    if prob >= 0.6:
        st.error(f"⚠️ High Risk of Default ({prob:.2%})")
        st.markdown("**Suggested Action:** Reject or apply stricter loan terms")
    elif prob > 0.3 and prob < 0.6:
        st.warning(f"⚠️ Medium Risk of Default ({prob:.2%})")
        st.markdown("**Suggested Action:** Manual review recommended")
    else:
        st.success(f"✅ Low Risk of Default ({prob:.2%})")
        st.markdown("**Suggested Action:** Future Loan can be approved if applied")

    st.caption(
        "This system provides risk estimation only. Final decisions must follow business policies."
    )





