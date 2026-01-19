# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:03:35 2026

@author: aksha
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.config import RiskConfig
from src.schema import SchemaValidator
from src.model_service import LoanRiskModel
from src.decision import RiskDecisionEngine
from src.explainability import ShapExplainer

st.set_page_config(page_title="Loan Default Risk Assessment", layout="centered")
st.title("🏦 Loan Default Risk Assessment")

st.markdown(
    "This tool estimates the **risk of loan default** to support informed lending decisions."
)

config = RiskConfig()
validator = SchemaValidator(config.EXPECTED_COLS)
model = LoanRiskModel(config.MODEL_PATH)
decision_engine = RiskDecisionEngine(config.LOW_RISK, config.HIGH_RISK)
explainer = ShapExplainer(model.model)
cat_model = joblib.load('models/catboost_model_v1.joblib')

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
name_bank = st.sidebar.text_input(placeholder="Please enter bank name", label='Bank Name')

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

if st.button("🔍 Assess Risk"):
    df = validator.validate(user_data)
    prob = model.predict_proba(df)
    st.subheader("📈 Risk Assessment Result")
    risk, action = decision_engine.decide(prob)
    if risk == "HIGH":
        st.error(f"❌ Estimated Risk of Default ({prob:.2%})")
    elif risk == "MEDIUM":
        st.warning(f"⚠️ Estimated Risk of Default ({prob:.2%})")
    else:
        st.success(f"✅ Estimated Risk of Default ({prob:.2%})")

    st.markdown(f"**Suggested Action:** {action}")


    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        df = validator.validate(user_data)
        prob = cat_model.predict_proba(df)[0, 1]
        feature_importances = pd.DataFrame({
            'Features': df.columns,
            'Importances': cat_model.feature_importances_
        })
        
        fig = px.bar(
                feature_importances.sort_values(by='Importances', ascending=False).head(10),
                x="Importances",
                y="Features",
                title="Feature Importance / F-score (Catboost)",
                text_auto=True,
                orientation='v'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("SHAPLEY explanations")
        st.text('Displaying important features which contribute to the final outcome')
        fig = explainer.plot(df)
        st.pyplot(fig, use_container_width=True)

    st.caption("This system provides risk estimation only. Final decisions must follow business policies.")









