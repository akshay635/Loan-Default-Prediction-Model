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
from src.load_data import load_data

st.set_page_config(page_title="Loan Default Risk Assessment", layout="wide")
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

user_data = load_data()

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
                text_auto=True
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("SHAPLEY explanations")
        st.text('Displaying important features which contribute to the final outcome')
        fig = explainer.plot(df)
        st.pyplot(fig, use_container_width=True)

    st.caption("This system provides risk estimation only. Final decisions must follow business policies.")
















