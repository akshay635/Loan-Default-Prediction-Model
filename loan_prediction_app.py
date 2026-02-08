# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import importlib
import src.config as config
importlib.reload(config)
from src.config import RiskConfig
from src.schema import SchemaValidator
from src.model_service import LoanRiskModel
from src.decision import RiskDecisionEngine
from src.explainability import ShapExplainer
from src.load_data import load_data
from src.insights import generate_feature_insight
from src.prediction_tab1 import pred_tab
from src.exploration_tab2 import exp_tab

# Page setup
st.set_page_config(page_title="Loan Risk Assessment System", layout="wide")

st.title("💡 Loan Risk Assessment & Decision System")
st.markdown(
    """
    This system evaluates applicant risk and explains 
    loan approval decisions using data-driven evidence.
    """
)

with st.expander("How to interpret this risk score?"):
    st.write(
        "The risk score estimates the likelihood of repayment difficulty "
        "based on historical financial patterns. It should be used as "
        "decision support rather than a definitive outcome."
    )

# Initialize components
validator = SchemaValidator(RiskConfig.EXPECTED_COLS)
model = LoanRiskModel(RiskConfig.MODEL_PATH)
decision_engine = RiskDecisionEngine(RiskConfig.LOW_RISK, RiskConfig.HIGH_RISK)
explainer = ShapExplainer(model=model.model)

user_data = load_data()

# Tabs for storytelling
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Exploration"])

# ---------------- Prediction Tab ----------------
pred_tab()

# ---------------- Exploration Tab ----------------
exp_tab()

st.caption("This dashboard provides readiness estimation only. Final lending decisions must follow business policies.")



































































