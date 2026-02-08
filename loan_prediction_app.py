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
from src.prediction_tab1 import tab1

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
tab1()

# ---------------- Exploration Tab ----------------
with tab2:
    st.header("Explore Model Insights")
    st.markdown(
        """
        This section helps you understand the **bigger picture**:  
        - Which features matter most overall  
        - How borrowers compare across different profiles  
        - Narrative insights into repayment stability
        """
    )

    feature_imp_df = pd.read_csv(RiskConfig.FEATURE_IMP_PATH)
    st.subheader("📊 Global Feature Importance")
    fig = px.bar(
        feature_imp_df,
        x="Importances",
        y="Features",
        title="Overall Feature Importance",
        text_auto=True
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    top_features = feature_imp_df.sort_values("Importances", ascending=False).head(3)
    st.markdown(
        f"""
        🗣️ **Dynamic Insights:**  
        Right now, the model is most influenced by **{top_features.iloc[0]['Features']}**, 
        followed by **{top_features.iloc[1]['Features']}** and **{top_features.iloc[2]['Features']}**.  
        This means these features are the strongest drivers of repayment readiness in your profile.
        """
    )

st.caption("This dashboard provides readiness estimation only. Final lending decisions must follow business policies.")































































