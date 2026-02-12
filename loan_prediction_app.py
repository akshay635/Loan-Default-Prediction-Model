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
from src.feature_engineering import FE
from src.feature_importances import Feature_IMP

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
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Exploration", "🧮 EMI and Credit Score calculator"])

# ---------------- Prediction Tab ----------------
with tab1:
    st.header("Your repayment risk assessment")
    if st.button("🔍 Assess Risk"):
        df, issues = validator.validate_inference(user_data)
        
        if issues:
            st.error(" ".join([i for i in issues]))
            st.stop()

        df = FE(df)

        df = df[RiskConfig.EXPECTED_COLS]
        prob = model.predict_proba(df)
        
        # Narrative risk output
        st.subheader("📈 Repayment Risk assessment")
        risk, action = decision_engine.decide(prob)
        if risk == "HIGH":
          st.error(f"❌ High repayment risk ({prob:.2%})")
          st.markdown("""This application shows a higher-than-average probability of repayment
                          difficulty based on financial indicators such as income stability and
                          debt obligations. The customer has higher chances to stop repayments 
                          and default the loan.""")
        elif risk == "MEDIUM":
          st.warning(f"⚠️ Moderate repayment risk ({prob:.2%})")
          st.markdown("""This assessment indicates a moderate probability (35%-60%) of repayment difficulty
                         based on the available financial information, suggesting that further review may be 
                         appropriate.""")
        else:
          st.success(f"✅ Low risk of repayment ({prob:.2%})")
          st.markdown("""This assessment indicates a lower probability of repayment difficulty,
                         suggesting comparatively lower risk based on the available information.""")
        
        st.markdown(f"**Suggested Action:** {action}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
          st.subheader("Global Feature Importance")
          feature_imp_df = pd.read_csv(RiskConfig.FEATURE_IMP_PATH)
          fig = Feature_IMP(feature_imp_df)
          st.plotly_chart(fig, use_container_width=False)
          with st.expander('Feature Summary'):
              st.markdown(generate_feature_insight(df, feature_imp_df, top_n = 5))
        
        with col2:
          st.subheader("Personalized SHAP Explanation")
          fig = explainer.plot(df)
          st.pyplot(fig, use_container_width=False)
          st.markdown(
          """Features pushing the risk higher are shown in red, 
             while features reducing risk are shown in blue.""")

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
    fig = Feature_IMP(feature_imp_df)
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

with tab3:
    st.header("EMI & Credit Score calculator")
    st.subheader('Calculates EMI and Credit Score based on the given features')
    
    principal_amount = st.text_input("Loan Amount", "1,00,000")
    principal_amount = int(principal_amount.replace(',', ''))
    if principal_amount < 1000:
        st.error('Please enter valid principle amount')
        
    interest_rate = st.slider('Enter the Interest rate(%)', 0.0, 30.0)
    if interest_rate < 0 and interest_rate > 30:
        st.error('Please enter valid interest_rate')
    
    loan_tenure = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    
    monthly_rate = interest_rate / (12 * 100)
    
    # EMI formula
    emi = (principal_amount * monthly_rate * (1 + monthly_rate) ** loan_tenure) / \
          ((1 + monthly_rate) ** tenure_months - 1)
    
    emi = round(emi, 2)
    st.success(st.subheader(f"EMI: {emi}"))
    
st.caption("This dashboard provides readiness estimation only. Final lending decisions must follow business policies.")











































































