# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import importlib
import src.config as config
importlib.reload(config)
from src.config import RiskConfig
from src.schema import SchemaValidator
from src.model_service import LoanRiskModel
from src.decision import RiskDecisionEngine
from src.explainability import ShapExplainer
from src.load_data import load_data
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Loan Default Risk Assessment", layout="centered")
st.title("🏦 Loan Default Risk Assessment")

st.markdown(
    "This tool estimates the **risk of loan default** to support informed lending decisions."
)

validator = SchemaValidator(RiskConfig.EXPECTED_COLS)
model = LoanRiskModel(RiskConfig.MODEL_PATH)
decision_engine = RiskDecisionEngine(RiskConfig.LOW_RISK, RiskConfig.HIGH_RISK)
explainer = ShapExplainer(model=model.model)

user_data = load_data()

col1, col2 = st.columns(2)

if st.button("🔍 Assess Risk"):
    df = validator.validate(user_data)
    df['EMI/Income_ratio'] = round((df['EMI'] / df['Monthly_Income']), 2)
    df['Post_DTI'] = df['DTIRatio'] + df['EMI/Income_ratio']
    df['age_post_dti'] = df['Age'] * df['Post_DTI']
    df['tenure_age_ratio'] = df['MonthsEmployed'] / (df['Age'] + 1e-6)
    df['debt_stress'] = df['EMI/Income_ratio'] * df['DTIRatio']
    
    df = df[RiskConfig.EXPECTED_COLS]
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
    
    with col1:
        feature_imp = permutation_importance(model.model, df, np.array(prob), scoring='recall')
        feature_imp_df = pd.DataFrame({
            'Features' : df.columns,
            'Importances' : feature_imp.importances_mean
        }).sort_values(by='Importances', ascending=False)
        # ---------------- Visualization ----------------
        fig = px.bar(
            feature_imp_df.head(10),
            x="Importances",
            y="Features",
            title="Feature Importance / F-score (Random Forest)",
            text_auto=True
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("SHAPLEY explanations")
        fig = explainer.plot(df)
        st.pyplot(fig, use_container_width=False)

st.caption("This system provides risk estimation only. Final decisions must follow business policies.")


























