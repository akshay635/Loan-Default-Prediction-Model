# -*- coding: utf-8 -*-
# importing required modules, builtins and classes
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
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
from src.feature_engineering import FeatureEngineering
from src.outputs import RiskAssessment, Exploration, EMICalculator, CreditScoreCalculator

# Page setup
st.set_page_config(page_title="Loan Risk Assessment System", layout="wide")

st.title("💡 Loan Risk Assessment & Decision System")
st.markdown(
    """
    This system evaluates applicant risk and explains 
    loan approval decisions using data-driven evidence.
    """
)


# Tabs for storytelling
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Single Borrower Prediction", "Batch Processing", "📊 Exploration", "🧮 EMI calculator", "💹 Credit Score Calculator"])

with tab1:
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
    FE = FeatureEngineering()
    user_data = load_data()
    st.header("Your repayment risk assessment")
    if st.button("🔍 Assess Risk"):
        risk_assessor = RiskAssessment(model, validator, FE, decision_engine, RiskConfig, explainer)
        risk_assessor.assess(user_data)

with tab2:
    st.title("📊 Portfolio Risk Evaluation – Batch Processing")

    st.markdown("""
    Upload a borrower dataset to perform portfolio-level risk scoring.
    The model applies cost-sensitive learning and threshold-based decision logic.
    """)

    uploaded_file = st.file_uploader(
    "Upload CSV file containing borrower data (must include target column)",
    type=["csv"])
    
    FE = FeatureEngineering()
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        df_batch['MonthlyIncome'] = round((df_batch['Income']//12), 2)
        df_batch['EMI'] = ((df_batch['LoanAmount']*df_batch['InterestRate']) + df_batch['LoanAmount'])/(df_batch['LoanTerm'])
        df_batch['EMI'] = round(df_batch['EMI'], 2)
        new_df = FE.derived_features(df_batch)
        
        del df_batch
        
        required_cols = RiskConfig.EXPECTED_COLS + RiskConfig.TARGET_COL

        missing_cols = [col for col in required_cols if col not in new_df.columns]
    
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
    
        st.success(f"File uploaded successfully. Records detected: {len(df_batch)}")
    
        st.subheader("Preview of Uploaded Data")
        st.dataframe(new_df.head())
    
        st.sidebar.header("⚙️ Decision Configuration")
    
        threshold = st.sidebar.slider("Decision Threshold",
                                      min_value=0.0, max_value=1.0,
                                      value=0.5, step=0.01)
    
        y_true = new_df[RiskConfig.TARGET_COL]
        X_batch = new_df[RiskConfig.EXPECTED_COLS]
        
        @st.cache_resource
        def run_batch_prediction(model, X):
            return model.predict_proba(X)

        if st.button("🚀 Run Batch Evaluation"):
            with st.spinner("Processing portfolio..."):
                y_proba = run_batch_prediction(model, X_batch)
                y_pred = (y_proba >= threshold).astype(int)
        
        new_df["Probability"] = y_proba
        new_df["Prediction"] = y_pred
    
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        recall = tp / (tp + fn)
        miss_rate = fn / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        flagged_rate = y_pred.mean()
    
        st.header("📌 Portfolio Summary")
    
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Records", len(new_df))
        col2.metric("Flagged High Risk", f"{flagged_rate*100:.2f}%")
        col3.metric("Recall (Catch Rate)", f"{recall*100:.2f}%")
        col4.metric("Miss Rate", f"{miss_rate*100:.2f}%")
    
        st.subheader("🔎 Confusion Matrix")
    
        st.write(f"""
        - True Positives: {tp}
        - False Positives: {fp}
        - True Negatives: {tn}
        - False Negatives: {fn}
        """)
    
        new_df["Risk Bucket"] = pd.cut(y_proba, bins=[0, 0.3, 0.6, 1],
                                       labels=["Low Risk", "Medium Risk", "High Risk"])
    
        st.subheader("📊 Risk Segmentation Distribution")
        
        st.bar_chart(new_df["Risk Bucket"].value_counts())
    
        st.subheader("⬇️ Export Scored Portfolio")
    
        st.download_button(
            label="Download Scored Dataset",
            data=new_df.to_csv(index=False),
            file_name="scored_portfolio.csv",
            mime="text/csv"
        )
    
        st.info(f"""
        At threshold {threshold}, the model detects {recall*100:.1f}% of defaulters 
        while missing {miss_rate*100:.1f}%. Approximately {flagged_rate*100:.1f}% 
        of the portfolio is flagged for review.""")

with tab3:
    explainer = ShapExplainer(model=model.model)
    explorer = Exploration(RiskConfig)
    explorer.show()

with tab4:
    principal = st.number_input('Enter the principal amount')
    if principal < 1000:
        st.error('Please enter valid amount')
    rate = st.slider('Enter the Interest rate(%)', 1.0, 30.0)
    
    if rate < 1.0 and rate > 30.0:
        st.error('Please provide valid interest rate')
        
    tenure = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    emi_calc = EMICalculator(principal, rate, tenure)
    emi = emi_calc.calculate()
    st.subheader(f"EMI: ₹{emi}/-")
    emi_calc.plot(emi)

with tab5:
    payment_history = st.slider('Payment History(%)', 0, 100)
    cu_ratio = st.slider('Credit Utilization ratio', 0.0, 1.0)
    history_years = st.number_input('Credit History(in years)', 0)
    credit_inquiries = st.number_input('No of credit inquiries', 0)
    
    calc = CreditScoreCalculator(payment_history, cu_ratio, history_years, credit_inquiries)
    score = calc.calculate_score()
    st.success(f"Credit Score: {score}")

    # To display gauge in Streamlit:
    st.plotly_chart(calc.plot_gauge())




















































































































