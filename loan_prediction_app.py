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

def initialize_components():
    # Initialize components
    validator = SchemaValidator(RiskConfig.EXPECTED_COLS)
    model = LoanRiskModel(RiskConfig.MODEL_PATH)
    decision_engine = RiskDecisionEngine(RiskConfig.LOW_RISK, RiskConfig.HIGH_RISK)
    FE = FeatureEngineering()
    user_data = load_data()
    return validator, model, decision_engine, FE, user_data

# Tabs for storytelling
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Single Borrower Prediction", "Batch Processing",  "📊 Exploration", "🧮 EMI calculator", "💹 Credit Score Calculator"])

with tab1:
    with st.expander("How to interpret this risk score?"):
        st.write(
            "The risk score estimates the likelihood of repayment difficulty "
            "based on historical financial patterns. It should be used as "
            "decision support rather than a definitive outcome."
        )
    # Initialize components
    validator, model, decision_engine, FE, user_data = initialize_components()
    st.header("Your repayment risk assessment")
    if st.button("🔍 Assess Risk"):
        risk_assessor = RiskAssessment(model, validator, FE, decision_engine, RiskConfig)
        risk_assessor.assess(user_data)

with tab2:
    st.header('Batch Processing of data')
    st.markdown('''User will provide the bank data and we will estimate the number of cases where model predicted 
                   as non-default but they are the actual defaulters''')

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(2))
        df['MonthlyIncome'] = df['Income']//12
        df['EMI'] = ((df['LoanAmount']*df['InterestRate']) + df['LoanAmount'])/df['LoanTerm']
        df['EMI'] = round(df['EMI'], 2)
        
with tab3:
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


































































































































