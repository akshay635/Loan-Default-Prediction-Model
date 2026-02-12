import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.feature_engineering import FE
from src.feature_importances import Feature_IMP

class RiskAssessment:
    def __init__(self, model, validator, FE, decision_engine, RiskConfig, explainer):
        self.model = model
        self.validator = validator
        self.FE = FE
        self.decision_engine = decision_engine
        self.RiskConfig = RiskConfig
        self.explainer = explainer

    def assess(self, user_data):
        df, issues = self.validator.validate_inference(user_data)
        if issues:
            st.error(" ".join(issues))
            st.stop()

        df = self.FE(df)[self.RiskConfig.EXPECTED_COLS]
        prob = self.model.predict_proba(df)
        risk, action = self.decision_engine.decide(prob)

        # Narrative output
        if risk == "HIGH":
            st.error(f"❌ High repayment risk ({prob:.2%})")
        elif risk == "MEDIUM":
            st.warning(f"⚠️ Moderate repayment risk ({prob:.2%})")
        else:
            st.success(f"✅ Low risk of repayment ({prob:.2%})")

        st.markdown(f"**Suggested Action:** {action}")

        # Feature importance + SHAP
        col1, col2 = st.columns([1, 1])
        with col1:
            feature_imp_df = pd.read_csv(self.RiskConfig.FEATURE_IMP_PATH)
            fig = Feature_IMP(feature_imp_df)
            st.plotly_chart(fig)
        with col2:
            fig = self.explainer.plot(df)
            st.pyplot(fig)


class Exploration:
    def __init__(self, RiskConfig):
        self.RiskConfig = RiskConfig

    def show(self):
        st.header("Explore Model Insights")
        feature_imp_df = pd.read_csv(self.RiskConfig.FEATURE_IMP_PATH)
        labels = feature_imp_df['Features'].head().tolist()
        values = feature_imp_df['Importances'].head().tolist()

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.3, 0])])
        st.plotly_chart(fig)

        top_features = feature_imp_df.sort_values("Importances", ascending=False).head(3)
        st.markdown(
            f"""🗣️ Dynamic Insights:  
            Model is most influenced by **{top_features.iloc[0]['Features']}**, 
            then **{top_features.iloc[1]['Features']}**, 
            and **{top_features.iloc[2]['Features']}**."""
        )


class EMICalculator:
    def __init__(self, principal, rate, tenure):
        self.principal = principal
        self.rate = rate
        self.tenure = tenure

    def calculate(self):
        monthly_rate = self.rate / (12 * 100)
        emi = (self.principal * monthly_rate * (1 + monthly_rate) ** self.tenure) / \
              ((1 + monthly_rate) ** self.tenure - 1)
        return round(emi, 2)

    def plot(self, emi):
        total_loan_amount = self.principal + self.principal * (self.rate / 100)
        interest_amount = total_loan_amount - self.principal
        labels = ['Total Loan', 'Principal amount', 'Interest amount']
        values = [total_loan_amount, self.principal, interest_amount]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.3, 0])])
        st.plotly_chart(fig)

class CreditScoreCalculator:
    def __init__(self, payment_history, utilization, history_years, inquiries):
        self.payment_history = payment_history
        self.utilization = utilization
        self.history_years = history_years
        self.inquiries = inquiries

    def normalize_features(self):
        """Normalize all features to [0,1] scale."""
        PH_norm = self.payment_history / 100
        CU_norm = 1 - self.utilization
        LH_norm = min(self.history_years, 20) / 20
        NC_norm = 1 - min(self.inquiries, 10) / 10
        return PH_norm, CU_norm, LH_norm, NC_norm

    def calculate_score(self):
        """Calculate credit score using weighted formula."""
        PH_norm, CU_norm, LH_norm, NC_norm = self.normalize_features()
        score = 300 + 550 * (
            0.40 * PH_norm +
            0.30 * CU_norm +
            0.20 * LH_norm +
            0.10 * NC_norm
        )
        return round(score)

    def plot_gauge(self):
        """Return a Plotly gauge chart for visualization."""
        score = self.calculate_score()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Credit Score"},
            gauge={
                'axis': {'range': [300, 850]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [300, 550], 'color': "red"},
                    {'range': [550, 650], 'color': "orange"},
                    {'range': [650, 750], 'color': "yellow"},
                    {'range': [750, 850], 'color': "green"}
                ],
            }
        ))
        return fig
