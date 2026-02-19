import pandas as pd
import streamlit as st
import joblib
from src.config import RiskConfig

def batch_data_modeling(df):
  df'MonthlyIncome'] = df['Income']//12
  df['EMI'] = ((df['LoanAmount']*df['InterestRate']) + df['LoanAmount'])/df['LoanTerm']
  df['EMI'] = round(df['EMI'], 2)
  df['EMI/Income_ratio'] = round((df['EMI'] / df['MonthlyIncome']), 2)
  df['Post_DTI'] = df['DTIRatio'] + df['EMI/Income_ratio']
  df['age_post_dti'] = df['Age'] * df['Post_DTI']
  df['tenure_age_ratio'] = df['MonthsEmployed'] / (df['Age'] + 1e-6)
  df['debt_stress'] = df['EMI/Income_ratio'] * df['DTIRatio']

  required_cols = RiskConfig.EXPECTED_COLS + RiskConfig.TARGET_COL
  missing_cols = [col for col in required_cols if col not in df.columns]

  if missing_cols:
      st.error(f"Missing required columns: {missing_cols}")
      st.stop()
  
  st.header("⚙️ Decision Configuration")

  threshold = st.slider(
      "Decision Threshold",
      min_value=0.0,
      max_value=1.0,
      value=0.5,
      step=0.01
  )

  y_true = df[RiskConfig.TARGET_COL]
  X_batch = df[RiskConfig.EXPECTED_COLS]
  
  model = joblib.load(RiskConfig.MODEL_PATH)

  y_proba = model.predict_proba(X_batch)[:, 1]
  y_pred = (y_proba >= threshold).astype(int)
  
  df["Probability"] = y_proba
  df["Prediction"] = y_pred

  return df, y_proba, y_pred, y_true
      
