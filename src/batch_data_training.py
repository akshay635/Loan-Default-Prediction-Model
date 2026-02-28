import pandas as pd
import streamlit as st
import joblib
import importlib
import src.config as config
importlib.reload(config)
from src.config import RiskConfig

def batch_data_modeling(df):
  
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

  lgd_mapping = {
    "Home": 0.30,
    "Education": 0.40,
    "Auto": 0.50,
    "Business": 0.60,
    "Personal": 0.75
  }
  
  df["Probability"] = y_proba
  df["Prediction"] = y_pred
  df['LGD'] = df['LoanPurpose'].map(lgd_mapping)
  df['Expected_loss'] = (df['Probability']*df['LoanAmount'])*df['LGD']

  return df, y_proba, y_pred, y_true, threshold
      
