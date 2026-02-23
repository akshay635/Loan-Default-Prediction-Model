# -*- coding: utf-8 -*-

import joblib
import streamlit as st

@st.cache_resource
def load_model(path):
    return joblib.load(path)

class LoanRiskModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        
    def predict_proba(self, df):
        proba = self.model.predict_proba(df)[0, 1]
        return proba









