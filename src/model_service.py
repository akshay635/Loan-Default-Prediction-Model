# -*- coding: utf-8 -*-

import joblib
import streamlit as st

class LoanRiskModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        
    @st.cache_resource
    def load_model(self, path):
        return joblib.load(path)
        
    @st.cache_data
    def predict_proba(self, df):
        return self.model.predict_proba(df)[0, 1]




