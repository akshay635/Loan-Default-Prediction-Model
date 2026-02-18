# -*- coding: utf-8 -*-

import joblib
import streamlit as st

@st.cache_resource
def load_model(path):
    return joblib.load(path)

class LoanRiskModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        
    @st.cache_data
    def predict_proba(self, df):
        return self.model.predict_proba(df)
        if proba.ndim == 2:
            return proba[:, 1]   # Return all class-1 probabilities
        else:
            return proba





