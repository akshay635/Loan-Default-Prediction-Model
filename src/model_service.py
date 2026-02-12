# -*- coding: utf-8 -*-

import joblib
import streamlit as st

class LoanRiskModel:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        
    @st.cache_resource
    def _load_model(_self, path):
        return joblib.load(path)

    def predict_proba(self, df):
        return self.model.predict_proba(df)[0, 1]


