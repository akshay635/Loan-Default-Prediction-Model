# -*- coding: utf-8 -*-
# importing the modules
import streamlit as st
import numpy as np
import pandas as pd

class SchemaValidator:
    def __init__(self, expected_cols):
        self.expected_cols = expected_cols

    def validate_inference(self, data: dict):
        df = pd.DataFrame([data])
        issues = []
        # Ensure schema
        for col in self.expected_cols:
            if col not in df.columns: 
                issues.append(f'Missing {col} in the data') 

        if df["Age"].isna().any():
            issues.append("Age missing")
        elif not df["Age"].between(18, 71).all():
            issues.append("Age out of range")

        if df["LoanAmount"].isna().any():
            issues.append("LoanAmount missing")
        elif not (df["LoanAmount"] > 0).all():
            issues.append("LoanAmount invalid")

        for col in data.keys():
            if df[col].isna().any():
                issues.append(f"NaN values present in {col}")
            elif df[col].isnull().any():
                issues.append(f"Null values present in {col}")
            else:
                pass 

        # ❗ DO NOT assert on NaNs here
        return df[self.expected_cols], issues
       














