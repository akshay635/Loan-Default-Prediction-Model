# -*- coding: utf-8 -*-
# importing the modules
import numpy as np
import pandas as pd

class SchemaValidator:
    def __init__(self, expected_cols):
        self.expected_cols = expected_cols

    def validate_inference(self, data: dict):
        df = pd.DataFrame([data])
        # Ensure schema
        for col in self.expected_cols:
            if col not in df.columns: 
                df[col] = np.nan 
        
        issues = []

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
                issues.append(f"Null/NaN values present in {col}")
            else:
                pass 

        # ❗ DO NOT assert on NaNs here
        return df[self.expected_cols], issues
       










