# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:57:40 2026

@author: aksha
"""

class RiskConfig:
    LOW_RISK = 0.30
    HIGH_RISK = 0.60

    MODEL_PATH = "C:/Users/aksha/Documents/models/catboost_model_v1.joblib"

    EXPECTED_COLS = [
        'Age', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose',
        'HasCoSigner', 'Monthly_Income', 'EMI'
    ]
