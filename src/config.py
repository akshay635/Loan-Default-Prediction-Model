# -*- coding: utf-8 -*-
class RiskConfig:
    LOW_RISK = 0.35
    HIGH_RISK = 0.60
    
    MODEL_PATH = "models/loan_pred_model_v1.joblib"
    DATA_PATH = "data/Loan_default.csv"
    FEATURE_IMP_PATH = "data/feature_imp.csv"
    CONFUSION_MATRIX = "data/confusion_matrix.csv"
    
    EXPECTED_COLS = [ 
       'Age', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'DTIRatio',
       'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
       'HasDependents', 'LoanPurpose', 'HasCoSigner', 'Monthly_Income', 'EMI',
       'Post_DTI', 'age_post_dti', 'tenure_age_ratio', 'debt_stress'
    ]





