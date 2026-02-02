# -*- coding: utf-8 -*-
class RiskConfig:
    LOW_RISK = 0.30
    HIGH_RISK = 0.60
    
    MODEL_PATH = "C:/Users/aksha/Documents/models/loan_pred_model_v1.joblib"
    DATA_PATH = "C:/Users/aksha/Documents/data/loan_path.csv"
    FEATURE_IMP_PATH = "C:/Users/aksha/Documents/data/feature_imp.csv"
    CONFUSION_MATRIX = "C:/Users/aksha/Documents/data/feature_imp.csv"
    
    EXPECTED_COLS = [
    'Age', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'DTIRatio',
       'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
       'HasDependents', 'LoanPurpose', 'HasCoSigner', 'Monthly_Income', 'EMI',
       'Post_DTI', 'age_post_dti', 'tenure_age_ratio', 'debt_stress'
    ]


