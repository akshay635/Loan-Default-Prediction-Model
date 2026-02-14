# imporrting Pandas
import pandas as pd

class Feature_Engineering:
  def derived_features(df):
    # Feature engineering
    df['EMI/Income_ratio'] = round((df['EMI'] / df['Monthly_Income']), 2)
    df['Post_DTI'] = df['DTIRatio'] + df['EMI/Income_ratio']
    df['age_post_dti'] = df['Age'] * df['Post_DTI']
    df['tenure_age_ratio'] = df['MonthsEmployed'] / (df['Age'] + 1e-6)
    df['debt_stress'] = df['EMI/Income_ratio'] * df['DTIRatio']

    return df
