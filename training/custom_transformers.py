from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureAdder(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
    
  def fit(self, X, y=None):
    return self
    
  def transform(self, X):
    # your logic
    X = X.copy()
    # Example: add ratio of feature1 / feature2
    X['MonthlyIncome'] = X['Income'] // 12
    X['EMI'] = (((X['LoanAmount']*(X['InterestRate']/100))) + X['LoanAmount'])/(X['LoanTerm'])
    X['EMI'] = round(X['EMI'], 2)
    X['EMI/Income_ratio'] = np.where(X['MonthlyIncome']<=0, 0, X['EMI']/X['MonthlyIncome'])
    X['Post_DTI'] = X['DTIRatio'] + X['EMI/Income_ratio']
    X['age_post_dti'] = X['Age'] * X['Post_DTI']
    X['tenure_age_ratio'] = X['MonthsEmployed'] / (X['Age'] + 1e-6)
    X['debt_stress'] = X['EMI/Income_ratio'] * X['DTIRatio']
    X = X.drop(columns=['Income'])
    return X

class ConditionalLogTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, numeric_cols, threshold=1.0):
    self.threshold = threshold
    self.skewed_cols = []
    self.numeric_cols = numeric_cols
    
  def fit(self, X, y=None):
    skewness = X[self.numeric_cols].skew()
    self.skewed_cols = skewness[skewness > self.threshold].index.tolist()
    return self
    
  def transform(self, X):
    X = X.copy()
    for col in self.skewed_cols:
      # log1p handles zeros safely
      X[col] = np.log1p(X[col])
    return X
