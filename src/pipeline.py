from src.custom_transformers import FeatureAdder, ConditionalLogTransformer
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():
  df = pd.read_csv("data/train.csv")

  X = df.drop(columns=['Default', 'LoanID'])
  y = df[['Default']]

  numeric_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
  categoric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

  num_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
  ])
  
  cat_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
  ])
  
  preprocessor = ColumnTransformer(transformers=[
      ('num', num_transformer, numeric_cols),
      ('cat', cat_transformer, categoric_cols),
  ], remainder='passthrough')
  
  neg_count = np.sum(y['Default'] == 0)
  pos_count = np.sum(y['Default'] == 1)

  scale_pos_weight = neg_count / pos_count
  scale_pos_weight = round(scale_pos_weight, 2)
  
  lg = LogisticRegression(
          max_iter=400, random_state=42,
          class_weight={0: 1.0, 1: scale_pos_weight},
          solver="lbfgs", l1_ratio=0.0, C=2.1544)
  
  lg_pipe = Pipeline(steps = [
    ('FE', FeatureAdder()),
    ('transformer', ConditionalLogTransformer(threshold=1.0, numeric_cols)),
    ('preprocessing', preprocessor),
    ('ml_model', lg)
  ])
  
  lg_pipe.fit(X, y.values.ravel())
  
  joblib.dump(lg_pipe, 'models/loan_pred_model_v3.joblib')

if __name__ == "__main__":
  main()
