import pandas as pd
import numpy as np
import joblib
import json 
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from training.custom_transformers import FeatureAdder, ConditionalLogTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def main():
  np.random.seed(42)

  pd.set_option('display.max_columns', 500)
  
  # 1) data ingestion
  df = pd.read_csv('data/Loan_default.csv')
  
  # 2) Data validation
  print(df.columns)
  print(df.info())
  print(df.isna().sum())
  print(df.isnull().sum())
  df = df.dropna()
  df = df.drop_duplicates()
  
  # 3) Train-test data split
  X = df.drop(columns=['LoanID', 'Default'])
  y = df[['Default']]
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  
  # 4) Handling class imbalance using scale_pos_weight
  neg_count = np.sum(y['Default'] == 0)
  pos_count = np.sum(y['Default'] == 1)
  
  scale_pos_weight = neg_count / pos_count
  
  scale_pos_weight = round(scale_pos_weight)
  
  # 5) Feature Transformation & Engineering
  num_cols = X_train.select_dtypes(include=['int', 'float']).columns.tolist()
  cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
  
  num_cols.remove('Income')
  
  print(num_cols)
  print(cat_cols)
  
  num_transformer = Pipeline(steps=[
      ('impute', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
  ])
  
  cat_transformer = Pipeline(steps=[
      ('impute', SimpleImputer(strategy='most_frequent')),
      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
  ])
  
  preprocessor = ColumnTransformer(transformers=[
      ('num', num_transformer, num_cols),
      ('cat', cat_transformer, cat_cols)
  ])
  
  # 5) Model training and evaluation using cross-validation techniques and folding mechanism
  models = {
      'Log_Reg': LogisticRegression(random_state=42, class_weight={0:1.0, 1:scale_pos_weight}, max_iter=2000),
      'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight={0:1.0, 1:scale_pos_weight}),
      'Random Forest': RandomForestClassifier(n_estimators=1000, random_state=42, class_weight={0:1.0, 1:scale_pos_weight}),
      'XGBoost': XGBClassifier(random_state=42, n_estimators=1000, scale_pos_weight=scale_pos_weight),
      'LightGBM': LGBMClassifier(n_estimators=1000, class_weight={0: 1.0, 1: scale_pos_weight}, num_leaves=31, random_state=42)
  }
  
  scoring = {'Accuracy': 'accuracy',
             'Precision': 'precision', 
             'Recall': 'recall',
             'F1_score': 'f1',
             'Roc-auc': 'roc_auc',
             'PR-auc': 'average_precision'}
  
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  
  row = []
  for name, model in models.items():
      
      pipe = Pipeline(steps=[
          ('FE', FeatureAdder()),
          ('log_transfomr', ConditionalLogTransformer(threshold=1.0)),
          ('Preprocess', preprocessor),
          ('ml_model', model)
      ])
      
      cv_scores = cross_validate(pipe, X_train, y_train.values.ravel(), cv=skf, scoring=scoring, n_jobs=-1)
      
      row.append({
          'Model' : name,
          'cv_accuracy' : cv_scores['test_Accuracy'].mean(),
          'cv_precision' : cv_scores['test_Precision'].mean(),
          'cv_recall' : cv_scores['test_Recall'].mean(),
          'cv_f1': cv_scores['test_F1_score'].mean(),
          'cv_roc-auc': cv_scores['test_Roc-auc'].mean(),
          'cv_pr-auc': cv_scores['test_PR-auc'].mean()
  })
     
  # cross-validation scores
  cv_scores = pd.DataFrame(row).sort_values(by='cv_recall', ascending=False)
  
  print(cv_scores.head())
  
  best_model = cv_scores.iloc[0]
  best_model_name = cv_scores.iloc[0]['Model']
  
  print(best_model)

  cv_scores.to_csv('artifacts/performance_metrics.csv')
  
  # 6) Hyperparameter tuning
  # Define parameter distributions per solver to avoid mismatches
  param_distributions = {
    'Log_Reg': {
      'ml_model__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
      'ml_model__l1_ratio': np.arange(0.0, 1.0),
      'ml_model__solver': ['liblinear'],
      'ml_model__penalty': ['l1']
    },
    'Decision Tree': {
      'ml_model__max_depth': [1, 2, 4, 6, 8, 16, 32, 64]
    },
    'Random Forest': {
      'ml_model__max_depth': [1, 2, 4, 6, 8, 16, 32, 64],
      'ml_model__max_leaf_nodes': [15, 31, 63]
    },
    'XGBoost': {
      'ml_model__max_depth': [1, 2, 4, 6, 8, 16, 32, 64],
      'ml_model__learning_rate': np.arange(0, 1)
    },
    'LightGBM': {
      'ml_model__max_depth': [1, 2, 4, 6, 8, 16, 32, 64],
      'ml_model__learning_rate': np.arange(0, 1)
    }
  }
  
  best_pipe = Pipeline(steps=[
      ('FE', FeatureAdder()),
      ('log_transfomr', ConditionalLogTransformer(threshold=1.0)),
      ('Preprocess', preprocessor),
      ('ml_model', models[best_model_name])
  ])
  
  # Example randomized search
  random_search = RandomizedSearchCV(
      estimator=best_pipe,
      param_distributions=param_distributions[best_model_name],
      n_iter=10,
      scoring='recall',
      cv=skf,
      random_state=42,
      n_jobs=-1
  )
  
  random_search.fit(X_train, y_train.values.ravel())
  
  best_params = random_search.best_params_
  best_score = random_search.best_score_
  best_estimator = random_search.best_estimator_
  
  print(best_params, best_score)
  
  # Remove pipeline prefixes
  clean_params = {k.split("__", 1)[-1]: v for k, v in best_params.items()}
  
  # storing the best params in json
  with open("artifacts/best_params.json", "w") as f:
      json.dump(clean_params, f)
  
  # loading the best params from json
  with open("artifacts/best_params.json", "r") as f:
      best_params = json.load(f)
      
  # 7) Final training and evaluation of the log_reg pipeline with best_params
  
  proba = best_estimator.predict_proba(X_test)[:, 1]
  
  roc_auc = roc_auc_score(y_test, proba)
  pr_auc = average_precision_score(y_test, proba)

  threshold = 0.5
  pred = (proba >= threshold).astype(int)
  
  accuracy = accuracy_score(y_test, pred)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)
  f1 = f1_score(y_test, pred)
  
  print('roc_auc score for tuned model', roc_auc)
  print('pr_auc score for tuned model', pr_auc)
  print('accuracy score for tuned model', accuracy)
  print('precision score for tuned model', precision)
  print('recall score for tuned model', recall)
  print('f1_score for tuned model', f1)
  
  schema = {
      "features" : X_train.columns.tolist(),
      'dtypes' : X_train.dtypes.astype(str).to_dict()
  }

  # confusion matrix plot
  fig, ax = plt.subplots()
  sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax)
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Actual")
  ax.set_title("Confusion Matrix")
  
  plt.tight_layout()
  plt.savefig("artifacts/confusion_matrix.png")
  plt.close()
  
  # storing the best params in json
  with open("artifacts/schema.json", "w") as f:
      json.dump(schema, f, indent=4)
    
  joblib.dump(best_estimator, 'models/loan_pred_model_v1.joblib')
  data_hash = hashlib.md5(open('data/Loan_default.csv','rb').read()).hexdigest()

  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  mlflow.set_experiment("Loan_Default_Project")
  mlflow.autolog(log_models=False)
  
  with mlflow.start_run(run_name=best_model_name):

      run_id = run.info.run_id
  
      mlflow.log_params(best_params)
  
      mlflow.log_metrics({
          "cv_mean_recall": best_score,
          "test_accuracy": accuracy,
          "test_recall": recall,
          "test_precision": precision,
          "test_f1": f1,
          "test_roc_auc": roc_auc,
          "test_pr_auc": pr_auc
      })
  
      mlflow.sklearn.log_model(best_estimator, "Log_Reg_model")
  
      mlflow.log_artifact("artifacts/confusion_matrix.png")
  
      mlflow.log_params({
        "dataset_hash", data_hash,
        "decision_threshold", threshold
      })

  model_uri = f"runs:/{run_id}/Log_Reg_model"

  mlflow.register_model(
      model_uri=model_uri,
      name="Loan_Default_Model"
  )
    
  print("Logged to MLflow successfully.")
