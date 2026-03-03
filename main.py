from training.train import main

import mlflow
import os

# Ensure directories exist
os.makedirs("outputs/mlflow/mlruns", exist_ok=True)

# Point MLflow to SQLite backend
mlflow.set_tracking_uri("sqlite:///outputs/mlflow/mlflow.db")

# Optional: set experiment name
mlflow.set_experiment("default")

if __name__ == "__main__":
    main()
