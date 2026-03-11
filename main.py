from training.train import main
import mlflow
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Default_Project")

if __name__ == "__main__":
    main()
