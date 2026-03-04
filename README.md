# 🏦 Credit Risk Assessment System – Loan Default Prediction
📌 Project Overview

This project predicts whether a loan application will be approved or rejected based on applicant information such as income, credit history, employment status, and loan amount.

The goal of this project is to build an end-to-end machine learning pipeline, including:

Data preprocessing

Feature engineering

Model training and evaluation

Experiment tracking using MLflow

Model deployment using Streamlit

This project demonstrates how machine learning models can assist financial institutions in automating loan approval decisions.

📊 Dataset Description

The dataset contains information about loan applicants.

Feature	Description
Gender	Applicant gender
Married	Marital status
Dependents	Number of dependents
Education	Applicant education level
Self_Employed	Employment status
ApplicantIncome	Applicant income
CoapplicantIncome	Co-applicant income
LoanAmount	Loan amount requested
Loan_Amount_Term	Loan repayment term
Credit_History	Credit history status
Property_Area	Urban / Semiurban / Rural
Loan_Status	Target variable (Approved/Rejected)
⚙️ Machine Learning Pipeline

The project follows a complete ML workflow:

1️⃣ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature scaling

2️⃣ Feature Engineering

Transforming categorical features

Preparing dataset for model training

3️⃣ Model Training

Multiple models were evaluated:

Logistic Regression

Random Forest

Decision Tree

4️⃣ Model Evaluation

Models were evaluated using:

Accuracy

Precision

Recall

Confusion Matrix

5️⃣ Experiment Tracking

Model experiments are tracked using MLflow.

6️⃣ Deployment

The final model is deployed using Streamlit for interactive predictions.

🖥️ Streamlit Web Application

The Streamlit application allows users to input applicant information and get real-time loan approval predictions.

Features:

Interactive UI

Real-time prediction

Simple input form for applicant details

⚙️ Installation and Setup
1️⃣ Clone the repository
git clone <repository-url>
cd loan-prediction-project
2️⃣ Create Virtual Environment
python -m venv venv

Activate environment:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Streamlit App
streamlit run loan_prediction_app.py
📊 Model Performance
Model	Accuracy
Logistic Regression	XX%
Random Forest	XX%
Decision Tree	XX%
📂 Project Structure
loan-prediction-project
│
├── data
│   └── loan_dataset.csv
│
├── notebooks
│   └── EDA.ipynb
│
├── models
│   └── loan_model.pkl
│
├── app
│   └── loan_prediction_app.py
│
├── requirements.txt
├── README.md
└── .github/workflows
🔄 CI/CD Pipeline

This project uses GitHub Actions for automation:

Automated model training

Experiment tracking using MLflow

Artifact logging

📸 Application Screenshot

(Add screenshot of your Streamlit app here)

🚀 Future Improvements

Hyperparameter tuning

Docker deployment

API deployment using FastAPI

Model monitoring

👨‍💻 Author

Akshay Atanure

Data Science & Machine Learning Enthusiast

Transitioning from SAP ABAP to Data Science
