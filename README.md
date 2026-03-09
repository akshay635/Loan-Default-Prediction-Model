# 🏦 Credit Risk Assessment System – Loan Default Prediction
# 📌 Project Overview

This project is a Loan Default Risk Estimation application built using Python, Machine Learning, and Streamlit. The goal is to help financial institutions estimate the probability that a borrower may default on a loan.

The dataset was obtained from Kaggle and processed through a complete machine learning workflow including data validation, feature engineering, feature transformation, and model training. Since the dataset contained class imbalance between default and non-default cases, imbalance handling techniques such as class weighting and threshold tuning were applied.

Multiple classification models were evaluated using cross-validation and hyperparameter tuning to select the best-performing pipeline. The final model was deployed as an interactive web application, while experiments, metrics, and artifacts were tracked using MLflow.

Application Modules

1. Single Borrower Prediction
Predicts the probability of loan default for an individual borrower based on financial and credit attributes.

2. Batch Processing
Processes multiple borrowers simultaneously and generates default risk predictions. Includes threshold tuning to analyze recall, false positives, and false negatives.

3. Model Exploration
Displays the top features influencing the model predictions.

4. EMI Calculator
Calculates the Equated Monthly Installment based on loan details.

5. Credit Score Calculator
Estimates a borrower’s credit score using financial and credit information.


# 📊 Dataset Description

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

# ⚙️ Machine Learning Pipeline

The project follows a complete ML workflow:

# 1️⃣ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature scaling

# 2️⃣ Feature Engineering

Transforming categorical features

Preparing dataset for model training

# 3️⃣ Model Training

Multiple models were evaluated:

Logistic Regression

Random Forest

Decision Tree

XGboost


# 4️⃣ Model Evaluation

Models were evaluated using:

Accuracy

Precision

Recall

Confusion Matrix

# 5️⃣ Experiment Tracking

Model experiments are tracked using MLflow.

# 6️⃣ Deployment

The final model is deployed using Streamlit for interactive predictions.

# 🖥️ Streamlit Web Application

The Streamlit application allows users to input applicant information and get real-time loan approval predictions.

Features:

Interactive UI

Real-time prediction

Simple input form for applicant details

# ⚙️ Installation and Setup
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

# 📊 Model Performance
Model	Recall
Logistic Regression	67.97%
Random Forest	65.52%
Decision Tree	67.92%

🔄 CI/CD Pipeline

This project uses GitHub Actions for automation:

Automated model training

Experiment tracking using MLflow

Artifact logging

📸 Application Screenshot
<img width="1908" height="853" alt="image" src="https://github.com/user-attachments/assets/9be50bc0-2027-4443-8be8-6c507343f6b0" />


🚀 Future Improvements

Hyperparameter tuning

Docker deployment

API deployment using FastAPI

Model monitoring

👨‍💻 Author

Akshay Atanure

Data Science & Machine Learning Enthusiast

Transitioning from SAP ABAP to Data Science
