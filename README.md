# Loan-Default-Prediction-Model
🏦 Credit Risk Assessment System – Loan Default Prediction

📌 Overview
Financial institutions face significant losses when borrowers default on loans. Identifying high-risk borrowers early is critical to reduce financial risk while continuing to approve loans for genuine customers.
This project builds a credit risk assessment system that estimates the probability of loan default using applicant financial, credit, and demographic information.
The system supports risk-based lending decisions such as approval, manual review, or stricter loan terms.

🎯 Business Objective
The primary objective is to identify potentially risky borrowers at key decision points in the loan lifecycle so that the bank can:
Reduce losses caused by missed defaulters
Strengthen credit risk controls
Avoid unnecessary rejection of genuine customers
Support regulatory-compliant lending practices
Note:
The model does not approve or reject loans.
It provides a risk score that supports downstream business decisions.

🧠 Problem Framing (Business Perspective)
Loan defaults are rare but costly
Missing a defaulter (False Negative) leads to direct financial loss
Flagging a good customer (False Positive) may only require manual review
Therefore, the project prioritizes catching high-risk borrowers, even at the cost of reviewing more applications.

📊 Dataset Description
The dataset contains historical loan and borrower information, including:
Financial attributes (Income, Loan Amount, DTI Ratio)
Credit behavior (Credit Score, Number of Credit Lines)
Employment stability (Employment Type, Months Employed)
Household indicators (Mortgage, Dependents)
Loan outcome (Default)
Target variable:
Default → Whether the borrower defaulted on the loan

🔍 Exploratory Data Analysis (EDA)
Key findings from EDA:
Severe class imbalance (defaulters are a minority)
Strong non-linear relationships between default risk and:
Credit score
Debt-to-Income ratio
Loan amount vs income
Presence of noisy and extreme values (e.g., unusually high DTI ratios)
Accuracy alone is misleading due to imbalance
EDA was used to guide:
Feature handling
Model choice
Evaluation strategy

🛠️ Modeling Approach
Models Evaluated
Logistic Regression (baseline)
HistGradientBoostingClassifier
XGBoostClassifier
LightGBMClassifier (class-balanced)
Key Design Choices
Used class-balanced learning to address default rarity
Focused evaluation on:
ROC-AUC (ranking ability)
Confusion matrix (business impact)
Prioritized reducing false negatives

🏆 Model Selection
The class-balanced LightGBM model consistently provided:
Strong recall for defaulters
Significant reduction in missed defaulters
Acceptable false-positive trade-off
Stable and interpretable probability outputs
Although multiple models achieved similar ROC-AUC,
balanced LightGBM offered the best business-aligned trade-off.

📈 Evaluation Metrics (Business-Driven)
ROC-AUC → Measures how well borrowers are ranked by risk
Recall (Default class) → Ability to catch risky borrowers
Confusion Matrix → Direct view of financial risk vs operational cost
Accuracy was intentionally not prioritized, as it is misleading in imbalanced credit risk problems.

⚖️ Decision Strategy
The model outputs a default probability, which is converted into risk categories using business thresholds:
Risk Probability
Category
Suggested Action
Low
Low Risk
Auto-approve
Medium
Medium Risk
Manual review
High
High Risk
Reject / stricter terms
Thresholds are adjustable based on business risk tolerance.

🧩 Key Engineering Practices
End-to-end sklearn pipeline
Schema enforcement at inference
Imputers and encoders for production robustness
Probability-based decision logic (not hard labels)

📌 Key Takeaways
Credit risk modeling is a decision support problem, not just classification
Business cost asymmetry must drive evaluation
Simpler, well-aligned models often outperform complex ensembles
Clear separation of:
Risk identification (model)
Loan decision (business policy)

🧠 Skills Demonstrated
Business problem framing
Imbalanced classification handling
Model evaluation beyond accuracy
Risk-based decision thinking
Deployment-ready ML pipelines
Translating ML results into business insights

📎 Future Enhancements
Expected loss modeling (cost-based decisions)
Probability calibration
Model monitoring and drift detection
Behavioral risk scoring for existing borrowers

👤 Author
Akshay Atanure
Aspiring Data Scientist | Credit Risk & Business-Driven ML
📧 akshayatanure11@gmail.com