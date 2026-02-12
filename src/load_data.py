import streamlit as st
# --------------------------------------------------
# Input Sections
# --------------------------------------------------
def load_data():
    st.sidebar.header("👤 Applicant Profile")
    
    # Loan ID
    loan_id = st.sidebar.text_input('Enter your Loan ID')
    
    # Name of the applicant/customer
    name = st.sidebar.text_input('Enter your full name:')
    
    # Age
    age = st.sidebar.slider("Age", 18, 70, 35)
    
    # Education level
    education = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Post Graduate"])
    
    # Employment status
    employment = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
    
    # Marital status
    marital_status = st.sidebar.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )
    
    months_employed, income, monthly_income, emi = 0, 0, 0, 0
    
    st.sidebar.header("💰 Financial Information & Credit history")
    
    # Applying conditions based on employment status
    if employment == "Salaried" or employment == 'Self-employed':
        # Number of Months being employed till now
        months_employed = st.sidebar.slider('Months employed', 0, 480, 60)
        # Annual Income
        income = st.sidebar.text_input("Annual Income", "10,000")
        # type conversion from string to int
        income = int(income.replace(',', ''))
        # Monthly Income
        monthly_income = round(income//12, 2)
    
    # Debt-To-Income ratio
    dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
    # Credit Score
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 720)
    # Number of Credit lines
    num_credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 15, 4)
    
    st.sidebar.header("📊 Household Details")
    
    has_mortgage = st.sidebar.selectbox("Has Mortgage?", ["Yes", "No"])
    has_dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
    cosigner = st.sidebar.selectbox("Has Co-Signer?", ["Yes", "No"])
    
    st.sidebar.header("📌 Loan Details")
    
    # Name of the Bank
    name_bank = st.sidebar.text_input(placeholder="Please enter bank name", label='Bank Name')
    
    # Type of Loan
    loan_purpose = st.sidebar.selectbox(
        "Purpose of Loan",
        ["Home", "Education", "Personal", "Auto", "Business"]
    )
    # Loan amount required/taken from the bank
    loan_amount = st.sidebar.text_input("Loan Amount", "1,00,000")
    loan_amount = int(loan_amount.replace(',', ''))
    # Interest Rate
    interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.5)
    # Loan Tenure
    loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    
    # Equated Monthly Installments(EMI)
    monthly_rate = interest_rate / (12 * 100)
    
    # EMI formula
    emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term) / \
          ((1 + monthly_rate) ** loan_term - 1)
    
    # Input data which will be passed to the model
    user_data = {
            "Age": age, "LoanAmount": loan_amount,
            "CreditScore": credit_score, "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
            "LoanTerm": loan_term, "DTIRatio": dti,
            "Education": education, "EmploymentType": employment,
            "MaritalStatus": marital_status, "HasMortgage": has_mortgage,
            "HasDependents": has_dependents, "LoanPurpose": loan_purpose,
            "HasCoSigner": cosigner, "Monthly_Income": monthly_income,
            "EMI": emi
    }
    return user_data
