import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained SVM model
loaded_model = joblib.load('trained_svm_model.joblib')

st.set_page_config(page_title="Loan Prediction App", layout="centered")

st.title('🏠 Loan Application Status Prediction')
st.markdown("### Predict if a loan application will be approved or rejected based on the applicant's details.")

st.write('---')
st.header('Applicant Details')

# Personal Information
st.container()
with st.expander("Personal Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'], help='Select the applicant\'s gender.')
    with col2:
        married = st.selectbox('Married', ['Yes', 'No'], help='Indicate if the applicant is married.')

    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'], help='Number of dependents the applicant has.')
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'], help='Applicant\'s education level.')
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'], help='Is the applicant self-employed?')

# Employment and Financial Details
st.container()
with st.expander("Employment and Financial Details", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        applicant_income = st.number_input('Applicant Income (USD)', min_value=0, value=5000, help='Monthly income of the applicant.')
    with col4:
        coapplicant_income = st.number_input('Coapplicant Income (USD)', min_value=0, value=1000, help='Monthly income of the coapplicant, if any.')

# Loan Request Details
st.container()
with st.expander("Loan Request Details", expanded=True):
    loan_amount = st.number_input('Loan Amount (in thousands USD)', min_value=1, value=150, help='Desired loan amount in thousands.')
    loan_amount_term = st.selectbox('Loan Amount Term (in months)', [12, 36, 60, 120, 180, 240, 300, 360, 480], index=7, help='Term of the loan in months.')

# Credit and Property Information
st.container()
with st.expander("Credit and Property Information", expanded=True):
    credit_history = st.selectbox('Credit History (1: Met guidelines, 0: Not met guidelines)', [0, 1], help='Applicant\'s credit history. 1 indicates a good credit history, 0 indicates a poor one.')
    property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'], help='Location of the property.')

st.write('---')

# Button to make prediction
if st.button('Predict Loan Status'):
    # Basic input validation
    if applicant_income <= 0:
        st.error('Applicant Income must be a positive number.')
        st.stop()
    if coapplicant_income < 0:
        st.error('Coapplicant Income cannot be negative.')
        st.stop()
    if loan_amount <= 0:
        st.error('Loan Amount must be a positive number.')
        st.stop()

    # Preprocess user input to match model's training format
    # Convert categorical features to numerical
    gender_encoded = 1 if gender == 'Male' else 0
    married_encoded = 1 if married == 'Yes' else 0
    education_encoded = 1 if education == 'Graduate' else 0
    self_employed_encoded = 1 if self_employed == 'Yes' else 0

    dependents_encoded = dependents.replace('3+', '4')  # Handle '3+'

    property_area_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    property_area_encoded = property_area_mapping[property_area]

    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([[gender_encoded, married_encoded, int(dependents_encoded),
                                education_encoded, self_employed_encoded,
                                applicant_income, coapplicant_income,
                                loan_amount, loan_amount_term,
                                credit_history, property_area_encoded]],
                              columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                                       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                       'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Display result with enhanced visuals and context
    st.markdown("### Prediction Result:")
    if prediction[0] == 1:
        st.success('## ✅ Loan Status: **Approved!**')
        st.markdown("Congratulations! Based on the provided details, your loan application is highly likely to be approved. This means you meet the key criteria evaluated by our model.")
    else:
        st.error('## ❌ Loan Status: **Rejected!**')
        st.markdown("Unfortunately, based on the provided details, your loan application is likely to be rejected. Please review the input values, especially credit history and income, or contact your financial advisor for more information.")
    st.write('---')
    st.info("Note: This prediction is based on a trained machine learning model and should be used as a guide only and does not guarantee actual loan approval or rejection.")
