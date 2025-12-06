import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained SVM model
loaded_model = joblib.load('trained_svm_model.joblib')

st.title('Loan Prediction System')

st.write('Please enter the applicant\'s details:')

# Input fields for features
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=1000)
loan_amount = st.number_input('Loan Amount (in thousands)', min_value=1, value=150)
loan_amount_term = st.selectbox('Loan Amount Term (in months)', [12, 36, 60, 120, 180, 240, 300, 360, 480], index=7)
credit_history = st.selectbox('Credit History (1: Yes, 0: No)', [0, 1])
property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])

# Button to make prediction
if st.button('Predict Loan Status'):
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

# Display result
if prediction[0] == 1:
    st.success('Loan Status: Approved')
else:
    st.error('Loan Status: Rejected')