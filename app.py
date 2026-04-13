from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "trained_svm_model.joblib")
loaded_model = joblib.load(model_path)

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    data: dict with raw user inputs (same semantics as your Streamlit app).
    Returns a one-row DataFrame with the exact columns your model expects.
    """

    gender = data["gender"]                
    married = data["married"]              
    dependents = data["dependents"]        
    education = data["education"]          
    self_employed = data["self_employed"]  

    applicant_income = float(data["applicant_income"])
    coapplicant_income = float(data["coapplicant_income"])
    loan_amount = float(data["loan_amount"])
    loan_amount_term = int(data["loan_amount_term"])
    credit_history = int(data["credit_history"])  
    property_area = data["property_area"]         


    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Yes" else 0
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    dependents_encoded = dependents.replace("3+", "4")

    property_area_mapping = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_area_encoded = property_area_mapping[property_area]

    
    input_df = pd.DataFrame(
        [[
            gender_encoded,
            married_encoded,
            int(dependents_encoded),
            education_encoded,
            self_employed_encoded,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area_encoded,
        ]],
        columns=[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ],
    )

    return input_df

@app.route("/")
def index():
    
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

       
        if float(data["applicant_income"]) <= 0:
            return jsonify({"error": "Applicant Income must be positive."}), 400
        if float(data["coapplicant_income"]) < 0:
            return jsonify({"error": "Coapplicant Income cannot be negative."}), 400
        if float(data["loan_amount"]) <= 0:
            return jsonify({"error": "Loan Amount must be positive."}), 400

        input_df = preprocess_input(data)
        prediction = loaded_model.predict(input_df)[0]

        if prediction == 1:
            status = "Approved"
            message = (
                "Congratulations! Based on the provided details, your loan application "
                "is highly likely to be approved."
            )
        else:
            status = "Rejected"
            message = (
                "Unfortunately, based on the provided details, your loan application "
                "is likely to be rejected. Please review your credit history, income, etc."
            )

        return jsonify(
            {
                "prediction": int(prediction),
                "status": status,
                "message": message,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
