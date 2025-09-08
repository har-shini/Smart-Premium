# har.py (Streamlit app)
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_pipeline():
    return joblib.load("ml_pipeline.pkl")

pipeline = load_pipeline()

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("Smart Insurance Premium Predictor")
st.write("Enter customer details below to get the predicted premium amount.")

with st.form("premium_form"):
    st.subheader("Customer Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income", min_value=10000, max_value=10000000, value=500000, step=1000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    health_score = st.slider("Health Score", 0, 100, 70)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    prev_claims = st.number_input("Previous Claims", min_value=0, max_value=50, value=0)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    insurance_duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=30, value=5)
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

    submitted = st.form_submit_button("Predict Premium")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": prev_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise_freq,
        "Property Type": property_type,

        
        "Customer Feedback": "Good",
        "Policy Start Date": "2025-01-01",
        "id": 0
    }])

    try:
        prediction = pipeline.predict(input_data)[0]
        st.success(f"Predicted Premium Amount: **₹{prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


