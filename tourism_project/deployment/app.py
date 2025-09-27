import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# Define the model repo ID and the filename of the saved model
HF_MODEL_REPO_ID = "sammysri12/Tourism_Package_Prediction_Model"
MODEL_FILENAME = "best_model.pkl"

# Download the model from Hugging Face Hub
try:
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=MODEL_FILENAME)
    model = joblib.load(model_path)
    st.success("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    st.error(f"Error loading model from Hugging Face Hub: {e}")
    st.stop() # Stop the app if the model can't be loaded

st.title("Wellness Tourism Package Purchase Prediction")

st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Collect input features from the user
age = st.number_input("Age", min_value=18, max_value=100, value=30)
number_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=1)
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=1)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
city_tier = st.selectbox("City Tier", [1, 2, 3])
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Small Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director", "Senior Director", "CEO"])
product_pitched = st.selectbox("Product Pitched", ["Domestic", "International", "Luxury", "Business", "Club"])


# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'Age': [age],
    'NumberOfPersonVisiting': [number_of_person_visiting],
    'PreferredPropertyStar': [preferred_property_star],
    'NumberOfTrips': [number_of_trips],
    'Passport': [passport],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [number_of_children_visiting],
    'MonthlyIncome': [monthly_income],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'NumberOfFollowups': [number_of_followups],
    'DurationOfPitch': [duration_of_pitch],
    'CityTier': [city_tier],
    'TypeofContact': [typeof_contact],
    'Occupation': [occupation],
    'Gender': [gender],
    'MaritalStatus': [marital_status],
    'Designation': [designation],
    'ProductPitched': [product_pitched]
})

# Preprocess the input data using the loaded model's preprocessor
# Assuming the loaded model is a sklearn pipeline with a preprocessor step named 'preprocessor'
try:
    preprocessed_input = model.named_steps['preprocessor'].transform(input_data)
except KeyError:
    st.error("Could not find 'preprocessor' step in the loaded model pipeline. Please check your model training script.")
    preprocessed_input = None # Set to None if preprocessor is not found

# Make prediction
if st.button("Predict") and preprocessed_input is not None:
    prediction = model.named_steps['model'].predict(preprocessed_input)
    prediction_proba = model.named_steps['model'].predict_proba(preprocessed_input)[:, 1] # Probability of the positive class

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"Prediction: Customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.info(f"Prediction: Customer is unlikely to purchase the Wellness Tourism Package.")

    st.write(f"Probability of purchase: {prediction_proba[0]:.4f}")
