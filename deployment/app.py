# ---------------------------------------------------------------------------
# Streamlit App: Tourism Package Purchase Prediction
# Loads the trained pipeline from Hugging Face Model Hub, collects customer
# inputs through an interactive form, and displays the purchase prediction.
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load the trained pipeline from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="NiiOsa1/tourism-package-model",
    filename="best_model.joblib"
)
model = joblib.load(model_path)

# App header
st.title("Tourism Package Purchase Prediction")
st.write("Enter customer details below to predict whether they will "
         "purchase the Wellness Tourism Package.")

# Collect customer inputs via interactive widgets
Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact",
                             ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (minutes)", 0, 130, 15)
Occupation = st.selectbox("Occupation",
                          ["Salaried", "Small Business",
                           "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 6, 3)
ProductPitched = st.selectbox("Product Pitched",
                              ["Basic", "Standard", "Deluxe",
                               "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status",
                             ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips per Year", 1, 22, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 3, 1)
Designation = st.selectbox("Designation",
                           ["Executive", "Manager", "Senior Manager",
                            "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0,
                                max_value=100000.0, value=23000.0)

# Build input DataFrame matching the training feature order and format
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# Classification threshold (same as used during evaluation)
classification_threshold = 0.45

# Predict on button click
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    if pred == 1:
        st.success(f"This customer is likely to purchase. "
                   f"(Probability: {prob:.1%})")
    else:
        st.warning(f"This customer is unlikely to purchase. "
                   f"(Probability: {prob:.1%})")
