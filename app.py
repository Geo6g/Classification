import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("🎓 Student Pass/Fail Predictor")

# --- Inputs ---

math = st.number_input("Math Score", min_value=0, max_value=100, value=80)
reading = st.number_input("Reading Score", min_value=0, max_value=100, value=85)
writing = st.number_input("Writing Score", min_value=0, max_value=100, value=90)

gender = st.selectbox("Gender", ["female", "male"])
ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Education", [
    "associate's degree", "bachelor's degree", "some college", 
    "high school", "master's degree", "some high school"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
prep = st.selectbox("Test Prep Course", ["none", "completed"])

average = (math + reading + writing) / 3

# --- Create DataFrame in required order ---
input_dict = {
    "math score": [math],
    "reading score": [reading],
    "writing score": [writing],
    "gender": [gender],
    "race/ethnicity": [ethnicity],
    "parental level of education": [parent_edu],
    "lunch": [lunch],
    "test preparation course": [prep],
    "average": [average]
}
input_df = pd.DataFrame(input_dict)

# Encode categorical fields
for col in input_df.select_dtypes(include="object").columns:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure column order matches model
input_df = input_df[model.feature_names_in_]

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Pass" if prediction == 1 else "Fail"
    st.subheader(result)
