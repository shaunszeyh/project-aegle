import streamlit as st
import pandas as pd

st.markdown(
'''
    # Calculate Your Risk
    Note: We will not be collecting your data so be rest assured when you key in your information.    
'''
)

conversion = {
    "_gender": {
        "Male": 1.0,
        "Female": 0.0,
    },
    "_hypertension": {
        "Yes": 1.0,
        "No": 0.0,
    },
    "_heart": {
        "Yes": 1.0,
        "No": 0.0,
    },
    "_marriage": {
        "Yes": 1.0,
        "No": 0.0,
    },
    "_work": {
        "Government job": 0.0,
        "Private": 0.5,
        "Self-employed": 0.75,
        "Child": 1.0
    },
    "_residence": {
        "Rural": 0.0,
        "Urban": 1.0,
    },
    "_smoking": {

    },
}

df = pd.read_csv("healthcare-dataset-stroke-data.csv", index_col=False)

def convert_numeric(val, type):
    return min(val / max(df[type]), 1.0)

with st.form("my_form"):
    gender = conversion["_gender"][st.radio("Select your gender", ["Male", "Female"])]
    age = convert_numeric(st.number_input("Enter your age", min_value=0, step=1), "age")
    hypertension = conversion["_hypertension"][st.radio("Do you have hypertension?", ["Yes", "No"])]
    heart = conversion["_heart"][st.radio("Do you have heart disease?", ["Yes", "No"])]
    marriage = conversion["_marriage"][st.radio("Have you been married?", ["Yes", "No"])]
    work = conversion["_work"][st.radio("What kind of job do you have?", ["Government job", "Private", "Self-employed", "Child"])]
    residence = conversion["_residence"][st.radio("What is your residence type?", ["Rural", "Urban"])]
    glucose = convert_numeric(st.number_input("What is your average glucose level? (in mmol / L)", min_value=0), "avg_glucose_level")
    bmi = convert_numeric(st.number_input("What is your BMI (Body Mass Index)?", min_value=0.0, step=0.1), "bmi")
    smoking = st.radio("What is your smoking status?", ["Fomerly smoked", "Never smoked", "Smokes"])
    
    submitted = st.form_submit_button("Submit My Data")
    
if submitted:
    st.write(gender, age, hypertension, heart, marriage, work, residence, glucose, bmi, smoking)
