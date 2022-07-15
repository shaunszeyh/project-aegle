import streamlit as st
import pandas as pd
from classes import *
import numpy as np

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
        "Formerly smoked": 0.25,
        "Never smoked": 0.5,
        "Smokes": 1.0,
    },
}

df = pd.read_csv("healthcare-dataset-stroke-data.csv", index_col=False)

def convert_numeric(val, type):
    return min(val / max(df[type]), 1.0)

def run_neural_network(inputs): 
    # Inputs are given in order
    # age, hypertension, heart, glucose, bmi, gender, marriage, work, residence, smoking
    weights1 = np.load("parameters/weights1.npy")
    biases1 = np.load("parameters/biases1.npy")
    weights2 = np.load("parameters/weights2.npy")
    biases2 = np.load("parameters/biases2.npy")
    dense1 = Layer_Dense(10, 64)
    dense2 = Layer_Dense(64, 2)
    dense1.weights = weights1
    dense1.biases = biases1
    dense2.weights = weights2
    dense2.biases = biases2
    activation = Activation_ReLU()
    softmax = Activation_Softmax()
    dense1.forward(inputs)
    activation.forward(dense1.output)
    dense2.forward(activation.output)
    softmax.forward(dense2.output)
    predictions = np.argmax(softmax.output, axis=1)

    if predictions:
        predictions = "Stroke"
    else:
        predictions = "No stroke"

    return predictions, np.max(softmax.output)

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
    smoking = conversion["_smoking"][st.radio("What is your smoking status?", ["Formerly smoked", "Never smoked", "Smokes"])]
    
    submitted = st.form_submit_button("Submit My Data")
    
if submitted:
    st.write(gender, age, hypertension, heart, marriage, work, residence, glucose, bmi, smoking)

    stroke, confidence = run_neural_network(np.array([age, hypertension, heart, glucose, bmi, gender, marriage, work, residence, smoking]))

    st.write(stroke)
    st.write("Confidence:", str(round(confidence * 100, 1)) + "%")