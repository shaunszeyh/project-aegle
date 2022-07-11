import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import hyper

#print(run_neural_network(np.array([1.0,1.0,0.0,0.22985206447339368,0.20799180327868855,0.0,1.0,0.5,1.0,0.0]))) # Should return [0]

st.set_page_config(
    page_title = "Project Aegle",
    page_icon = "👋",
)

st.markdown(
    '''
    # Project Aegle
    Hello there! 👋 This website calculates your risk of stroke and gives you some suggestions to lower it!\n
    **👈 Click on the arrow** on the top left to navigate to the site to calculate your risk.
    ## Stats about strokes
    - In Singapore, stroke is the **fourth** most common cause of death
    - Accounts for more than **10%** of all deaths
    - **26** new stroke cases per day in SG
    - This number will only **increase** with our rising ageing population
    - According to research, early prediction and prevention highly effective in reducing incidence rate
    ## About the data used
    - Our model is generated from the dataset found from the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
    - Data is taken from more than 5000 people from India (We would have taken it from SG if this data was readily available)
    - Datapoints are gender, age, hypertension status, heart disease status, marriage status, work type, residence type, average glucose level, bmi and smoking status
    ### Breakdown of the data
    '''
)

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

male = len(df[df["gender"] == "Male"])
female = len(df[df["gender"] == "Female"])
gender = plt.figure(figsize=(10, 5))
plt.bar(["Male", "Female"], [male, female], color="maroon", width=0.4)
plt.title("Count of People Used in Dataset by Gender")
plt.ylabel("Number of People")
st.write(gender)

ages = df["age"]
for i in range(len(ages)):
    ages[i] = round(ages[i], 0)
n_bins = len(set(ages))
age = plt.figure(figsize=(10, 5))
plt.hist(ages, bins=n_bins, color="maroon")
plt.title("Distribution of Ages of People Used in Dataset")
plt.xlabel("Age")
plt.ylabel("Number of People")
st.write(age)

pos_hypertension = len(df[df["hypertension"] == 1])
neg_hypertension = len(df[df["hypertension"] == 0])
hypertension = plt.figure(figsize=(10, 5))
plt.bar(["Has Hypertension", "No Hypertension"], [pos_hypertension, neg_hypertension], color="maroon", width=0.4)
plt.title("Count of People Used in Dataset by Hypertension Status")
plt.ylabel("Number of People")
st.write(hypertension)

pos_heart_disease = len(df[df["heart_disease"] == 1])
neg_heart_disease = len(df[df["heart_disease"] == 0])
heart_disease = plt.figure(figsize=(10, 5))
plt.bar(["Has Heart Disease", "No Heart Disease"], [pos_heart_disease, neg_heart_disease], color="maroon", width=0.4)
plt.title("Count of People Used in Dataset by Heart Disease Status")
plt.ylabel("Number of People")
st.write(heart_disease)

st.markdown(
    '''
    ## How do we calculate your risk of stroke?
    '''
)