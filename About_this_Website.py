import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    - Data is taken from more than 5000 patients from India (We would have taken it from SG if this data was made readily available here)
    - Datapoints are gender, age, hypertension status, heart disease status, marriage status, work type, residence type, average glucose level, bmi and smoking status
    ### Breakdown of the data
    '''
)

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

### Start of Graph 1 ###

df.fillna(df.mean(numeric_only=True), inplace=True)
df["age"].round(decimals=0)
n_bins = 10
age_pos = df[df["stroke"] == 1]["age"]
age_neg = df[df["stroke"] == 0]["age"]
glucose_pos = df[df["stroke"] == 1]["avg_glucose_level"]
glucose_neg = df[df["stroke"] == 0]["avg_glucose_level"]
bmi_pos = df[df["stroke"] == 1]["bmi"]
bmi_neg = df[df["stroke"] == 0]["bmi"]

fig, axs = plt.subplots(1, 3, figsize=(16, 9))

axs[0].hist(age_neg, stacked=True, color="#1984c5", ec="black", label="No stroke")
axs[0].hist(age_pos, stacked=True, color="#d5a036", ec="black", label="Stroke")
axs[0].set_title("Stroke Status by Age")
axs[0].set_ylabel("Number of patients")
axs[0].legend()

axs[1].hist(glucose_neg, stacked=True, color="#1984c5", ec="black", label="No stroke")
axs[1].hist(glucose_pos, stacked=True, color="#d5a036", ec="black", label="Stroke")
axs[1].set_title("Stroke Status by Average Glucose Level")
axs[1].set_ylabel("Number of patients")
axs[1].legend()

axs[2].hist(bmi_neg, stacked=True, color="#1984c5", ec="black", label="No stroke")
axs[2].hist(bmi_pos, stacked=True, color="#d5a036", ec="black", label="Stroke")
axs[2].set_title("Stroke Status by BMI")
axs[2].set_ylabel("Number of patients")
axs[2].legend()

st.write(fig, "\n")

### End of Graph 1, Start of Graph 2 ###

age_pos_median = np.median(age_pos)
age_neg_median = np.median(age_neg)
glucose_pos_median = np.median(glucose_pos)
glucose_neg_median = np.median(glucose_neg)
bmi_pos_median = np.median(bmi_pos)
bmi_neg_median = np.median(bmi_neg)

fig2, axs2 = plt.subplots(1, 3, figsize=(16, 9))

axs2[0].bar(["Stroke", "No Stroke"], [age_pos_median, age_neg_median], color=["#d5a036", "#1984c5"], ec="black")
axs2[0].set_title("Median Age of Patients by Stroke Status")
axs2[0].set_ylabel("Median Age (Years)")
axs2[0].text(0, age_pos_median + 1, int(age_pos_median), ha="center")
axs2[0].text(1, age_neg_median + 1, int(age_neg_median), ha="center")

axs2[1].bar(["Stroke", "No Stroke"], [glucose_pos_median, glucose_neg_median], color=["#d5a036", "#1984c5"], ec="black")
axs2[1].set_title("Median Glucose Level of Patients by Stroke Status")
axs2[1].set_ylabel("Median Age (Years)")
axs2[1].text(0, glucose_pos_median + 1, int(glucose_pos_median), ha="center")
axs2[1].text(1, glucose_neg_median + 1, int(glucose_neg_median), ha="center")

axs2[2].bar(["Stroke", "No Stroke"], [bmi_pos_median, bmi_neg_median], color=["#d5a036", "#1984c5"], ec="black")
axs2[2].set_title("Median BMI of Patients by Stroke Status")
axs2[2].set_ylabel("Median Age (Years)")
axs2[2].text(0, bmi_pos_median + 0.25, int(bmi_pos_median), ha="center")
axs2[2].text(1, bmi_neg_median + 0.25, int(bmi_neg_median), ha="center")

st.write(fig2, "\n")

### End of Graph 2, Start of Graph 3 ###

male_pos = len(df[(df["gender"] == "Male") & (df["stroke"] == 1)])
female_pos = len(df[(df["gender"] == "Female") & (df["stroke"] == 1)])
male_neg = len(df[(df["gender"] == "Male") & (df["stroke"] == 0)])
female_neg = len(df[(df["gender"] == "Female") & (df["stroke"] == 0)])
male_pos_percentage = (male_pos / (male_pos + male_neg)) * 100
male_neg_percentage = (male_neg / (male_pos + male_neg)) * 100
female_pos_percentage = (female_pos / (male_pos + male_neg)) * 100
female_neg_percentage = (female_neg / (male_pos + male_neg)) * 100

fig3, axs3 = plt.subplots(1, 3, figsize=(16, 9))

st.write(fig3)

st.markdown(
    '''
    ## How do we calculate your risk of stroke?
    '''
)