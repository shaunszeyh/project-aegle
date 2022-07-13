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
    ### Analysis and breakdown of the data
    From here we can see which are the factors which are more likely to cause a stroke
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

fig, axs = plt.subplots(1, 3, figsize=(12, 9))

axs[0].hist(age_neg, stacked=True, color="#1984c5", ec="black", label="No stroke", bins=10)
axs[0].hist(age_pos, stacked=True, color="#d5a036", ec="black", label="Stroke", bins=10)
axs[0].set_title("Stroke Status by Age")
axs[0].set_ylabel("Number of patients")
axs[0].legend()

axs[1].hist(glucose_neg, stacked=True, color="#1984c5", ec="black", label="No stroke", bins=10)
axs[1].hist(glucose_pos, stacked=True, color="#d5a036", ec="black", label="Stroke", bins=10)
axs[1].set_title("Stroke Status by Average Glucose Level")
axs[1].set_ylabel("Number of patients")
axs[1].legend()

axs[2].hist(bmi_neg, stacked=True, color="#1984c5", ec="black", label="No stroke", bins=10)
axs[2].hist(bmi_pos, stacked=True, color="#d5a036", ec="black", label="Stroke", bins=10)
axs[2].set_title("Stroke Status by BMI")
axs[2].set_ylabel("Number of patients")
axs[2].legend()

fig.tight_layout()

st.write(fig, "\n")

### End of Graph 1, Start of Graph 2 ###
# Graph for the boxplot of glucose level and bmi

age_pos_median = np.median(age_pos)
age_neg_median = np.median(age_neg)
glucose_pos_median = np.median(glucose_pos)
glucose_neg_median = np.median(glucose_neg)
bmi_pos_median = np.median(bmi_pos)
bmi_neg_median = np.median(bmi_neg)

fig2, axs2 = plt.subplots(1, 3, figsize=(12, 9))

axs2[0].bar(["Stroke", "No Stroke"], [age_pos_median, age_neg_median], color=["#d5a036", "#1984c5"], ec="black")
axs2[0].set_title("Median Age of Patients by Stroke Status")
axs2[0].set_ylabel("Median Age / years")
axs2[0].text(0, age_pos_median + 1, int(age_pos_median), ha="center")
axs2[0].text(1, age_neg_median + 1, int(age_neg_median), ha="center")

axs2[1].bar(["Stroke", "No Stroke"], [glucose_pos_median, glucose_neg_median], color=["#d5a036", "#1984c5"], ec="black")
axs2[1].set_title("Median Glucose Level of Patients by Stroke Status")
axs2[1].set_ylabel("Median Age / years")
axs2[1].text(0, glucose_pos_median + 1, int(glucose_pos_median), ha="center")
axs2[1].text(1, glucose_neg_median + 1, int(glucose_neg_median), ha="center")

axs2[2].bar(["Stroke", "No Stroke"], [bmi_pos_median, bmi_neg_median], color=["#d5a036", "#1984c5"], ec="black")
axs2[2].set_title("Median BMI of Patients by Stroke Status")
axs2[2].set_ylabel("Median Age / years")
axs2[2].text(0, bmi_pos_median + 0.25, int(bmi_pos_median), ha="center")
axs2[2].text(1, bmi_neg_median + 0.25, int(bmi_neg_median), ha="center")

fig2.tight_layout()

st.write(fig2, "\n")

### End of Graph 2 ###

def create_graph(fields, title, pos_percentage, neg_percentage):
    x = np.arange(len(fields))
    width = 0.4

    fig3, axs3 = plt.subplots()

    rects1 = axs3.bar(x - width/2, neg_percentage, width, label="No Stroke", color="#1984c5", ec="black")
    rects2 = axs3.bar(x + width/2, pos_percentage, width, label="Stroke", color="#d5a036", ec="black")
    axs3.set_title(title)
    axs3.set_ylabel("Percentage of patients / %")
    axs3.set_xticks([])

    for i in range(len(fields)):
        axs3.text(i, -5, fields[i], ha="center")

    axs3.bar_label(rects1, padding=1, fmt="%.1f")
    axs3.bar_label(rects2, padding=1, fmt="%.1f")
    axs3.legend()
    fig3.tight_layout()

    st.write(fig3, "\n")

### Start of Graph 3 ###

x = np.arange(2)
width = 0.4

labels = ["Male", "Female"]
male_pos = len(df[(df["gender"] == "Male") & (df["stroke"] == 1)])
female_pos = len(df[(df["gender"] == "Female") & (df["stroke"] == 1)])
male_neg = len(df[(df["gender"] == "Male") & (df["stroke"] == 0)])
female_neg = len(df[(df["gender"] == "Female") & (df["stroke"] == 0)])
male_pos_percentage = (male_pos / (male_pos + male_neg)) * 100
male_neg_percentage = (male_neg / (male_pos + male_neg)) * 100
female_pos_percentage = (female_pos / (female_pos + female_neg)) * 100
female_neg_percentage = (female_neg / (female_pos + female_neg)) * 100
pos_percentage = [male_pos_percentage, female_pos_percentage]
neg_percentage = [male_neg_percentage, female_neg_percentage]

create_graph(labels, "Stroke Status by Gender", pos_percentage, neg_percentage)

### End of Graph 3, Start of Graph 4 ###

labels = ["Hypertension", "No Hypertension"]
hypertension_pos = len(df[(df["hypertension"] == 1) & (df["stroke"] == 1)])
no_hypertension_pos = len(df[(df["hypertension"] == 0) & (df["stroke"] == 1)])
hypertension_neg = len(df[(df["hypertension"] == 1) & (df["stroke"] == 0)])
no_hypertension_neg = len(df[(df["hypertension"] == 0) & (df["stroke"] == 0)])
hypertension_pos_percentage = (hypertension_pos / (hypertension_pos + hypertension_neg)) * 100
hypertension_neg_percentage = (hypertension_neg / (hypertension_pos + hypertension_neg)) * 100
no_hypertension_pos_percentage = (no_hypertension_pos / (no_hypertension_pos + no_hypertension_neg)) * 100
no_hypertension_neg_percentage = (no_hypertension_neg / (no_hypertension_pos + no_hypertension_neg)) * 100
pos_percentage = [hypertension_pos_percentage, no_hypertension_pos_percentage]
neg_percentage = [hypertension_neg_percentage, no_hypertension_neg_percentage]

create_graph(labels, "Stroke Status by Hypertension", pos_percentage, neg_percentage)

### End of Graph 4, Start of Graph 5 ###

labels = ["Heart Disease", "No Heart Disease"]
heart_disease_pos = len(df[(df["heart_disease"] == 1) & (df["stroke"] == 1)])
no_heart_disease_pos = len(df[(df["heart_disease"] == 0) & (df["stroke"] == 1)])
heart_disease_neg = len(df[(df["heart_disease"] == 1) & (df["stroke"] == 0)])
no_heart_disease_neg = len(df[(df["heart_disease"] == 0) & (df["stroke"] == 0)])
heart_disease_pos_percentage = (heart_disease_pos / (heart_disease_pos + heart_disease_neg)) * 100
heart_disease_neg_percentage = (heart_disease_neg / (heart_disease_pos + heart_disease_neg)) * 100
no_heart_disease_pos_percentage = (no_heart_disease_pos / (no_heart_disease_pos + no_heart_disease_neg)) * 100
no_heart_disease_neg_percentage = (no_heart_disease_neg / (no_heart_disease_pos + no_heart_disease_neg)) * 100
pos_percentage = [heart_disease_pos_percentage, no_heart_disease_pos_percentage]
neg_percentage = [heart_disease_neg_percentage, no_heart_disease_neg_percentage]

create_graph(labels, "Stroke Status by Heart Disease", pos_percentage, neg_percentage)

### End of Graph 5, Start of Graph 6 ###

labels = ["Ever Married", "Never Married"]
ever_married_pos = len(df[(df["ever_married"] == "Yes") & (df["stroke"] == 1)])
no_ever_married_pos = len(df[(df["ever_married"] == "No") & (df["stroke"] == 1)])
ever_married_neg = len(df[(df["ever_married"] == "Yes") & (df["stroke"] == 0)])
no_ever_married_neg = len(df[(df["ever_married"] == "No") & (df["stroke"] == 0)])
ever_married_pos_percentage = (ever_married_pos / (ever_married_pos + ever_married_neg)) * 100
ever_married_neg_percentage = (ever_married_neg / (ever_married_pos + ever_married_neg)) * 100
no_ever_married_pos_percentage = (no_ever_married_pos / (no_ever_married_pos + no_ever_married_neg)) * 100
no_ever_married_neg_percentage = (no_ever_married_neg / (no_ever_married_pos + no_ever_married_neg)) * 100
pos_percentage = [ever_married_pos_percentage, no_ever_married_pos_percentage]
neg_percentage = [ever_married_neg_percentage, no_ever_married_neg_percentage]

create_graph(labels, "Stroke Status by Marriage Status", pos_percentage, neg_percentage)

### End of Graph 6, Start of Graph 7 ###

labels = ["Rural", "Urban"]
rural_pos = len(df[(df["Residence_type"] == "Rural") & (df["stroke"] == 1)])
urban_pos = len(df[(df["Residence_type"] == "Urban") & (df["stroke"] == 1)])
rural_neg = len(df[(df["Residence_type"] == "Rural") & (df["stroke"] == 0)])
urban_neg = len(df[(df["Residence_type"] == "Rural") & (df["stroke"] == 0)])
rural_pos_percentage = (rural_pos / (rural_pos + rural_neg)) * 100
rural_neg_percentage = (rural_neg / (rural_pos + rural_neg)) * 100
urban_pos_percentage = (urban_pos / (urban_pos + urban_neg)) * 100
urban_neg_percentage = (urban_neg / (urban_pos + urban_neg)) * 100
pos_percentage = [rural_pos_percentage, urban_pos_percentage]
neg_percentage = [rural_neg_percentage, urban_neg_percentage]

create_graph(labels, "Stroke Status by Residence Type", pos_percentage, neg_percentage)

### End of Graph 7, Start of Graph 8 ###

labels = ["Government Job", "Private", "Self-employed", "Child"]
govt_pos = len(df[(df["work_type"] == "Govt_job") & (df["stroke"] == 1)])
private_pos = len(df[(df["work_type"] == "Private") & (df["stroke"] == 1)])
self_pos = len(df[(df["work_type"] == "Self-employed") & (df["stroke"] == 1)])
child_pos = len(df[(df["work_type"] == "children") & (df["stroke"] == 1)])
govt_neg = len(df[(df["work_type"] == "Govt_job") & (df["stroke"] == 0)])
private_neg = len(df[(df["work_type"] == "Private") & (df["stroke"] == 0)])
self_neg = len(df[(df["work_type"] == "Self-employed") & (df["stroke"] == 0)])
child_neg = len(df[(df["work_type"] == "children") & (df["stroke"] == 0)])
govt_pos_percentage = (govt_pos / (govt_pos + govt_neg)) * 100
govt_neg_percentage = (govt_neg / (govt_pos + govt_neg)) * 100
private_neg_percentage = (private_neg / (private_pos + private_neg)) * 100
private_pos_percentage = (private_pos / (private_pos + private_neg)) * 100
self_pos_percentage = (self_pos / (self_pos + self_neg)) * 100
self_neg_percentage = (self_neg / (self_pos + self_neg)) * 100
child_neg_percentage = (child_neg / (child_pos + child_neg)) * 100
child_pos_percentage = (child_pos / (child_pos + child_neg)) * 100
pos_percentage = [govt_pos_percentage, private_pos_percentage, self_pos_percentage, child_pos_percentage]
neg_percentage = [govt_neg_percentage, private_neg_percentage, self_neg_percentage, child_neg_percentage]

create_graph(labels, "Smoke Status by Work Type", pos_percentage, neg_percentage)

### End of Graph 8, Start of Graph 9 ###

labels = ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"]
unknown_pos = len(df[(df["smoking_status"] == "Unknown") & (df["stroke"] == 1)])
former_pos = len(df[(df["smoking_status"] == "formerly smoked") & (df["stroke"] == 1)])
never_pos = len(df[(df["smoking_status"] == "never smoked") & (df["stroke"] == 1)])
smokes_pos = len(df[(df["smoking_status"] == "smokes") & (df["stroke"] == 1)])
unknown_neg = len(df[(df["smoking_status"] == "Unknown") & (df["stroke"] == 0)])
former_neg = len(df[(df["smoking_status"] == "formerly smoked") & (df["stroke"] == 0)])
never_neg = len(df[(df["smoking_status"] == "never smoked") & (df["stroke"] == 0)])
smokes_neg = len(df[(df["smoking_status"] == "smokes") & (df["stroke"] == 0)])
unknown_pos_percentage = (unknown_pos / (unknown_pos + unknown_neg)) * 100
unknown_neg_percentage = (unknown_neg / (unknown_pos + unknown_neg)) * 100
former_neg_percentage = (former_neg / (former_pos + former_neg)) * 100
former_pos_percentage = (former_pos / (former_pos + former_neg)) * 100
never_pos_percentage = (never_pos / (never_pos + never_neg)) * 100
never_neg_percentage = (never_neg / (never_pos + never_neg)) * 100
smokes_neg_percentage = (smokes_neg / (smokes_pos + smokes_neg)) * 100
smokes_pos_percentage = (smokes_pos / (smokes_pos + smokes_neg)) * 100
pos_percentage = [unknown_pos_percentage, former_pos_percentage, never_pos_percentage, smokes_pos_percentage]
neg_percentage = [unknown_neg_percentage, former_neg_percentage, never_neg_percentage, smokes_neg_percentage]

create_graph(labels, "Smoke Status by Smoking Status", pos_percentage, neg_percentage)

### End of Graph 9 ###

# Graph for the distribution of all the factors

st.markdown(
    '''
    ## How do we calculate your risk of stroke?
    '''
)


