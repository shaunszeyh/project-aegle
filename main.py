import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import metrics

# Cleaning up the data (dropping columns, setting label, etc)
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.fillna(df.mean(), inplace=True)
le_gender = LabelEncoder()
le_ever_married = LabelEncoder()
le_work_type = LabelEncoder()
le_smoking_status = LabelEncoder()
df["gender_n"] = le_gender.fit_transform(df["gender"])
df["ever_married_n"] = le_ever_married.fit_transform(df["ever_married"])
df["work_type_n"] = le_ever_married.fit_transform(df["work_type"])
df["smoking_status_n"] = le_ever_married.fit_transform(df["smoking_status"])
target = df["stroke"]
df_n = df.drop(["gender", "ever_married", "work_type", "smoking_status", "Residence_type", "stroke", "id"], axis="columns")

# Trying with decision tree
model = tree.DecisionTreeClassifier()
model.fit(df_n.values, target.values)
prediction = model.predict([[44.0, 0, 0, 85.28, 26.2, 0, 1, 0, 0]])