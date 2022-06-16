import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cleaning up the data (dropping columns, setting label, etc) and saving to a new csv
df = pd.read_csv("healthcare-dataset-stroke-data.csv", index_col=False)
df.fillna(df.mean(numeric_only=True), inplace=True)

le_gender = LabelEncoder()
le_ever_married = LabelEncoder()
le_work_type = LabelEncoder()
le_smoking_status = LabelEncoder()

df["gender_n"] = le_gender.fit_transform(df["gender"])
df["ever_married_n"] = le_ever_married.fit_transform(df["ever_married"])
df["work_type_n"] = le_ever_married.fit_transform(df["work_type"])
df["smoking_status_n"] = le_ever_married.fit_transform(df["smoking_status"])

df.drop(["gender", "ever_married", "work_type", "smoking_status", "Residence_type"], axis=1, inplace=True)

df.to_csv("healthcare-dataset-stroke-data-n.csv", sep=',', index=False)