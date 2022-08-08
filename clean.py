import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Cleaning up the data (dropping columns, setting label, etc)
df = pd.read_csv("healthcare-dataset-stroke-data.csv", index_col=False)
df.fillna(df.mean(numeric_only=True), inplace=True)

le_gender = LabelEncoder()
le_ever_married = LabelEncoder()
le_work_type = LabelEncoder()
le_residence_type = LabelEncoder()
le_smoking_status = LabelEncoder()

df["gender_n"] = le_gender.fit_transform(df["gender"])
df["ever_married_n"] = le_ever_married.fit_transform(df["ever_married"])
df["work_type_n"] = le_work_type.fit_transform(df["work_type"])
df["residence_type_n"] = le_work_type.fit_transform(df["Residence_type"])
df["smoking_status_n"] = le_smoking_status.fit_transform(df["smoking_status"])

df.drop(["id", "gender", "ever_married", "work_type", "smoking_status", "Residence_type"], axis=1, inplace=True)

# Normalize values and remove outliers
def Zscore_outlier(df):
    out = []
    m = np.mean(df)
    sd = np.std(df)

    for i in df:
        z = (i - m) / sd
        if np.abs(z) > 3:
            out.append(i)

    out.sort()
    return min(out)

df.loc[df.bmi > Zscore_outlier(df["bmi"]), "bmi"] = Zscore_outlier(df["bmi"])
df.loc[df.avg_glucose_level > Zscore_outlier(df["avg_glucose_level"]), "avg_glucose_level"] = Zscore_outlier(df["avg_glucose_level"])

for column in df:
    if column != "stroke":
        df[column] = df[column] / max(df[column])

# Update CSV file
df.to_csv("healthcare-dataset-stroke-data-n.csv", sep=',', index=False)

print("Program has completed")
