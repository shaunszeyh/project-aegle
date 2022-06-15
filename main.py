import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree, metrics, neighbors, svm
from sklearn.model_selection import train_test_split

# Cleaning up the data (dropping columns, setting label, etc)
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
le_gender = LabelEncoder()
le_ever_married = LabelEncoder()
le_work_type = LabelEncoder()
le_smoking_status = LabelEncoder()
df["gender_n"] = le_gender.fit_transform(df["gender"])
df["ever_married_n"] = le_ever_married.fit_transform(df["ever_married"])
df["work_type_n"] = le_ever_married.fit_transform(df["work_type"])
df["smoking_status_n"] = le_ever_married.fit_transform(df["smoking_status"])
target = df["stroke"]
df_n = df[["age", "gender_n", "hypertension", "heart_disease", "ever_married_n", "work_type_n", "avg_glucose_level", "bmi", "smoking_status_n",]]

# Create train test split
X_train, X_test, y_train, y_test = train_test_split(df_n, target, test_size=0.2)

def get_accuracy(model): # run and return accuracy of a model
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    return accuracy

# Trying with decision tree (Accuracy around 90%)
decision_tree_model = tree.DecisionTreeClassifier()
print("Accuracy for Decision Tree:", get_accuracy(decision_tree_model))

# Trying with K-Nearest Neighbours (Accuracy around 95%)
knn_model = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
print("Accuracy for KNN:", get_accuracy(knn_model))

# Trying with Support Vector Machines (Accuracy exact match with KNN)
svm_model = svm.SVC()
print("Accuracy for SVM:", get_accuracy(svm_model))