import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree, metrics, neighbors
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

# Trying with decision tree (Accuracy around 90%)
decision_tree = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(df_n, target, test_size=0.2)
decision_tree.fit(X_train, y_train)
decision_tree_prediction = decision_tree.predict(X_test)
decision_tree_accuracy = metrics.accuracy_score(y_test, decision_tree_prediction)
print("Accuracy for decision tree:", decision_tree_accuracy)

# Trying with K-Nearest Neighbours (Accuracy around)
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
knn_accuracy = metrics.accuracy_score(y_test, knn_prediction)
print("Accuracy for KNN:", knn_accuracy)

