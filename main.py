import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, metrics, neighbors, svm
from sklearn.model_selection import train_test_split

# Get the columns needed
df = pd.read_csv("healthcare-dataset-stroke-data-n.csv")
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
