from classes import Layer_Dense, Activation_ReLU, Activation_Softmax
import pandas as pd
import numpy as np

df = pd.read_csv("healthcare-dataset-stroke-data-n.csv")
X = df[["age", "gender_n", "hypertension", "heart_disease", "ever_married_n", "work_type_n", "avg_glucose_level", "bmi", "smoking_status_n",]]
y = df[["stroke"]]

layer1 = Layer_Dense(9, 32)
layer1.forward(np.array(X.loc[[0]]))
print(layer1.output)

