from classes import *
import pandas as pd
import numpy as np
from nnfs.datasets import spiral_data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("healthcare-dataset-stroke-data-n.csv")
X = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke", "gender_n", "ever_married_n", "work_type_n", "residence_type_n","smoking_status_n"]]
y = df[["stroke"]]
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X, y = pipeline.fit_resample(np.array(X), np.array(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#X, y = spiral_data(samples=100, classes=2)

dense1 = Layer_Dense(11, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_SGD(learning_rate=0.85)

for epoch in range(10001):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_train)
    predictions = np.argmax(loss_activation.output, axis=1)
    acccuracy = np.mean(predictions==y_train)

    if not epoch % 100:
        print("Epoch:", epoch, "acc:", acccuracy, "loss:", loss)

    loss_activation.backward(loss_activation.output, y_train)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)
predictions = np.argmax(loss_activation.output, axis=1)
acccuracy = np.mean(predictions==y_test)
print("Validation:", "acc:", acccuracy, "loss:", loss)


