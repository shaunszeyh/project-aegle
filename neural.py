from classes import *
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Neural network (Accuracy aroud 86%)
df = pd.read_csv("healthcare-dataset-stroke-data-n.csv")
X = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "gender_n", "ever_married_n", "work_type_n", "residence_type_n","smoking_status_n"]]
y = df[["stroke"]]
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(np.array(X), np.array(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y, shuffle=True)

dense1 = Layer_Dense(10, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)

for epoch in range(10001):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_train)
    predictions = np.argmax(loss_activation. output, axis=1)
    acccuracy = np.mean(predictions==y_train)

    if not epoch % 100:
        print("Epoch:", epoch, "acc:", acccuracy, "loss:", loss, "lr:", optimizer.current_learning_rate)

    loss_activation.backward(loss_activation.output, y_train)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    if not epoch % 100:
        softmax_test = Activation_Softmax_Loss_CategoricalCrossEntropy()
        dense1.forward(X_test)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = softmax_test.forward(dense2.output, y_test)
        predictions = np.argmax(softmax_test.output, axis=1)
        acccuracy = np.mean(predictions==y_test)
        print("Test acc:", acccuracy, "loss:", loss)

# Save the weights and biases to a .npy file for use in website

np.save('parameters/weights1.npy', dense1.weights)
np.save('parameters/biases1.npy', dense1.biases)
np.save('parameters/weights2.npy', dense2.weights)
np.save('parameters/biases2.npy', dense2.biases)

# Check test split after training

def run_neural_network(inputs):
    softmax = Activation_Softmax()
    dense1.forward(inputs)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    softmax.forward(dense2.output)
    predictions = np.argmax(softmax.output, axis=1)
    return predictions, np.max(softmax.output)

print(run_neural_network(np.array([1.0,1.0,0.0,0.22985206447339368,0.20799180327868855,0.0,1.0,0.5,1.0,0.0]))) # Should return [0]


