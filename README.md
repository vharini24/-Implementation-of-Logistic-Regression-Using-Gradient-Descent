# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create a dataset with input features and binary target values, then normalize the feature data.

2. Initialize weights, bias, learning rate, and define the sigmoid activation function for Logistic Regression.

3. Train the model using Stochastic Gradient Descent by calculating prediction, error, and updating weights and bias for each data sample.

4. Use the trained model to calculate probabilities and predict binary class labels for all input data.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: V. HARINI
RegisterNumber:  212225040113
*/

import numpy as np

# Input Features
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target (Binary Classification)
y = np.array([0, 0, 1, 1, 1], dtype=float)

# Feature Scaling
X = X / np.max(X, axis=0)

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

# Hyperparameters
lr = 0.1
epochs = 1000

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# SGD Training
for epoch in range(epochs):

    for i in range(len(X)):

        # Linear equation
        z = np.dot(X[i], w) + b

        # Logistic prediction
        y_pred = sigmoid(z)

        # Error
        error = y_pred - y[i]

        # Update weights and bias
        w = w - lr * error * X[i]
        b = b - lr * error

# Final weights
print("Weights:", w)
print("Bias:", b)

# Predictions
print("\nPredictions:")

for i in range(len(X)):

    z = np.dot(X[i], w) + b
    prob = sigmoid(z)

    prediction = 1 if prob >= 0.5 else 0

    print(f"Actual: {int(y[i])}, Predicted: {prediction}, Probability: {prob:.2f}")

```

## Output:
<img width="512" height="202" alt="image" src="https://github.com/user-attachments/assets/e87ef706-012a-4adc-833d-ee087200f03d" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

