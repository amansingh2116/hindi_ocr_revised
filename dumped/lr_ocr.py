import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import pandas as pd


'''
x_1: This should be a matrix where each row represents a training example, and each column represents a feature.
y_1: This should be a vector containing the labels corresponding to the training examples.
x_t1: This should be a matrix where each row represents a test example, and each column represents a feature.
y_t1: This should be a vector containing the labels corresponding to the test examples.
'''
x_1, y_1, x_t1, y_t1 = []*4

# Logistic Regression

def sigmoid(z):
    sig = 1.0 / (1 + np.exp(-z))
    return sig

# Regularized cost function
def cost_fun(angle, X, y, _lambda=0.1):
    m = len(y)
    h = sigmoid(X.dot(angle))
    reg = (_lambda / (2 * m)) * np.sum(angle**2)
    ct_fn = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg
    return ct_fn

# Regularized gradient function
def gradient(angle, X, y, _lambda=0.1):
    m, n = X.shape
    angle = angle.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(angle))
    reg = _lambda * angle / m
    grad = ((1 / m) * X.T.dot(h - y)) + reg
    return grad

# Optimal angle
def logisticRegression(X, y, angle, _lambda=0.1):
    result = minimize(fun=cost_fun, x0=angle, args=(X, y, _lambda),
                      method='TNC', jac=gradient)
    return result.x

# Training

k = 10
n = 784

# One vs all

ord = np.unique(y_t1)
lambda_ = [0.001, 0.01, 0.1, 1, 10, 100]
inter_array = {}
final_predictions = []
final_accuracy = []

for lamba in lambda_:
    i = 0
    all_angle = np.zeros((k, n + 1))
    for dig in ord:
        # Set the labels in 0 and 1
        tmp_y = np.array(y_1 == dig, dtype=int)
        optangle = logisticRegression(x_1, tmp_y, np.zeros((n + 1, 1)), lamba)
        all_angle[i] = optangle
        i += 1
    inter_array[lamba] = all_angle

    # Predictions
    P = sigmoid(x_t1.dot(all_angle.T))  # Probability for each dig
    p = [ord[np.argmax(P[i, :])] for i in range(x_t1.shape[0])]
    final_predictions.append(p)
    final_accuracy.append(accuracy_score(y_t1, p) * 100)
    print("Test Accuracy ", accuracy_score(y_t1, p) * 100, '%')

acc = {'Accuracy in %': final_accuracy, 'Lambda': lambda_}
accuracy_frame = pd.DataFrame(acc)
accuracy_frame
