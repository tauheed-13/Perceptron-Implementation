## Perceptron implementation using Hinge loss function and class as -1 or 1

import numpy as np

class Perceptron3:
    def __init__(self, learning_rate=0.05, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else -1
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            for i in range(n_samples):
                x_i = X[i]
                z = np.dot(x_i, self.weights) + self.bias
                if y[i] * z < 1:   # hinge condition
                    self.weights += self.learning_rate * y[i] * x_i
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        y_pred = np.dot(self.weights*X) + self.bias
        return np.array([self.activation_function(x) for x in y_pred])

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

model = Perceptron3(learning_rate=0.1, epochs=100)
model.fit(X, y)

predictions = model.predict(X)
print("Predictions:", predictions)