import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def predict(self, features):
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return np.dot(features, self.weights) + self.bias

    def evaluate(self, features, true_values):
        predictions = self.predict(features)
        return np.mean((predictions - true_values) ** 2)

    def train(self, features, true_values, learning_rate, epochs):
        n = len(true_values)
        features = np.array(features, dtype=float)
        true_values = np.array(true_values, dtype=float)

        if self.weights is None:
            num_features = features.shape[1]
            self.weights = np.random.randn(num_features) * 0.01

        for _ in range(epochs):
            predictions = self.predict(features)
            error = predictions - true_values

            gradient_w = (2/n) * np.dot(features.T, error)
            gradient_b = (2/n) * np.sum(error)

            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b
