Class LinearRegression,

LinearRegression is a simple linear regression implementation in Python. It uses gradient descent for optimization and supports training, prediction, and evaluation.

Train:
model = MyLinearRegression()
model.train(features, true_values, learning_rate=0.01, epochs=1000)
  features: A 2D list or NumPy array of input data
  true_values: A 1D list or NumPy array of target values
  learning_rate: Step size for gradient descent (default: 0.01)
  epochs: Number of iterations (default: 1000)

Predict:
predictions = model.predict(new_features)
mse = model.evaluate(test_features, test_true_values)
