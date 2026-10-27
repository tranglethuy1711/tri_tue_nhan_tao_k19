# Logistic Regression — dự đoán xác suất qua sigmoid, tối ưu bằng gradient descent
#
# z = X @ w + b
# y_hat = sigmoid(z) = 1 / (1 + e^(-z))
# Mất mát: L = -(1/N) Σ [ y*log(ŷ) + (1-y)*log(1-ŷ) ]
# Gradient: dw = (1/N) X.T @ (ŷ - y),  db = mean(ŷ - y)
import numpy as np


def sigmoid(z):
    # σ(z) = 1 / (1 + e^(-z)), clip để tránh overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionModel:

    def __init__(self, learning_rate=0.1, epochs=300):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        # Học w và b bằng batch gradient descent
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        y = y.astype(np.float64)

        for epoch in range(self.epochs):
            # Tính xác suất dự đoán
            y_hat = sigmoid(X @ self.w + self.b)

            # Gradient của Cross-Entropy
            diff = y_hat - y
            dw = (X.T @ diff) / n_samples
            db = diff.mean()

            # Cập nhật trọng số
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        # Dự đoán 1 nếu xác suất >= 0.5
        y_hat = sigmoid(X @ self.w + self.b)
        return (y_hat >= 0.5).astype(int)
