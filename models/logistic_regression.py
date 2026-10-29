# Logistic Regression — multi-class dùng Softmax + Cross-Entropy
#
# z = X @ W + b          (W có shape n_features x n_classes)
# y_hat = softmax(z)     mỗi hàng là xác suất cho từng lớp
# Gradient: dW = X.T @ (y_hat - Y_onehot) / N,  db = mean(y_hat - Y_onehot)
import numpy as np


def softmax(z):
    # Trừ max để tránh overflow, rồi chuẩn hóa thành xác suất
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LogisticRegressionModel:

    def __init__(self, learning_rate=0.1, epochs=300):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # One-hot encode nhãn
        Y = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes):
            Y[y == c, i] = 1.0

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        for _ in range(self.epochs):
            y_hat = softmax(X @ self.W + self.b)
            diff  = y_hat - Y
            self.W -= self.learning_rate * (X.T @ diff) / n_samples
            self.b -= self.learning_rate * diff.mean(axis=0)

    def predict(self, X):
        # Lấy lớp có xác suất cao nhất
        probs = softmax(X @ self.W + self.b)
        return self.classes[np.argmax(probs, axis=1)].astype(int)
