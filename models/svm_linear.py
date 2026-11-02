# Linear SVM (One-vs-Rest)
import numpy as np

class SVMLinearModel:

    def __init__(self, learning_rate=0.01, epochs=200, lambda_reg=0.01):
        self.classes = None
        self.weights = None
        self.biases = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_reg = lambda_reg

    def fit_one_class(self, X, y_binary):
        # Train 1 binary SVM bang sub-gradient descent
        # y_binary: +1 cho lop dang xet, -1 cho cac lop con lai
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.epochs):
            margins = y_binary * (X @ w + b)  # (n,)

            # Gradient cua phan regularization
            dw = self.lambda_reg * w
            db = 0.0

            # Cong them gradient tu cac mau vi pham (margin < 1)
            violated = margins < 1
            if violated.any():
                dw -= (y_binary[violated, None] * X[violated]).mean(axis=0)
                db -= y_binary[violated].mean()

            w -= self.learning_rate * dw
            b -= self.learning_rate * db
        return w, b

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.weights = np.zeros((n_classes, n_features))
        self.biases  = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            # OvR: nhan +1 neu la lop c, -1 neu la lop khac
            y_binary = np.where(y == c, 1, -1).astype(float)
            self.weights[i], self.biases[i] = self.fit_one_class(X, y_binary)

    def predict(self, X):
        # Diem quyet dinh cho tung lop, chon lop co diem cao nhat
        scores = X @ self.weights.T + self.biases  # (n, n_classes)
        return self.classes[np.argmax(scores, axis=1)].astype(int)
