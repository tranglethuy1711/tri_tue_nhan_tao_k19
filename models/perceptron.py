# Perceptron — cập nhật trọng số mỗi khi dự đoán sai, bỏ qua khi đúng
#
# Đổi nhãn 0/1 → -1/+1. Nếu sai: w += lr * y_true * x,  b += lr * y_true
import numpy as np


class PerceptronModel:

    def __init__(self, learning_rate=0.01, epochs=100, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        # Duyệt nhiều lần, cập nhật trọng số khi dự đoán sai
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Đổi nhãn 0 → -1, giữ 1 → +1
        y_signed = np.where(y == 1, 1.0, -1.0)

        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            # Xáo trộn thứ tự mẫu mỗi epoch
            order = rng.permutation(n_samples)

            for i in order:
                z = np.dot(self.w, X[i]) + self.b
                y_pred = 1.0 if z >= 0 else -1.0

                # Chỉ cập nhật khi dự đoán sai
                if y_pred != y_signed[i]:
                    self.w += self.learning_rate * y_signed[i] * X[i]
                    self.b += self.learning_rate * y_signed[i]

    def predict(self, X):
        # Điểm >= 0 → lớp 1, điểm < 0 → lớp 0
        z = X @ self.w + self.b
        return np.where(z >= 0, 1, 0).astype(int)
