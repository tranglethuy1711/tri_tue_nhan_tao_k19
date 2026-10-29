# Perceptron — multi-class: mỗi lớp có 1 bộ trọng số riêng
#
# Với mỗi mẫu, tính điểm cho tất cả lớp rồi chọn lớp cao nhất.
# Nếu sai: tăng điểm lớp đúng, giảm điểm lớp bị chọn nhầm.
import numpy as np


class PerceptronModel:

    def __init__(self, learning_rate=0.01, epochs=100, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # W[c] là trọng số cho lớp c, b[c] là bias
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.epochs):
            for i in rng.permutation(n_samples):
                scores    = self.W @ X[i] + self.b
                pred_idx  = int(np.argmax(scores))
                true_idx  = int(np.where(self.classes == y[i])[0][0])

                if pred_idx != true_idx:
                    # Tăng điểm lớp đúng, giảm điểm lớp bị chọn nhầm
                    self.W[true_idx] += self.learning_rate * X[i]
                    self.b[true_idx] += self.learning_rate
                    self.W[pred_idx] -= self.learning_rate * X[i]
                    self.b[pred_idx] -= self.learning_rate

    def predict(self, X):
        scores = X @ self.W.T + self.b
        return self.classes[np.argmax(scores, axis=1)].astype(int)
