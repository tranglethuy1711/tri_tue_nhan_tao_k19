# Gaussian Naive Bayes — phân lớp dựa trên xác suất Bayes + phân phối chuẩn
import numpy as np


class GaussianNaiveBayesModel:

    def fit(self, X, y):
        # Tính prior, mean, variance cho từng lớp
        self.classes = np.unique(y)
        n_total = X.shape[0]

        self.log_prior = {}   # log P(y=c)
        self.mean = {}        # mean[c] = array (n_features,)
        self.var  = {}        # var[c]  = array (n_features,)

        for c in self.classes:
            X_c = X[y == c]
            self.log_prior[c] = np.log(len(X_c) / n_total)
            self.mean[c] = X_c.mean(axis=0)
            # Cộng 1e-9 để tránh variance = 0
            self.var[c]  = X_c.var(axis=0) + 1e-9

    def predict(self, X):
        # Tính log-posterior cho từng lớp, chọn lớp nào cao hơn
        predictions = []

        for x in X:
            scores = {}
            for c in self.classes:
                # log P(x_j | c) theo công thức Gaussian
                log_likelihood = -0.5 * np.log(2 * np.pi * self.var[c]) \
                                 - (x - self.mean[c]) ** 2 / (2 * self.var[c])
                scores[c] = self.log_prior[c] + log_likelihood.sum()

            # Chọn lớp có điểm cao hơn
            predictions.append(max(scores, key=scores.get))

        return np.array(predictions, dtype=int)
