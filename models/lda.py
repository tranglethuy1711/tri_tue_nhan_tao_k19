# Linear Discriminant Analysis (LDA)
# A @ B = nhan ma tran A voi B
import numpy as np


class LDAModel:

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples    = X.shape[0]
        n_features   = X.shape[1]
        n_classes    = len(self.classes)
        self.means   = {}
        self.priors  = {}
        Sigma        = np.zeros((n_features, n_features))

        for c in self.classes:
            X_c  = X[y == c]
            n_c  = len(X_c)
            mu_c = X_c.mean(axis=0)
            self.means[c]  = mu_c
            self.priors[c] = np.log(n_c / n_samples)  # dung log tranh so qua nho
            diff  = X_c - mu_c
            Sigma = Sigma + np.dot(diff.T, diff)

        Sigma = Sigma / (n_samples - n_classes)
        Sigma = Sigma + 1e-6 * np.eye(n_features)  # + 1e-6*I tranh ma tran suy bien
        # np.linalg.pinv: nghich dao gia, on dinh hon inv khi ma tran gan suy bien
        self.sigma_inv = np.linalg.pinv(Sigma)

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            mu           = self.means[c]
            sigma_inv_mu = np.dot(self.sigma_inv, mu)
            scores[:, i] = np.dot(X, sigma_inv_mu) \
                           - 0.5 * np.dot(mu, sigma_inv_mu) \
                           + self.priors[c]
        return self.classes[np.argmax(scores, axis=1)].astype(int)
