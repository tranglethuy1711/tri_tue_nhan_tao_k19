# Ridge Classifier
# A @ B = nhan ma tran A voi B
import numpy as np

class RidgeClassifierModel:

    def __init__(self, lambda_reg=1.0):
        self.lambda_reg = lambda_reg

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples    = X.shape[0]
        n_features   = X.shape[1]
        n_classes    = len(self.classes)
        # Them cot 1 vao X de mo hinh tu hoc bias ma khong can bien b rieng
        Z = np.hstack([X, np.ones((n_samples, 1))])
        Y = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes):
            Y[:, i] = (y == c).astype(float)
        # np.eye: ma tran don vi; dat [-1,-1]=0 de khong phat intercept
        R         = np.eye(n_features + 1)
        R[-1, -1] = 0.0
        # np.linalg.pinv: nghich dao gia, on dinh hon inv khi ma tran gan suy bien
        A          = Z.T @ Z + self.lambda_reg * R
        self.theta = np.linalg.pinv(A) @ Z.T @ Y

    def predict(self, X):
        Z = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.classes[np.argmax(Z @ self.theta, axis=1)].astype(int)
