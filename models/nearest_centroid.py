# Nearest Centroid
import numpy as np

class NearestCentroidModel:

    def __init__(self):
        self.classes = None
        self.centroids = None

    def fit(self, X, y):
        self.classes   = np.unique(y)
        self.centroids = {}
        for c in self.classes:
            self.centroids[c] = X[y == c].mean(axis=0)

    def predict(self, X):
        dists = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            diff        = X - self.centroids[c]
            dists[:, i] = np.sum(diff ** 2, axis=1)  # axis=1: tinh tong theo tung hang
        return self.classes[np.argmin(dists, axis=1)].astype(int)
