# Random Forest
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.decision_tree import DecisionTreeModel


def bootstrap(X, y, rng):
    # Lay mau co hoan lai: sinh n chi so ngau nhien (co the trung nhau)
    idx = rng.integers(0, len(y), size=len(y))
    return X[idx], y[idx]


class RandomForestModel:

    def __init__(self, n_trees=50, max_depth=5, min_samples_split=5, random_state=42):
        self.classes = None
        self.trees = None
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state

    def fit(self, X, y):
        self.classes = np.unique(y)
        rng          = np.random.default_rng(self.random_state)
        n_features   = X.shape[1]

        # Moi nut chi xet sqrt(n_features) feature ngau nhien — giam tuong quan giua cay
        max_feat = int(np.sqrt(n_features))

        self.trees = []
        for i in range(self.n_trees):
            seed     = int(rng.integers(0, 1000000))
            X_b, y_b = bootstrap(X, y, rng)
            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_b, y_b)
            self.trees.append(tree)

    def predict(self, X):
        max_label = int(self.classes.max()) + 1
        results   = []
        for i in range(X.shape[0]):
            x_row = X[i].reshape(1, -1)  # reshape de predict nhan dung shape (1, n_features)
            votes = np.zeros(max_label, dtype=int)
            for tree in self.trees:
                votes[int(tree.predict(x_row)[0])] += 1
            results.append(int(votes.argmax()))
        return np.array(results, dtype=int)

