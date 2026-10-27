# Decision Tree (CART) — chia dữ liệu theo feature/ngưỡng có Gini nhỏ nhất
#
# Gini(y) = 1 - Σ p_k²
# Gini_split = (n_L/n)*Gini(trái) + (n_R/n)*Gini(phải)
import numpy as np


def gini(y):
    # Gini(y) = 1 - Σ p_k²
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)


class Node:
    # Nút trong cây: nút lá chỉ có label, nút trong có feature + threshold + 2 nhánh

    def __init__(self):
        self.feature = None     # chỉ số feature dùng để chia
        self.threshold = None   # ngưỡng chia
        self.left = None        # nhánh trái (≤ threshold)
        self.right = None       # nhánh phải (> threshold)
        self.label = None       # nhãn (chỉ nút lá mới có)

    def is_leaf(self):
        return self.label is not None


class DecisionTreeModel:

    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def find_best_split(self, X, y):
        # Thử tất cả feature và ngưỡng, trả về cặp (feature, threshold) tốt nhất
        n = X.shape[0]
        current_gini = gini(y)
        best_gini = current_gini
        best_feature = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            values = np.unique(X[:, feature_idx])
            if len(values) < 2:
                continue
            # Ngưỡng ứng viên = trung điểm giữa 2 giá trị liên tiếp
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                n_left, n_right = left_mask.sum(), right_mask.sum()

                if n_left == 0 or n_right == 0:
                    continue

                gini_split = (n_left/n) * gini(y[left_mask]) \
                           + (n_right/n) * gini(y[right_mask])

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_feature is None:
            return None
        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        # Đệ quy xây cây: dừng khi nút thuần / quá sâu / quá ít mẫu / không chia được
        node = Node()

        # Điều kiện dừng → tạo nút lá
        all_same_class = len(np.unique(y)) == 1
        too_deep = depth >= self.max_depth
        too_few  = X.shape[0] < self.min_samples_split

        if all_same_class or too_deep or too_few:
            count = np.bincount(y, minlength=2)
            node.label = 0 if count[0] >= count[1] else 1
            return node

        # Tìm điểm chia tốt nhất
        result = self.find_best_split(X, y)
        if result is None:
            # Không có chia nào tốt hơn → nút lá
            count = np.bincount(y, minlength=2)
            node.label = 0 if count[0] >= count[1] else 1
            return node

        feature_idx, threshold = result
        left_mask = X[:, feature_idx] <= threshold

        node.feature   = feature_idx
        node.threshold = threshold
        node.left  = self.build_tree(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self.build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y):
        # Xây cây từ dữ liệu train
        self.root = self.build_tree(X, y.astype(int), depth=0)

    def predict_one(self, x, node):
        # Đi từ gốc xuống lá theo điều kiện chia để tìm nhãn của mẫu x
        if node.is_leaf():
            return node.label
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        # Dự đoán nhãn cho tất cả mẫu
        return np.array([self.predict_one(x, self.root) for x in X], dtype=int)
