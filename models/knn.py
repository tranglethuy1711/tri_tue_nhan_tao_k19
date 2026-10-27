# KNN — tìm k mẫu gần nhất, bỏ phiếu đa số để dự đoán nhãn
import numpy as np


def euclidean_distance_matrix(X1, X2):
    # Tính ma trận khoảng cách (n1, n2) dùng broadcasting, tránh vòng for lồng nhau
    # ||a-b||² = ||a||² + ||b||² - 2*(a·b)
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)  # (n2, 1)
    dot = X1 @ X2.T                                # (n1, n2)
    dist_sq = sq1 + sq2.T - 2 * dot
    return np.sqrt(np.clip(dist_sq, 0, None))      # clip để tránh sqrt âm


class KNNModel:

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN không train gì, chỉ lưu lại dữ liệu
        self.X_train = X.copy()
        self.y_train = y.copy()
        # Giảm k nếu lớn hơn số mẫu train
        if self.k > X.shape[0]:
            self.k = X.shape[0]

    def predict(self, X):
        # Tìm k láng giềng gần nhất và bỏ phiếu đa số
        # Tính toàn bộ ma trận khoảng cách một lần
        dist_matrix = euclidean_distance_matrix(X, self.X_train)  # (n_test, n_train)

        # Lớp đa số trong train (dùng để phá hòa)
        majority = int(np.bincount(self.y_train).argmax())

        predictions = []
        for i in range(X.shape[0]):
            # Lấy chỉ số k mẫu gần nhất
            nearest_idx = np.argsort(dist_matrix[i])[:self.k]
            neighbor_labels = self.y_train[nearest_idx]

            # Đếm phiếu
            count = np.bincount(neighbor_labels, minlength=2)
            if count[0] == count[1]:
                predictions.append(majority)  # hòa → chọn lớp đa số trong train
            else:
                predictions.append(int(count.argmax()))

        return np.array(predictions, dtype=int)
