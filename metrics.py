# metrics.py — 2 độ đo: Accuracy và Recall (macro, hỗ trợ nhiều lớp)
import numpy as np


def accuracy_score(y_true, y_pred):
    # Accuracy = số dự đoán đúng / tổng — dùng được cho cả 2 lớp lẫn nhiều lớp
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def recall_score(y_true, y_pred):
    # Macro Recall: tính recall riêng cho từng lớp rồi lấy trung bình
    # Recall của lớp c = TP_c / (TP_c + FN_c)
    # → phù hợp cả binary lẫn multi-class
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = []
    for c in np.unique(y_true):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))


def calculate_all_metrics(y_true, y_pred):
    # Tính cả 2 độ đo, trả về dict
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall":   recall_score(y_true, y_pred),
    }
