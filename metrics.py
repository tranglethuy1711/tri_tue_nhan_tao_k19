# metrics.py — 5 độ đo: Accuracy, Recall, Precision, Specificity, F1 (macro)
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


def precision_score(y_true, y_pred):
    # Macro Precision: TP_c / (TP_c + FP_c) cho tung lop c
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    precisions = []
    for c in np.unique(y_true):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(precisions))


def specificity_score(y_true, y_pred):
    # Macro Specificity: TN_c / (TN_c + FP_c) cho tung lop c
    # TN_c = khong phai c, du doan khong phai c
    # FP_c = khong phai c, du doan la c
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    specificities = []
    for c in np.unique(y_true):
        tn = np.sum((y_true != c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specificities))


def f1_score(y_true, y_pred):
    # Macro F1: 2*TP_c / (2*TP_c + FP_c + FN_c) cho tung lop c
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in np.unique(y_true):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def calculate_all_metrics(y_true, y_pred):
    # Tính cả 5 độ đo, trả về dict
    return {
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred),
        "Recall":      recall_score(y_true, y_pred),
        "Specificity": specificity_score(y_true, y_pred),
        "F1":          f1_score(y_true, y_pred),
    }
