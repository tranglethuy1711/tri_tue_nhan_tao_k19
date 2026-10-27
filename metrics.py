# metrics.py — 2 độ đo: Accuracy và Recall
from utility import confusion_matrix_counts


def accuracy_score(y_true, y_pred):
    # Accuracy = (TP + TN) / tổng — tỷ lệ dự đoán đúng
    TP, TN, FP, FN = confusion_matrix_counts(y_true, y_pred)
    total = TP + TN + FP + FN
    if total == 0:
        return 0.0
    return (TP + TN) / total


def recall_score(y_true, y_pred):
    # Recall = TP / (TP + FN) — trong số ca bệnh thật, tìm đúng được bao nhiêu
    # Recall thấp = bỏ sót nhiều ca bệnh → quan trọng nhất trong y tế
    TP, TN, FP, FN = confusion_matrix_counts(y_true, y_pred)
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def calculate_all_metrics(y_true, y_pred):
    # Tính cả 2 độ đo, trả về dict
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall":   recall_score(y_true, y_pred),
    }
