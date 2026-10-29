# utility.py — các hàm dùng chung: đọc data, chia tập, tiền xử lý, confusion matrix
import os
import numpy as np
import pandas as pd


def load_data(filepath, target_col=None, no_header=False):
    # Đọc CSV, tách X và y, kiểm tra nhãn nhị phân
    # target_col: tên cột nhãn (None = dùng cột cuối)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy file: '{filepath}'\n"
            "Kiểm tra lại đường dẫn hoặc dùng --data để chỉ định đúng file."
        )

    if no_header:
        df = pd.read_csv(filepath, header=None)
        df.columns = [f"feature_{i}" for i in range(len(df.columns))]
    else:
        df = pd.read_csv(filepath)

    # Xác định cột nhãn
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Cột '{target_col}' không có trong file. Các cột: {list(df.columns)}")
        label_col = target_col
    else:
        label_col = df.columns[-1]

    # Chuẩn hóa nhãn: chấp nhận bất kỳ nhãn số nguyên không âm nào (0, 1, 2, ...)
    y_raw = df[label_col].values
    unique_labels = np.unique(y_raw)

    # Kiểm tra nhãn có phải số nguyên không
    if not np.all(y_raw == y_raw.astype(int)):
        raise ValueError(f"Nhãn phải là số nguyên. Giá trị tìm thấy: {unique_labels[:5]}")

    y = y_raw.astype(int)
    print(f"Cot nhan '{label_col}': {len(unique_labels)} lop — {sorted(unique_labels.tolist())}")

    # Bỏ cột nhãn và cột id định danh nếu có
    drop_cols = [label_col]
    for col in df.columns:
        if col.strip().lower() in ("id", "patient_id"):
            drop_cols.append(col)

    X_df = df.drop(columns=drop_cols)

    # Kiểm tra các cột không phải số
    non_numeric = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        raise ValueError(f"Các cột sau không phải số, cần xử lý trước: {non_numeric}")

    X = X_df.values.astype(np.float64)
    feature_names = list(X_df.columns)
    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    # Chia train/test có phân tầng (tỷ lệ lớp 0/1 giống nhau ở cả hai tập)
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []

    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if len(idx_c) < 2:
            raise ValueError(f"Lớp {c} chỉ có {len(idx_c)} mẫu, không đủ để chia.")
        n_test = max(1, int(len(idx_c) * test_size))
        shuffled = rng.permutation(idx_c)
        test_idx.extend(shuffled[:n_test].tolist())
        train_idx.extend(shuffled[n_test:].tolist())

    train_idx = rng.permutation(np.array(train_idx))
    test_idx  = rng.permutation(np.array(test_idx))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_preprocessor(X_train):
    # Tính median, mean, std từ tập train (dùng lại cho test và bệnh nhân mới)
    return {
        "median": np.nanmedian(X_train, axis=0),
        "mean":   np.nanmean(X_train, axis=0),
        "std":    np.nanstd(X_train, axis=0),
    }


def transform_features(X, preprocess_info):
    # 1. Điền NaN bằng median của train
    # 2. Z-score: z = (x - mean) / std  (std=0 thì dùng 1)
    X_out = X.copy().astype(np.float64)

    # Điền NaN bằng median
    for j in range(X_out.shape[1]):
        mask = np.isnan(X_out[:, j])
        if mask.any():
            X_out[mask, j] = preprocess_info["median"][j]

    # Z-score normalization
    std = preprocess_info["std"].copy()
    std[std == 0] = 1.0
    X_out = (X_out - preprocess_info["mean"]) / std
    return X_out


def confusion_matrix_counts(y_true, y_pred):
    # Tính TP, TN, FP, FN từ nhãn thật và nhãn dự đoán
    # TP=đúng có bệnh, TN=đúng không bệnh, FP=báo nhầm, FN=bỏ sót
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    return TP, TN, FP, FN
