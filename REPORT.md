# REPORT — Mô hình và Độ đo đã triển khai

> **Phiên bản hiện tại:** 5 mô hình × 2 độ đo  
> **Dataset:** `diabetes_data.csv`  
> **Ngôn ngữ:** Python 3 + NumPy + pandas  

---

##  Cấu trúc dự án

```
tri_tue_nhan_tao_k19/   ← thư mục gốc repo
├── diabetes_data.csv
├── main.py             ← điều khiển toàn bộ luồng
├── utility.py          ← hàm dùng chung (đọc data, chia tập, tiền xử lý, ...)
├── metrics.py          ← 2 độ đo
├── requirements.txt
├── REPORT.md           ← file này
└── models/
    ├── naive_bayes.py           Gaussian Naive Bayes
    ├── logistic_regression.py   Logistic Regression
    ├── perceptron.py            Perceptron
    ├── knn.py                   K-Nearest Neighbors
    └── decision_tree.py         Decision Tree (CART)
```

---

##  Mô hình đã triển khai (5/10)

### 1. Gaussian Naive Bayes — `models/naive_bayes.py`
- **Ý tưởng:** Tính xác suất theo Bayes, giả định feature độc lập và theo phân phối chuẩn.
- **Công thức:** `log P(y=c|x) = log P(y=c) + Σ log P(x_j|y=c)`
- **Phương thức:** `fit(X, y)`, `predict(X)`
- **Lưu ý:** Dùng log để tránh underflow; cộng `1e-9` vào variance để tránh chia cho 0.

### 2. Logistic Regression — `models/logistic_regression.py`
- **Ý tưởng:** Dự đoán xác suất qua sigmoid, tối ưu bằng gradient descent.
- **Công thức:** `ŷ = sigmoid(Xw + b)`,  `L = -(1/N) Σ [y log(ŷ) + (1-y) log(1-ŷ)]`
- **Phương thức:** `fit(X, y)`, `predict(X)`
- **Tham số mặc định:** `learning_rate=0.1`, `epochs=300`

### 3. Perceptron — `models/perceptron.py`
- **Ý tưởng:** Cập nhật trọng số mỗi khi dự đoán sai, bỏ qua khi đúng.
- **Công thức:** `w += lr * y_true * x`,  `b += lr * y_true`  (chỉ khi sai)
- **Phương thức:** `fit(X, y)`, `predict(X)`
- **Nhãn nội bộ:** đổi 0/1 thành -1/+1 cho phù hợp công thức.

### 4. K-Nearest Neighbors — `models/knn.py`
- **Ý tưởng:** Tìm k mẫu gần nhất, bỏ phiếu đa số.
- **Công thức khoảng cách:** `d(a,b) = sqrt(Σ(a_j - b_j)²)` — tính bằng broadcasting.
- **Phương thức:** `fit(X, y)`, `predict(X)`
- **Tham số mặc định:** `k=5`

### 5. Decision Tree (CART) — `models/decision_tree.py`
- **Ý tưởng:** Chia dữ liệu theo feature/ngưỡng có chỉ số Gini nhỏ nhất.
- **Công thức:** `Gini(y) = 1 - Σ p_k²`,  `Gini_split = (n_L/n)·Gini(L) + (n_R/n)·Gini(R)`
- **Phương thức:** `fit(X, y)`, `predict(X)`, `find_best_split`, `build_tree`, `predict_one`
- **Tham số mặc định:** `max_depth=5`, `min_samples_split=5`

---

##  Độ đo đã triển khai (2/5) — `metrics.py`

| # | Tên | Công thức | Ý nghĩa |
|---|-----|-----------|---------|
| 1 | **Accuracy** | `(TP + TN) / tổng` | Tỷ lệ dự đoán đúng |
| 2 | **Recall** | `TP / (TP + FN)` | Trong số ca bệnh thật, tìm đúng được bao nhiêu |

> **Lý do chọn Recall:** Trong y tế, bỏ sót người bệnh (FN) nguy hiểm hơn báo nhầm (FP).  
> Recall thấp = bỏ sót nhiều ca → ưu tiên tối ưu Recall.

---

##  Chưa triển khai

### Mô hình (5/10 còn lại)
| # | Tên | File |
|---|-----|------|
| 6  | Linear SVM           | `models/svm_linear.py` |
| 7  | Random Forest        | `models/random_forest.py` |
| 8  | Nearest Centroid     | `models/nearest_centroid.py` |
| 9  | Ridge Classifier     | `models/ridge_classifier.py` |
| 10 | LDA                  | `models/lda.py` |

### Độ đo (3/5 còn lại)
| # | Tên | Công thức |
|---|-----|-----------|
| 3 | Precision   | `TP / (TP + FP)` |
| 4 | Specificity | `TN / (TN + FP)` |
| 5 | F1-Score    | `2TP / (2TP + FP + FN)` |

---

##  Cách chạy

```bash
pip install -r requirements.txt
python main.py
```

Output: bảng benchmark in ra console, `benchmark_results.csv`, `comparison_chart.png`,  
sau đó menu dự đoán tương tác cho bệnh nhân mới.
