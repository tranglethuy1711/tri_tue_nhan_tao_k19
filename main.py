"""
main.py — Chạy benchmark 5 mô hình và dự đoán tương tác
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from utility import load_data, split_data, fit_preprocessor, transform_features
from metrics  import calculate_all_metrics

from models.naive_bayes         import GaussianNaiveBayesModel
from models.logistic_regression import LogisticRegressionModel
from models.perceptron          import PerceptronModel
from models.knn                 import KNNModel
from models.decision_tree       import DecisionTreeModel


def run_benchmark(X_train, X_test, y_train, y_test):
    models = [
        ("Naive Bayes",         GaussianNaiveBayesModel()),
        ("Logistic Regression", LogisticRegressionModel(learning_rate=0.1, epochs=300)),
        ("Perceptron",          PerceptronModel(learning_rate=0.01, epochs=100)),
        ("KNN (k=5)",           KNNModel(k=5)),
        ("Decision Tree",       DecisionTreeModel(max_depth=5, min_samples_split=5)),
    ]

    results = []
    trained_models = []

    header = f"{'Model':<22} | {'Accuracy':>9} | {'Recall':>9} | {'Time(s)':>9}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    for name, model in models:
        row = {"Model": name, "Status": "OK", "Error": ""}
        try:
            t_start = time.perf_counter()
            model.fit(X_train, y_train)
            t_end = time.perf_counter()

            y_pred = model.predict(X_test)
            scores = calculate_all_metrics(y_test, y_pred)

            row["Accuracy"] = round(scores["Accuracy"], 4)
            row["Recall"]   = round(scores["Recall"],   4)
            row["Time(s)"]  = round(t_end - t_start,    4)

            acc_pct = row["Accuracy"] * 100
            rec_pct = row["Recall"]   * 100
            print(f"{name:<22} | {acc_pct:>8.2f}% | {rec_pct:>8.2f}% | {row['Time(s)']:>9.4f}")
            trained_models.append((name, model))

        except Exception as e:
            row.update({"Status": "FAILED", "Error": str(e),
                        "Accuracy": "N/A", "Recall": "N/A", "Time(s)": "N/A"})
            print(f"{name:<22} | {'FAILED':>9} | {str(e)[:25]}")

        results.append(row)

    print("─" * len(header))

    # In model dẫn đầu
    ok = [r for r in results if r["Status"] == "OK"]
    if ok:
        best_acc = max(ok, key=lambda r: r["Accuracy"])
        best_rec = max(ok, key=lambda r: r["Recall"])
        print(f"\n[KET QUA] Accuracy cao nhat : {best_acc['Model']} ({best_acc['Accuracy']*100:.2f}%)")
        print(f"[KET QUA] Recall cao nhat   : {best_rec['Model']} ({best_rec['Recall']*100:.2f}%)")

    # Lưu CSV
    pd.DataFrame(results).to_csv("benchmark_results.csv", index=False)
    print(f"\nDa luu ket qua: {os.path.abspath('benchmark_results.csv')}")

    return trained_models


def interactive_prediction(trained_models, preprocess_info, feature_names):
    if not trained_models:
        print("[LOI] Khong co mo hinh de du doan.")
        return

    while True:
        print("\n" + "=" * 45)
        print("  CHON MO HINH DE DU DOAN")
        print("=" * 45)
        for i, (name, _) in enumerate(trained_models, 1):
            print(f"  {i}. {name}")
        print("  0. Thoat")

        choice = input(f"Nhap so (0-{len(trained_models)}): ").strip()
        if choice == "0":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(trained_models)):
            print("Lua chon khong hop le, thu lai.")
            continue

        model_name, model = trained_models[int(choice) - 1]
        print(f"\nMo hinh da chon: {model_name}")
        print("Nhap thong tin benh nhan:")

        values = []
        for feat in feature_names:
            while True:
                raw = input(f"  {feat}: ").strip()
                try:
                    values.append(float(raw))
                    break
                except ValueError:
                    print(f"  '{raw}' khong phai so, nhap lai.")

        x_new = transform_features(np.array(values).reshape(1, -1), preprocess_info)
        pred  = int(model.predict(x_new)[0])
        label_map = {0: "Khong mac tieu duong (lop 0)",
                     1: "Tien tieu duong (lop 1)",
                     2: "Mac tieu duong (lop 2)"}
        print(f"\n[KET QUA] {model_name}: {label_map.get(pred, f'Lop {pred}')}")

        if input("\nDu doan benh nhan khac? (y/n): ").strip().lower() != "y":
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         default="diabetes_data.csv")
    parser.add_argument("--target",       default="Diabetes_012")
    parser.add_argument("--test-size",    type=float, default=0.2)
    parser.add_argument("--random-state", type=int,   default=42)
    parser.add_argument("--no-header",    action="store_true")
    parser.add_argument("--max-samples",  type=int,   default=3000)
    args = parser.parse_args()

    print("=" * 55)
    print("  5 MO HINH x 2 DO DO — CHAN DOAN TIEU DUONG")
    print("=" * 55)

    # Đọc dữ liệu
    X, y, feature_names = load_data(args.data, target_col=args.target, no_header=args.no_header)
    print(f"\nDu lieu: {len(y)} mau | {X.shape[1]} feature | Lop 1: {y.sum()} | Lop 0: {(y==0).sum()}")

    # Lấy mẫu con nếu dataset quá lớn
    if args.max_samples > 0 and len(y) > args.max_samples:
        print(f"Dataset lon → lay {args.max_samples} mau can bang lop...")
        rng  = np.random.default_rng(args.random_state)
        half = args.max_samples // 2
        idx0 = rng.choice(np.where(y == 0)[0], size=min(half, (y==0).sum()), replace=False)
        idx1 = rng.choice(np.where(y == 1)[0], size=min(half, (y==1).sum()), replace=False)
        sel  = rng.permutation(np.concatenate([idx0, idx1]))
        X, y = X[sel], y[sel]

    # Chia train/test
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.random_state)
    print(f"Train: {len(y_train)} mau | Test: {len(y_test)} mau")

    # Tiền xử lý
    preprocess_info = fit_preprocessor(X_train)
    X_train = transform_features(X_train, preprocess_info)
    X_test  = transform_features(X_test,  preprocess_info)

    # Benchmark
    trained_models = run_benchmark(X_train, X_test, y_train, y_test)

    # Dự đoán tương tác
    interactive_prediction(trained_models, preprocess_info, feature_names)


if __name__ == "__main__":
    main()

