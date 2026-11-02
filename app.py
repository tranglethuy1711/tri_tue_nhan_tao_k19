"""
app.py — Giao diện web Streamlit cho hệ thống chẩn đoán tiểu đường
Chạy: streamlit run app.py
"""
import os, sys, time
import numpy as np
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utility import load_data, split_data, fit_preprocessor, transform_features
from metrics import calculate_all_metrics
from models.naive_bayes import GaussianNaiveBayesModel
from models.logistic_regression import LogisticRegressionModel
from models.perceptron import PerceptronModel
from models.knn import KNNModel
from models.decision_tree import DecisionTreeModel
from models.svm_linear import SVMLinearModel
from models.random_forest import RandomForestModel
from models.nearest_centroid import NearestCentroidModel
from models.ridge_classifier import RidgeClassifierModel
from models.lda import LDAModel

app = Flask(__name__)

LABEL_MAP = {0: "Không mắc bệnh tiểu đường", 1: "Tiền tiểu đường", 2: "Mắc bệnh tiểu đường"}

# Luu mo hinh da train va thong so tien xu ly vao bien toan cuc
state = {}


def make_models():
    return [
        ("Naive Bayes", GaussianNaiveBayesModel()),
        ("Logistic Regression", LogisticRegressionModel(learning_rate=0.1, epochs=300)),
        ("Perceptron", PerceptronModel(learning_rate=0.01, epochs=100)),
        ("KNN (k=5)", KNNModel(k=5)),
        ("Decision Tree", DecisionTreeModel(max_depth=5, min_samples_split=5)),
        ("SVM Linear", SVMLinearModel(learning_rate=0.01, epochs=200, lambda_reg=0.01)),
        ("Random Forest", RandomForestModel(n_trees=50, max_depth=5, min_samples_split=5)),
        ("Nearest Centroid", NearestCentroidModel()),
        ("Ridge Classifier", RidgeClassifierModel(lambda_reg=1.0)),
        ("LDA", LDAModel()),
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_benchmark", methods=["POST"])
def run_benchmark():
    body = request.get_json(silent=True) or {}
    max_samples = int(body.get("max_samples", 3000))
    random_seed = 42

    # --- Doc va chuan bi du lieu ---
    X, y, feature_names = load_data("diabetes_data.csv", target_col="Diabetes_012")

    rng = np.random.default_rng(random_seed)
    half = max_samples // 2
    n1 = min(max_samples // 6, int((y == 1).sum()))
    n2 = min(max_samples // 3, int((y == 2).sum()))
    n0 = min(half, int((y == 0).sum()))
    sel = rng.permutation(np.concatenate([
        rng.choice(np.where(y == 0)[0], n0, replace=False),
        rng.choice(np.where(y == 1)[0], n1, replace=False),
        rng.choice(np.where(y == 2)[0], n2, replace=False),
    ]))
    X, y = X[sel], y[sel]

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=random_seed)
    pre = fit_preprocessor(X_train)
    X_train = transform_features(X_train, pre)
    X_test = transform_features(X_test, pre)

    # --- Train va danh gia tung mo hinh ---
    results = []
    trained = {}

    for name, model in make_models():
        try:
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0
            scores = calculate_all_metrics(y_test, model.predict(X_test))
            results.append({
                "name": name,
                "accuracy": round(scores["Accuracy"] * 100, 2),
                "precision": round(scores["Precision"] * 100, 2),
                "recall": round(scores["Recall"] * 100, 2),
                "specificity": round(scores["Specificity"] * 100, 2),
                "f1": round(scores["F1"] * 100, 2),
                "time": round(elapsed, 3),
                "status": "OK",
            })
            trained[name] = model
        except Exception as e:
            results.append({"name": name, "status": "FAILED", "error": str(e)})

    # Luu vao bien toan cuc de dung khi du doan
    state["trained"] = trained
    state["pre"] = pre
    state["features"] = feature_names

    return jsonify({
        "results": results,
        "feature_names": feature_names,
        "data_info": {
            "total": len(y),
            "train": len(y_train),
            "test": len(y_test),
            "train_dist": {str(c): int((y_train == c).sum()) for c in [0, 1, 2]},
            "test_dist": {str(c): int((y_test == c).sum()) for c in [0, 1, 2]},
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "trained" not in state:
        return jsonify({"error": "Chưa chạy benchmark. Vui lòng tải lại trang."}), 400

    body = request.get_json()
    model_name = body["model"]
    values = body["values"]

    model = state["trained"][model_name]
    x_new = transform_features(np.array(values).reshape(1, -1), state["pre"])
    pred = int(model.predict(x_new)[0])

    return jsonify({"prediction": pred, "label": LABEL_MAP.get(pred, f"Lớp {pred}")})


if __name__ == "__main__":
    print("App is running on: http://localhost:5000")
    app.run(debug=True, port=5000)
