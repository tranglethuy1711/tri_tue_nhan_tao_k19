// Cac ten do do tuong ung voi key trong JSON tra ve tu server
const METRIC_KEYS   = ["accuracy", "precision", "recall", "specificity", "f1"];
const METRIC_LABELS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"];

// Luu ten feature de dung khi tao form nhap lieu
let featureNames = [];

// Chay ngay khi trang load xong
window.onload = function () {
    runBenchmark();
};


// =====================================================================
// 1. Goi API /run_benchmark va hien thi ket qua
// =====================================================================
function runBenchmark() {
    document.getElementById("section-loading").classList.remove("hidden");
    document.getElementById("section-results").classList.add("hidden");

    fetch("/run_benchmark", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ max_samples: 3000 }),
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        featureNames = data.feature_names;
        showDataInfo(data.data_info);
        showResultsTable(data.results);
        showBestCards(data.results);
        buildPredictForm(data.feature_names, data.results);

        document.getElementById("section-loading").classList.add("hidden");
        document.getElementById("section-results").classList.remove("hidden");
    })
    .catch(function(err) {
        document.getElementById("loading-text").textContent = "Loi: " + err.message;
    });
}


// =====================================================================
// 2. Hien thi thong tin du lieu train / test
// =====================================================================
function showDataInfo(info) {
    var td = info.train_dist;
    var ed = info.test_dist;

    var html =
        '<div class="info-grid">' +
            makeInfoItem(info.total,  "Tổng mẫu")  +
            makeInfoItem(info.train,  "Tập train") +
            makeInfoItem(info.test,   "Tập test")  +
            makeInfoItem(td["0"] + " / " + ed["0"], "Lớp 0 (train/test)") +
            makeInfoItem(td["1"] + " / " + ed["1"], "Lớp 1 (train/test)") +
            makeInfoItem(td["2"] + " / " + ed["2"], "Lớp 2 (train/test)") +
        '</div>';

    document.getElementById("data-info-content").innerHTML = html;
}

function makeInfoItem(val, label) {
    return '<div class="info-item"><div class="val">' + val +
           '</div><div class="lbl">' + label + '</div></div>';
}


// =====================================================================
// 3. Hien thi bang so sanh 10 mo hinh
// =====================================================================
function showResultsTable(results) {
    var okResults = results.filter(function(r) { return r.status === "OK"; });

    // Tim gia tri cao nhat va thap nhat theo tung cot do do
    var maxVals = {}, minVals = {};
    METRIC_KEYS.forEach(function(key) {
        var vals = okResults.map(function(r) { return r[key]; });
        maxVals[key] = Math.max.apply(null, vals);
        minVals[key] = Math.min.apply(null, vals);
    });

    var html = "";
    results.forEach(function(r) {
        if (r.status !== "OK") {
            html += "<tr><td>" + r.name + "</td><td colspan='6' style='color:red'>FAILED: " + r.error + "</td></tr>";
            return;
        }
        html += "<tr><td>" + r.name + "</td>";
        METRIC_KEYS.forEach(function(key) {
            var cls = "";
            if (r[key] === maxVals[key]) cls = "cell-best";
            else if (r[key] === minVals[key]) cls = "cell-worst";
            html += '<td class="' + cls + '">' + r[key].toFixed(2) + "%</td>";
        });
        html += "<td>" + r.time + "s</td></tr>";
    });

    document.getElementById("results-tbody").innerHTML = html;
}


// =====================================================================
// 4. Cards mo hinh dan dau theo tung do do
// =====================================================================
function showBestCards(results) {
    var okResults = results.filter(function(r) { return r.status === "OK"; });
    var html = "";

    METRIC_KEYS.forEach(function(key, i) {
        var best = okResults.reduce(function(a, b) { return a[key] > b[key] ? a : b; });
        html +=
            '<div class="best-card">' +
                '<div class="metric">' + METRIC_LABELS[i] + '</div>' +
                '<div class="score">'  + best[key].toFixed(2) + '%</div>' +
                '<div class="model">'  + best.name + '</div>' +
            '</div>';
    });

    document.getElementById("best-cards").innerHTML = html;
}


// =====================================================================
// 5. Tao form nhap lieu du doan
// =====================================================================
function buildPredictForm(features, results) {
    // Dropdown chon mo hinh
    var select = document.getElementById("model-select");
    select.innerHTML = "";
    results.filter(function(r) { return r.status === "OK"; }).forEach(function(r) {
        var opt = document.createElement("option");
        opt.value = opt.textContent = r.name;
        select.appendChild(opt);
    });

    // Cac o nhap feature
    var html = "";
    features.forEach(function(name, i) {
        html +=
            '<label>' + name +
                '<input type="number" id="feat_' + i + '" value="0" step="any">' +
            '</label>';
    });
    document.getElementById("feature-grid").innerHTML = html;
}


// =====================================================================
// 6. Goi API /predict va hien thi ket qua
// =====================================================================
function runPredict() {
    // Doc gia tri tung feature
    var values = featureNames.map(function(_, i) {
        return parseFloat(document.getElementById("feat_" + i).value) || 0;
    });

    var modelName = document.getElementById("model-select").value;
    var btn       = document.getElementById("predict-btn");

    btn.disabled    = true;
    btn.textContent = "Đang dự đoán...";

    fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ model: modelName, values: values }),
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        showResult(data, modelName);
    })
    .catch(function(err) {
        alert("Loi: " + err.message);
    })
    .finally(function() {
        btn.disabled    = false;
        btn.textContent = "Dự đoán";
    });
}


function showResult(data, modelName) {
    var box = document.getElementById("result-box");

    // Xoa class cu
    box.className = "";

    if (data.error) {
        box.className    = "result-1";
        box.innerHTML    = '<div class="result-label">Loi</div><div class="result-model">' + data.error + '</div>';
        box.style.display = "block";
        return;
    }

    // Gan class mau theo nhan du doan (0 = xanh, 1 = cam, 2 = do)
    box.classList.add("result-" + data.prediction);
    box.innerHTML =
        '<div class="result-label">' + data.label + '</div>' +
        '<div class="result-model">Mô hình: ' + modelName + '</div>';
    box.style.display = "block";
}
