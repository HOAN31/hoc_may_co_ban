import os
import sys
import tempfile

# Hỗ trợ hiển thị tiếng Việt trên Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Thiết lập thư mục cache Matplotlib để tránh cảnh báo quyền ghi
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# BÀI 1: XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION (BREAST CANCER)
# ==============================================================================
print("=" * 70)
print("BÀI 1: XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION TRÊN TẬP BREAST CANCER")
print("=" * 70)

# 1. Load dữ liệu
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target
target_names_cancer = cancer.target_names  # ['malignant' (ác tính - 0), 'benign' (lành tính - 1)]

print(f"Số lượng mẫu: {X_cancer.shape[0]}, Số đặc trưng: {X_cancer.shape[1]}")
print(f"Các lớp nhãn: {target_names_cancer[0]} (0 - Ác tính), {target_names_cancer[1]} (1 - Lành tính)")
print(f"Tỷ lệ phân bố lớp: {np.bincount(y_cancer)} -> Ác tính: {np.bincount(y_cancer)[0]}, Lành tính: {np.bincount(y_cancer)[1]}")

# 2. Chia dữ liệu Train / Test (80% Train, 20% Test)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)
print(f"Kích thước tập Train: {X_train_c.shape}, Kích thước tập Test: {X_test_c.shape}")

# 3. Chuẩn hóa dữ liệu (Scaling)
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# 4. Khởi tạo và huấn luyện mô hình Logistic Regression
# Logistic Regression dùng hàm Sigmoid ép giá trị vào [0, 1] để phân loại với ngưỡng quyết định 0.5
lr_cancer = LogisticRegression(max_iter=1000, random_state=42)
lr_cancer.fit(X_train_c_scaled, y_train_c)
print("Huấn luyện mô hình Logistic Regression thành công!")

# ==============================================================================
# BÀI 2: ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI
# ==============================================================================
print("\n" + "=" * 70)
print("BÀI 2: ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI TRÊN TẬP TEST")
print("=" * 70)

# Dự đoán trên tập Test
y_pred_c = lr_cancer.predict(X_test_c_scaled)
y_prob_c = lr_cancer.predict_proba(X_test_c_scaled)[:, 1]  # Xác suất thuộc lớp 1 (Lành tính)

# Tính các chỉ số cơ bản
acc_c = accuracy_score(y_test_c, y_pred_c)
prec_c = precision_score(y_test_c, y_pred_c)
rec_c = recall_score(y_test_c, y_pred_c)
f1_c = f1_score(y_test_c, y_pred_c)
auc_c = roc_auc_score(y_test_c, y_prob_c)

print(f"1. Accuracy (Độ chính xác)  : {acc_c:.4f} ({acc_c * 100:.2f}%)")
print(f"2. Precision (Độ chuẩn xác) : {prec_c:.4f} -> 'Không bắt nhầm'")
print(f"3. Recall (Độ thu hồi)      : {rec_c:.4f} -> 'Không bỏ sót'")
print(f"   F1-Score (Cân bằng)      : {f1_c:.4f}")
print(f"   AUC (Chỉ số ROC-AUC)     : {auc_c:.4f}")

# Báo cáo phân loại chi tiết
print("\n--- Báo cáo chi tiết phân loại (Classification Report) ---")
print(classification_report(y_test_c, y_pred_c, target_names=target_names_cancer))

# ==============================================================================
# BÀI 3: MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# ==============================================================================
print("=" * 70)
print("BÀI 3: MA TRẬN NHẦM LẪN (CONFUSION MATRIX)")
print("=" * 70)

# Tạo ma trận nhầm lẫn
cm_c = confusion_matrix(y_test_c, y_pred_c)
print("Ma trận nhầm lẫn (Confusion Matrix 2x2):")
print(cm_c)

# ------------------------------------------------------------------------------
# GIẢI THÍCH CHI TIẾT 4 THÀNH PHẦN THEO ĐÚNG VỞ GHI BÀI HỌC:
# ------------------------------------------------------------------------------
# Ma trận nhầm lẫn dạng:
#                   Dự đoán 0 (Malignant)   Dự đoán 1 (Benign)
# Thực tế 0 (Malignant)     TN                     FP
# Thực tế 1 (Benign)        FN                     TP
#
# Ở đây ta xét Lớp Dương tính (Positive Class = 1: Lành tính - Benign)
#          và Lớp Âm tính (Negative Class = 0: Ác tính - Malignant):
#
# 1. TN (True Negative): Thực tế là Ác tính (0) và mô hình dự đoán đúng là Ác tính (0).
#    -> Dự đoán chính xác lớp âm tính.
# 2. FP (False Positive - Dương tính giả / "Bắt nhầm"): Thực tế là Ác tính (0) nhưng bị đoán nhầm thành Lành tính (1).
# 3. FN (False Negative - Âm tính giả / "Bỏ sót"): Thực tế là Lành tính (1) nhưng bị đoán nhầm thành Ác tính (0).
# 4. TP (True Positive): Thực tế là Lành tính (1) và mô hình dự đoán đúng là Lành tính (1).
#    -> Dự đoán chính xác lớp dương tính.
#
# Ý nghĩa khẩu quyết nhớ nhanh từ vở ghi:
# - TP & TN: Số lượng mẫu dự đoán ĐÚNG.
# - Precision = TP / (TP + FP): "Không bắt nhầm" (Trong những ca đoán Dương tính, bao nhiêu % là đúng thật).
# - Recall = TP / (TP + FN): "Không bỏ sót" (Trong tất cả ca Dương tính ngoài đời, phát hiện được bao nhiêu %).
# ------------------------------------------------------------------------------

tn, fp, fn, tp = cm_c.ravel()
print(f"\nPhân rã chi tiết các thành phần:")
print(f"- TN (True Negative - Đúng ca ác tính)     : {tn}")
print(f"- FP (False Positive - Bắt nhầm ca ác tính): {fp}")
print(f"- FN (False Negative - Bỏ sót ca lành tính): {fn}")
print(f"- TP (True Positive - Đúng ca lành tính)   : {tp}")

# Trực quan hóa ma trận nhầm lẫn bằng ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_c, display_labels=target_names_cancer)
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("Bài 3: Ma trận nhầm lẫn (Confusion Matrix) - Breast Cancer", fontsize=13, fontweight="bold")

chart_cm_path = os.path.join(os.path.dirname(__file__), "bai3_confusion_matrix.png")
plt.savefig(chart_cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nĐã lưu biểu đồ Ma trận nhầm lẫn tại: {chart_cm_path}")

# ==============================================================================
# BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - LOGISTIC REGRESSION TRÊN TẬP DIGITS
# ==============================================================================
print("\n" + "=" * 70)
print("BÀI 4: QUY TRÌNH HOÀN CHỈNH - PHÂN LOẠI ĐA LỚP TRÊN TẬP DIGITS (0-9)")
print("=" * 70)

# 1. Load data
digits = load_digits()
X_digits, y_digits = digits.data, digits.target
print(f"Tập Digits: {X_digits.shape[0]} ảnh chữ số, mỗi ảnh {X_digits.shape[1]} đặc trưng (8x8 pixels)")
print(f"Các lớp nhãn: {np.unique(y_digits)} (10 chữ số từ 0 đến 9)")

# Chia Train / Test (80% Train, 20% Test)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42, stratify=y_digits
)

# 2. Xây dựng Pipeline: Preprocess (StandardScaler) -> Logistic Regression (Multi-class)
digits_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]
)

# Huấn luyện mô hình
digits_pipeline.fit(X_train_d, y_train_d)
print("Đã hoàn thành huấn luyện Pipeline Logistic Regression trên tập Digits!")

# 3. Đánh giá mô hình trên tập Test
y_pred_d = digits_pipeline.predict(X_test_d)
acc_digits = accuracy_score(y_test_d, y_pred_d)
print(f"\nĐộ chính xác (Accuracy) trên tập Test Digits: {acc_digits:.4f} ({acc_digits * 100:.2f}%)")

print("\n--- Báo cáo phân loại chi tiết (Classification Report) - Digits ---")
print(classification_report(y_test_d, y_pred_d))

# 4. Tìm và in ra 5 mẫu dự đoán SAI của mô hình
misclassified_indices = np.where(y_test_d != y_pred_d)[0]
num_errors = len(misclassified_indices)
print(f"Tổng số mẫu dự đoán sai: {num_errors} / {len(y_test_d)} mẫu test.")

print("\n--- Danh sách 5 mẫu dự đoán SAI đầu tiên (Thực tế vs Dự đoán) ---")
table_rows = []
for i, idx in enumerate(misclassified_indices[:5]):
    true_label = y_test_d[idx]
    pred_label = y_pred_d[idx]
    table_rows.append({"Mẫu số": idx, "Nhãn thực tế": true_label, "Mô hình dự đoán": pred_label})
    print(f"Mẫu #{idx:3d} | Nhãn thực tế: {true_label} -> Mô hình dự đoán nhầm thành: {pred_label}")

# Trực quan hóa 5 mẫu bị đoán sai thành hình ảnh để quan sát
fig, axes = plt.subplots(1, min(5, num_errors), figsize=(12, 3))
for i, idx in enumerate(misclassified_indices[:5]):
    ax = axes[i] if min(5, num_errors) > 1 else axes
    ax.imshow(X_test_d[idx].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"Thực: {y_test_d[idx]} | Đoán: {y_pred_d[idx]}", color="red", fontweight="bold")
    ax.axis("off")

plt.suptitle("5 mẫu chữ số bị mô hình dự đoán SAI trên tập Digits", fontsize=14, fontweight="bold", y=1.05)
chart_digits_path = os.path.join(os.path.dirname(__file__), "bai4_digits_misclassified.png")
plt.savefig(chart_digits_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Đã lưu biểu đồ các mẫu đoán sai tại: {chart_digits_path}")

# ==============================================================================
# BÀI 5: GIẢNG VIÊN GIAO THÊM - ĐƯỜNG CONG ROC VÀ CHỈ SỐ AUC
# ==============================================================================
print("\n" + "=" * 70)
print("BÀI 5: PHẦN MỞ RỘNG - VẼ ĐƯỜNG CONG ROC & CHỈ SỐ AUC (THEO VỞ GHI)")
print("=" * 70)

# Như ghi chép trong vở ghi:
# "Cần vẽ biểu đồ => sử dụng đường cong ROC và chỉ số AUC: Dùng để đo lường khả năng
# phân tích tổng thể giữa các lớp (AUC càng gần 1 thì càng tốt)"
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    lr_cancer,
    X_test_c_scaled,
    y_test_c,
    name="Logistic Regression",
    ax=ax,
)
ax.plot([0, 1], [0, 1], "k--", label="Đường ngẫu nhiên (AUC = 0.5)")
ax.set_title("Bài 5: Đường cong ROC - Receiver Operating Characteristic (Breast Cancer)", fontsize=13, fontweight="bold")
ax.set_xlabel("Tỷ lệ dương tính giả (FPR)", fontsize=11)
ax.set_ylabel("Tỷ lệ dương tính thật (TPR)", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

chart_roc_path = os.path.join(os.path.dirname(__file__), "bai5_roc_curve.png")
plt.savefig(chart_roc_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Đã lưu biểu đồ đường cong ROC tại: {chart_roc_path}")
print(f"Chỉ số AUC thực tế đạt được: {auc_c:.4f} (rất gần 1.0, cho thấy khả năng phân tách cực kỳ xuất sắc!)")

print("\n" + "=" * 70)
print("HOÀN THÀNH TOÀN BỘ LAB 4 THEO ĐÚNG ĐỀ BÀI VÀ VỞ GHI!")
print("=" * 70)
