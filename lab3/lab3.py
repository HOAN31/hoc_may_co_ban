import os
import sys
import tempfile

# Hỗ trợ hiển thị tiếng Việt trên Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Tránh cảnh báo quyền ghi thư mục cache của Matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# BÀI 1: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH
# ==============================================================================
# Tìm đường thẳng khớp dữ liệu với tổng sai số nhỏ nhất.
print("=" * 70)
print("BÀI 1: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH (TẠO DỮ LIỆU GIẢ LẬP)")
print("=" * 70)

# 1. Tạo dữ liệu giả lập có mối quan hệ tuyến tính: y = 2X + 1 + nhiễu
np.random.seed(42)
X_synthetic = 2 * np.random.rand(100, 1)  # 100 điểm dữ liệu trong khoảng [0, 2]
noise = np.random.randn(100, 1) * 0.5     # Nhiễu Gauss ngẫu nhiên
y_synthetic = 2 * X_synthetic + 1 + noise # y = 2X + 1 + nhiễu

# 2. Huấn luyện mô hình Linear Regression
lr_synthetic = LinearRegression()
lr_synthetic.fit(X_synthetic, y_synthetic)

# 3. In ra các hệ số của mô hình (coefficient và intercept)
w_learned = lr_synthetic.coef_[0][0]
b_learned = lr_synthetic.intercept_[0]

print(f"Hệ số góc thực tế (Ground Truth Weight - a) : 2.0")
print(f"Hệ số chặn thực tế (Ground Truth Bias - b)   : 1.0")
print(f"Hệ số góc mô hình học được (w / coef_)      : {w_learned:.4f}")
print(f"Hệ số chặn mô hình học được (b / inter_)    : {b_learned:.4f}")
print(f"Phương trình hồi quy tìm được: y = {w_learned:.4f} * X + {b_learned:.4f}")

# ==============================================================================
# BÀI 2: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DIABETES
# ==============================================================================
print("\n" + "=" * 70)
print("BÀI 2: ĐÁNH GIÁ MÔ HÌNH VỚI TẬP DỮ LIỆU DIABETES (ĐẶC TRƯNG BMI)")
print("=" * 70)

# 1. Load dữ liệu diabetes và chỉ lấy 1 đặc trưng duy nhất (BMI - feature index 2)
diabetes = load_diabetes()
feature_idx = 2
feature_name = diabetes.feature_names[feature_idx]  # bmi
X_diabetes = diabetes.data[:, np.newaxis, feature_idx]
y_diabetes = diabetes.target

print(f"Đặc trưng được chọn: '{feature_name}' (Chỉ số khối cơ thể - BMI)")
print(f"Kích thước X: {X_diabetes.shape}, Kích thước y: {y_diabetes.shape}")

# 2. Chia dữ liệu Train / Test (80% Train, 20% Test)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

# 3. Huấn luyện mô hình Hồi quy tuyến tính
lr_diabetes = LinearRegression()
lr_diabetes.fit(X_train_d, y_train_d)

# Dự đoán trên tập Test
y_pred_d = lr_diabetes.predict(X_test_d)

# 4. Tính toán các chỉ số đánh giá: MAE, MSE, RMSE, R-squared
mae_d = mean_absolute_error(y_test_d, y_pred_d)
mse_d = mean_squared_error(y_test_d, y_pred_d)
rmse_d = np.sqrt(mse_d)
r2_d = r2_score(y_test_d, y_pred_d)

print(f"\nKết quả đánh giá trên tập Test:")
print(f"- MAE (Mean Absolute Error) : {mae_d:.4f} (Sai số tuyệt đối trung bình, càng thấp càng tốt)")
print(f"- MSE (Mean Squared Error)  : {mse_d:.4f} (Thước đo độ 'tệ' của mô hình, phạt nặng sai số lớn)")
print(f"- RMSE (Root MSE)           : {rmse_d:.4f} (Giữ nguyên đơn vị gốc của y)")
print(f"- R-squared (R2 Score)      : {r2_d:.4f} ({r2_d * 100:.2f}%)")

# ------------------------------------------------------------------------------
# GIẢI THÍCH CHI TIẾT CÁC CHỈ SỐ:
# ------------------------------------------------------------------------------
# 1. MAE (Mean Absolute Error):
#    - Đo khoảng cách trung bình giữa giá trị thực tế và dự đoán: (1/n) * sum(|y - y_hat|).
# 2. MSE / RMSE:
#    - MSE là thước đo độ "tệ" của mô hình, MSE càng nhỏ mô hình càng chính xác.
#    - RMSE = sqrt(MSE) đưa sai số về cùng đơn vị gốc với biến mục tiêu y.
# 3. R-squared (Hệ số xác định R2):
#    - Thể hiện mô hình giải thích được bao nhiêu % độ biến động của dữ liệu.
#    - ĐÁNH GIÁ MÔ HÌNH:
#      + Thế nào được coi là tốt? R2 > 0.7 (giải thích được > 70% dữ liệu).
#      + Tại sao ở đây R2 chỉ đạt ~ 23.34% (< 70%)?
#        Vì mô hình này chỉ dùng ĐƠN BIẾN (duy nhất 1 đặc trưng BMI), trong thực tế
#        bệnh tiểu đường phụ thuộc vào nhiều yếu tố khác (huyết áp, đường huyết, độ tuổi...)
#        và dữ liệu có nhiều điểm ngoại lệ (Outliers).
#      + CÁCH CẢI THIỆN:
#        Chuyển sang Hồi quy đa biến (thu thập thêm đặc trưng chất lượng như Bài 4),
#        xử lý ngoại lệ (Outliers), và biến đổi log.
# ------------------------------------------------------------------------------

# ==============================================================================
# BÀI 3: TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================
print("\n" + "=" * 70)
print("BÀI 3: TRỰC QUAN HÓA KẾT QUẢ VỚI MATPLOTLIB")
print("=" * 70)

# Vẽ biểu đồ tán xạ (scatter plot) của dữ liệu thực tế và đường thẳng hồi quy
plt.figure(figsize=(10, 6))

# Điểm dữ liệu thực tế
plt.scatter(X_test_d, y_test_d, color="blue", label="Dữ liệu thực tế (Test Data)", alpha=0.7)

# Đường thẳng hồi quy
X_range = np.linspace(X_test_d.min(), X_test_d.max(), 100).reshape(-1, 1)
y_range_pred = lr_diabetes.predict(X_range)
plt.plot(X_range, y_range_pred, color="red", linewidth=2.5, label="Đường hồi quy dự đoán")

plt.title("Bài 3: Hồi quy tuyến tính trên tập Diabetes (BMI vs Tiến triển bệnh)", fontsize=13, fontweight="bold")
plt.xlabel("Chỉ số BMI chuẩn hóa", fontsize=11)
plt.ylabel("Mức độ tiến triển bệnh (Target)", fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)

chart_bai3_path = os.path.join(os.path.dirname(__file__), "bai3_linear_regression.png")
plt.savefig(chart_bai3_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Đã lưu biểu đồ Bài 3 tại: {chart_bai3_path}")

# ==============================================================================
# BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - HỒI QUY ĐA BIẾN (CALIFORNIA HOUSING)
# ==============================================================================
print("\n" + "=" * 70)
print("BÀI 4: HỒI QUY ĐA BIẾN VỚI PIPELINE TRÊN TẬP CALIFORNIA HOUSING")
print("=" * 70)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
try:
    housing = fetch_california_housing(data_home=data_dir)
except Exception:
    housing = fetch_california_housing()

X_house, y_house = housing.data, housing.target
print(f"Số lượng mẫu: {X_house.shape[0]}, Số đặc trưng: {X_house.shape[1]}")
print(f"Các đặc trưng: {housing.feature_names}")

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Pipeline: Preprocess (Scaling) -> Train (Linear Regression)
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)
pipeline.fit(X_train_h, y_train_h)

# Đánh giá trên tập Test
y_pred_h = pipeline.predict(X_test_h)
mse_h = mean_squared_error(y_test_h, y_pred_h)
mae_h = mean_absolute_error(y_test_h, y_pred_h)
r2_h = r2_score(y_test_h, y_pred_h)

print(f"\nKết quả đánh giá mô hình California Housing (Hồi quy đa biến):")
print(f"- MSE  : {mse_h:.4f}")
print(f"- MAE  : {mae_h:.4f}")
print(f"- R2   : {r2_h:.4f} ({r2_h * 100:.2f}% -> Cải thiện rõ rệt so với mô hình 1 biến!)")

# Trực quan hóa
sample_size = 100
indices = np.arange(sample_size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(indices, y_test_h[:sample_size], label="Thực tế (Actual)", color="blue", linewidth=1.5, marker="o", markersize=3)
ax1.plot(indices, y_pred_h[:sample_size], label="Dự đoán (Predicted)", color="red", linewidth=1.5, linestyle="--", marker="x", markersize=4)
ax1.set_title(f"So sánh Giá trị Thực tế vs Dự đoán ({sample_size} mẫu đầu)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Mẫu số", fontsize=11)
ax1.set_ylabel("Giá nhà ($100k)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.scatter(y_test_h[:500], y_pred_h[:500], alpha=0.5, color="teal", label="Điểm dự đoán (500 mẫu)")
min_val = min(y_test_h[:500].min(), y_pred_h[:500].min())
max_val = max(y_test_h[:500].max(), y_pred_h[:500].max())
ax2.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2, label="Đường lý tưởng (y = x)")
ax2.set_title("Scatter Plot: Thực tế vs Dự đoán", fontsize=13, fontweight="bold")
ax2.set_xlabel("Giá trị thực tế", fontsize=11)
ax2.set_ylabel("Giá trị dự đoán", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
chart_bai4_path = os.path.join(os.path.dirname(__file__), "bai4_california_results.png")
plt.savefig(chart_bai4_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Đã lưu biểu đồ Bài 4 tại: {chart_bai4_path}")

# ==============================================================================
# BÀI 5: PHẦN MỞ RỘNG (SO SÁNH VỚI RIDGE REGRESSION)
# ==============================================================================
print("\n" + "=" * 70)
print("BÀI 5: PHẦN MỞ RỘNG - RIDGE REGRESSION (CHUẨN HÓA L2 HẠN CHẾ OVERFITTING)")
print("=" * 70)

pipeline_ridge = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ]
)
pipeline_ridge.fit(X_train_h, y_train_h)
y_pred_ridge = pipeline_ridge.predict(X_test_h)
mse_ridge = mean_squared_error(y_test_h, y_pred_ridge)
r2_ridge = r2_score(y_test_h, y_pred_ridge)

compare_df = pd.DataFrame({
    "Mô hình": ["Linear Regression (Đa biến)", "Ridge Regression (L2 Penalty)"],
    "MSE": [mse_h, mse_ridge],
    "MAE": [mae_h, mean_absolute_error(y_test_h, y_pred_ridge)],
    "R2 Score": [r2_h, r2_ridge]
})
print("Bảng so sánh hiệu năng các mô hình hồi quy:")
print(compare_df.to_string(index=False))

print("\nHoàn thành!")
