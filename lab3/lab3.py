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
print("=" * 60)
print("BÀI 1: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH")
print("=" * 60)

# 1. Tạo dữ liệu giả lập có mối quan hệ tuyến tính: y = 2X + 1 + nhiễu
np.random.seed(42)
X_synthetic = 2 * np.random.rand(100, 1)  # 100 điểm dữ liệu trong khoảng [0, 2]
noise = np.random.randn(100, 1) * 0.5  # Nhiễu Gauss ngẫu nhiên
y_synthetic = 2 * X_synthetic + 1 + noise  # y = 2X + 1 + nhiễu

# 2. Huấn luyện mô hình Linear Regression
lr_synthetic = LinearRegression()
lr_synthetic.fit(X_synthetic, y_synthetic)

# 3. In ra các hệ số của mô hình (coefficient và intercept)
w_learned = lr_synthetic.coef_[0][0]
b_learned = lr_synthetic.intercept_[0]

print(f"Hệ số góc thực tế (Ground Truth Weight) : 2.0")
print(f"Hệ số chặn thực tế (Ground Truth Bias)   : 1.0")
print(f"Hệ số góc mô hình học được (w / coef_)   : {w_learned:.4f}")
print(f"Hệ số chặn mô hình học được (b / inter_) : {b_learned:.4f}")
print(f"Phương trình hồi quy tìm được: y = {w_learned:.4f} * X + {b_learned:.4f}")

# ==============================================================================
# BÀI 2: ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
print("\n" + "=" * 60)
print("BÀI 2: ĐÁNH GIÁ MÔ HÌNH VỚI TẬP DỮ LIỆU DIABETES")
print("=" * 60)

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

# 4. Tính toán các chỉ số đánh giá: MAE, MSE, R-squared
mae_d = mean_absolute_error(y_test_d, y_pred_d)
mse_d = mean_squared_error(y_test_d, y_pred_d)
rmse_d = np.sqrt(mse_d)
r2_d = r2_score(y_test_d, y_pred_d)

print(f"\nKết quả đánh giá trên tập Test:")
print(f"- MAE (Mean Absolute Error) : {mae_d:.4f}")
print(f"- MSE (Mean Squared Error)  : {mse_d:.4f}")
print(f"- RMSE (Root MSE)           : {rmse_d:.4f}")
print(f"- R-squared (R2 Score)      : {r2_d:.4f}")

# ------------------------------------------------------------------------------
# GIẢI THÍCH Ý NGHĨA CỦA CÁC CHỈ SỐ ĐÁNH GIÁ (THEO YÊU CẦU ĐỀ BÀI):
# ------------------------------------------------------------------------------
# 1. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình):
#    - Đo lường khoảng cách trung bình giữa giá trị dự đoán và giá trị thực tế: (1/n) * sum(|y_i - y_hat_i|).
#    - Ý nghĩa: Cho biết trung bình mô hình dự đoán lệch bao nhiêu đơn vị so với thực tế.
#    - Đặc điểm: Trực quan, cùng đơn vị đo với y, không phóng đại các điểm dị biệt (outliers).
#
# 2. MSE (Mean Squared Error - Sai số bình phương trung bình):
#    - Đo lường trung bình bình phương khoảng cách giữa thực tế và dự đoán: (1/n) * sum((y_i - y_hat_i)^2).
#    - Ý nghĩa: Đánh giá độ phân tán của sai số.
#    - Đặc điểm: Phạt rất nặng các sai số lớn do có phép bình phương, rất nhạy với các điểm dị biệt (outliers).
#
# 3. R-squared (Hệ số xác định - R2 Score):
#    - Đo lường tỷ lệ phần trăm độ biến thiên của biến phụ thuộc (y) được giải thích bởi mô hình:
#      R2 = 1 - (SS_res / SS_tot).
#    - Ý nghĩa: Thể hiện độ "khớp" của mô hình với dữ liệu.
#    - Giá trị: Tối đa là 1 (mô hình khớp hoàn hảo). Càng gần 1 thì mô hình giải thích dữ liệu càng tốt.
#      Nếu R2 gần 0, mô hình không tốt hơn mức dự đoán bằng giá trị trung bình. R2 có thể âm nếu mô hình quá tệ.

# ==============================================================================
# BÀI 3: TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================
print("\n" + "=" * 60)
print("BÀI 3: TRỰC QUAN HÓA KẾT QUẢ VỚI MATPLOTLIB")
print("=" * 60)

# Vẽ biểu đồ tán xạ (scatter plot) của dữ liệu thực tế và đường thẳng hồi quy
plt.figure(figsize=(10, 6))

# Vẽ các điểm dữ liệu thực tế trên tập Test
plt.scatter(X_test_d, y_test_d, color="blue", label="Dữ liệu thực tế (Test Data)", alpha=0.7)

# Sắp xếp các điểm X để vẽ đường hồi quy mượt mà
X_range = np.linspace(X_test_d.min(), X_test_d.max(), 100).reshape(-1, 1)
y_range_pred = lr_diabetes.predict(X_range)

# Vẽ đường hồi quy (đường dự đoán)
plt.plot(X_range, y_range_pred, color="red", linewidth=2.5, label="Đường hồi quy (Mô hình)")

plt.title("Bài 3: Hồi quy tuyến tính trên tập dữ liệu Diabetes (Feature: BMI)", fontsize=14, fontweight="bold")
plt.xlabel("Chỉ số BMI chuẩn hóa (Feature)", fontsize=12)
plt.ylabel("Mức độ tiến triển bệnh (Target)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)

# Lưu đồ thị ra file ảnh
chart_bai3_path = os.path.join(os.path.dirname(__file__), "bai3_linear_regression.png")
plt.savefig(chart_bai3_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Đã lưu biểu đồ Bài 3 tại: {chart_bai3_path}")

# ==============================================================================
# BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - REGRESSION (CALIFORNIA HOUSING)
# ==============================================================================
print("\n" + "=" * 60)
print("BÀI 4: QUY TRÌNH ML HOÀN CHỈNH VỚI DỮ LIỆU CALIFORNIA HOUSING")
print("=" * 60)

# 1. Load dữ liệu (sử dụng thư mục data cục bộ đã tải)
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
try:
    housing = fetch_california_housing(data_home=data_dir)
except Exception:
    housing = fetch_california_housing()

X_house = housing.data
y_house = housing.target
feature_names_house = housing.feature_names

print(f"Số lượng mẫu dữ liệu: {X_house.shape[0]}, Số đặc trưng: {X_house.shape[1]}")
print(f"Danh sách đặc trưng: {feature_names_house}")

# Chia Train / Test
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# 2 & 3. Preprocess (Scaling) và Huấn luyện mô hình thông qua Pipeline
# Pipeline thực hiện chuẩn hóa dữ liệu (StandardScaler) sau đó đưa vào LinearRegression
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)

# Huấn luyện
pipeline.fit(X_train_h, y_train_h)

# 4. Đánh giá mô hình (Evaluate với MSE, MAE, R2)
y_pred_h = pipeline.predict(X_test_h)
mse_h = mean_squared_error(y_test_h, y_pred_h)
mae_h = mean_absolute_error(y_test_h, y_pred_h)
r2_h = r2_score(y_test_h, y_pred_h)

print(f"\nKết quả đánh giá mô hình California Housing trên tập Test:")
print(f"- MSE (Mean Squared Error)  : {mse_h:.4f}")
print(f"- MAE (Mean Absolute Error) : {mae_h:.4f}")
print(f"- R2 Score                  : {r2_h:.4f}")

# 5. Visualize một phần kết quả
# Trực quan hóa 100 mẫu đầu tiên: So sánh giá trị Thực tế vs Dự đoán
sample_size = 100
indices = np.arange(sample_size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Biểu đồ đường so sánh 100 mẫu đầu tiên
ax1.plot(indices, y_test_h[:sample_size], label="Giá trị thực tế (Actual)", color="blue", linewidth=1.5, marker="o", markersize=3)
ax1.plot(indices, y_pred_h[:sample_size], label="Giá trị dự đoán (Predicted)", color="red", linewidth=1.5, linestyle="--", marker="x", markersize=4)
ax1.set_title(f"So sánh Giá trị Thực tế vs Dự đoán ({sample_size} mẫu đầu)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Mẫu số", fontsize=11)
ax1.set_ylabel("Giá nhà (x 100,000 $)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)

# Biểu đồ 2: Scatter plot Thực tế vs Dự đoán kèm đường chuẩn y = x
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
# BÀI 5: GIẢNG VIÊN GIAO THÊM (MỞ RỘNG & SO SÁNH MÔ HÌNH)
# ==============================================================================
print("\n" + "=" * 60)
print("BÀI 5: PHẦN MỞ RỘNG (SO SÁNH MÔ HÌNH RIDGE REGRESSION VỚI REGULARIZATION)")
print("=" * 60)

# Thử nghiệm mô hình hồi quy Ridge (áp dụng L2 Regularization để hạn chế overfitting)
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

# Bảng so sánh giữa Linear Regression và Ridge Regression
compare_df = pd.DataFrame({
    "Mô hình": ["Linear Regression (Pipeline)", "Ridge Regression (L2 Penalty)"],
    "MSE": [mse_h, mse_ridge],
    "MAE": [mae_h, mean_absolute_error(y_test_h, y_pred_ridge)],
    "R2 Score": [r2_h, r2_ridge]
})

print("Bảng so sánh hiệu năng các mô hình hồi quy:")
print(compare_df.to_string(index=False))

# In một bảng mẫu dự đoán thực tế vs dự đoán của 5 căn nhà
sample_table = pd.DataFrame({
    "Thực tế ($100k)": np.round(y_test_h[:5], 3),
    "Dự đoán ($100k)": np.round(y_pred_h[:5], 3),
    "Sai lệch tuyệt đối": np.round(np.abs(y_test_h[:5] - y_pred_h[:5]), 3),
})
print("\nBảng dự đoán mẫu của 5 căn nhà đầu tiên:")
print(sample_table.to_string(index=False))
print("\nHoàn thành toàn bộ Lab 3!")
