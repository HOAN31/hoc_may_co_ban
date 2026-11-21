import os
import sys

# Hỗ trợ hiển thị tiếng Việt trên Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ==============================================================================
# BÀI 1: LÀM SẠCH DỮ LIỆU (THEO ĐỀ LAB 2 & VỞ GHI NOTE 2)
# ==============================================================================
# Theo vở ghi:
# - Dữ liệu thô như "Dầu mỏ" => vô giá trị nếu không được tinh chế (tiền xử lý chiếm 80% ML).
# - Xử lý giá trị thiếu (NaN): Điền bằng Mean (trung bình), Median (trung vị), hoặc Mode.
# - Xử lý ngoại lệ (Outliers): Các giá trị cách xa phần còn lại, xử lý bằng 3σ hoặc IQR.
print("=" * 60)
print("=== Bài 1: Làm sạch dữ liệu (Missing Values & Outliers) ===")
print("=" * 60)

df = pd.DataFrame(
    {
        "Tuổi": [25, 30, np.nan, 35, 40, 55, 28, 33, 60, 38],
        "Thu_nhap": [15, 18, 17, np.nan, 22, 95, 16, 14, 80, 25],
    }
)
# Thêm ngoại lệ (outlier) để minh họa
df.loc[5, "Thu_nhap"] = 500  # Thu nhập bất thường
df.loc[8, "Tuổi"] = 120      # Tuổi bất thường (>100)

print("1. Dữ liệu gốc:")
print(df)
print("\nSố lượng giá trị thiếu (NaN) trên từng cột:")
print(df.isnull().sum())

cleaned_df = df.copy()
for col in cleaned_df.columns:
    # 1. Điền giá trị thiếu bằng Mean (hoặc Median như trong vở ghi)
    mean_value = cleaned_df[col].mean()
    cleaned_df[col] = cleaned_df[col].fillna(mean_value)
    
    # 2. Phát hiện và xử lý ngoại lệ theo quy tắc 3-sigma: |x - mean| > 3*std
    std_value = cleaned_df[col].std(ddof=0)
    if std_value == 0:
        continue
    outlier_mask = (cleaned_df[col] - cleaned_df[col].mean()).abs() > 3 * std_value
    cleaned_df.loc[outlier_mask, col] = mean_value

print("\n2. Dữ liệu sau khi điền giá trị thiếu và xử lý outlier (3σ):")
print(cleaned_df)

# ==============================================================================
# BÀI 2: CHUẨN HÓA DỮ LIỆU - SCALING (THEO ĐỀ LAB 2 & VỞ GHI NOTE 2)
# ==============================================================================
# Theo vở ghi:
# - Chuẩn hóa tránh việc cột có giá trị lớn (ví dụ: Thu nhập/Lương) lấn át cột có giá trị nhỏ (Tuổi).
# - StandardScaler: Đưa về phân phối chuẩn (Mean = 0, Std = 1).
# - MinMaxScaler: Đưa về khoảng từ 0 đến 1.
print("\n" + "=" * 60)
print("=== Bài 2: Chuẩn hóa dữ liệu (StandardScaler & MinMaxScaler) ===")
print("=" * 60)

X = np.array(
    [
        [25, 15],
        [30, 28],
        [35, 40],
        [45, 60],
        [50, 70],
        [60, 100],
    ]
)
feature_names = ["Tuổi", "Thu_nhap"]

# StandardScaler
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)
print("\n1. Kết quả sau khi dùng StandardScaler (Mean=0, Std=1):")
print(pd.DataFrame(X_standardized, columns=feature_names))

# MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
print("\n2. Kết quả sau khi dùng MinMaxScaler (về khoảng [0, 1]):")
print(pd.DataFrame(X_minmax, columns=feature_names))

# ==============================================================================
# BÀI 3: CHIA TẬP DỮ LIỆU TRAIN / TEST
# ==============================================================================
# Theo vở ghi:
# - Chia tập dữ liệu (Train 70-80%, Test 20-30%) để đánh giá năng lực thực tế.
# - LƯU Ý CỐT LÕI: Không bao giờ cho máy nhìn thấy tập Test trước để tránh Rò rỉ dữ liệu (Data Leak).
print("\n" + "=" * 60)
print("=== Bài 3: Chia tập dữ liệu Train/Test trên tập Diabetes ===")
print("=" * 60)

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data,
    diabetes.target,
    test_size=0.2,
    random_state=42,
)
print("Kích thước dữ liệu gốc (Features):", diabetes.data.shape)
print("Kích thước nhãn gốc (Target):", diabetes.target.shape)
print("Kích thước tập Train (80%):", X_train.shape, y_train.shape)
print("Kích thước tập Test (20%):", X_test.shape, y_test.shape)

# ==============================================================================
# BÀI 4: QUY TRÌNH ML HOÀN CHỈNH VỚI PIPELINE
# ==============================================================================
print("\n" + "=" * 60)
print("=== Bài 4: Quy trình ML hoàn chỉnh với Pipeline trên tập Digits ===")
print("=" * 60)

digits = load_digits()
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42,
    stratify=digits.target,
)

# Pipeline đóng gói tự động: Chuẩn hóa -> Huấn luyện mô hình
# Tránh hoàn toàn việc rò rỉ dữ liệu giữa tập Train và Test
model_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
model_pipeline.fit(X_digits_train, y_digits_train)
accuracy = model_pipeline.score(X_digits_test, y_digits_test)

print("Kích thước tập Train Digits:", X_digits_train.shape, y_digits_train.shape)
print("Kích thước tập Test Digits:", X_digits_test.shape, y_digits_test.shape)
print(f"Độ chính xác (Accuracy) trên tập Test: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# ==============================================================================
# BÀI 5: GIẢNG VIÊN GIAO THÊM - MÃ HÓA ENCODING (THEO VỞ GHI NOTE 2)
# ==============================================================================
# Theo vở ghi Note 2:
# - Encoding: Chuyển đổi dữ liệu chữ sang số.
# - LabelEncoder: Gán số nguyên cho dữ liệu có thứ bậc (S < M < L < XL) giúp giữ nguyên trật tự kích thước.
# - One-Hot Encoding: Tạo cột giả (0, 1) cho dữ liệu định danh không có thứ bậc (ví dụ màu sắc).
print("\n" + "=" * 60)
print("=== Bài 5: Phần mở rộng theo vở ghi - Mã hóa dữ liệu (Encoding) ===")
print("=" * 60)

kich_thuoc = ["S", "M", "L", "XL", "M", "S"]
df_size = pd.DataFrame({"Kich_Thuoc": kich_thuoc})

encoder = LabelEncoder()
df_size["Encoded"] = encoder.fit_transform(df_size["Kich_Thuoc"])
print("Kết quả mã hóa kích thước quần áo bằng LabelEncoder (giữ trật tự S < M < L < XL):")
print(df_size)

print("\nHoàn thành toàn bộ Lab 2 theo đúng đề bài và vở ghi!")
