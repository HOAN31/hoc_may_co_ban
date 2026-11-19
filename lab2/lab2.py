import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Bài 1: Làm sạch dữ liệu
print("=== Bài 1: Làm sạch dữ liệu ===")
df = pd.DataFrame(
    {
        "Tuổi": [25, 30, np.nan, 35, 40, 55, 28, 33, 60, 38],
        "Thu_nhap": [15, 18, 17, np.nan, 22, 95, 16, 14, 80, 25],
    }
)
df.loc[5, "Thu_nhap"] = 500
# Thêm một outlier đơn giản ở cột Tuổi để minh họa
# Giá trị 120 nằm khá xa so với phần còn lại và có thể bị loại bỏ
# sau khi dùng ngưỡng 3σ.
df.loc[8, "Tuổi"] = 120
print("Dữ liệu gốc:")
print(df)

cleaned_df = df.copy()
for col in cleaned_df.columns:
    mean_value = cleaned_df[col].mean()
    cleaned_df[col] = cleaned_df[col].fillna(mean_value)
    std_value = cleaned_df[col].std(ddof=0)
    if std_value == 0:
        continue
    outlier_mask = (cleaned_df[col] - cleaned_df[col].mean()).abs() > 3 * std_value
    cleaned_df.loc[outlier_mask, col] = mean_value

print("\nDữ liệu sau khi điền giá trị thiếu và xử lý outlier:")
print(cleaned_df)

# Bài 2: Chuẩn hóa dữ liệu (Scaling)
print("\n=== Bài 2: Chuẩn hóa dữ liệu ===")
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

standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)
print("\nKết quả sau khi dùng StandardScaler:")
print(pd.DataFrame(X_standardized, columns=feature_names))

minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
print("\nKết quả sau khi dùng MinMaxScaler:")
print(pd.DataFrame(X_minmax, columns=feature_names))

# Bài 3: Chia tập dữ liệu Train/Test
print("\n=== Bài 3: Chia tập dữ liệu Train/Test ===")
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data,
    diabetes.target,
    test_size=0.2,
    random_state=42,
)
print("Kích thước dữ liệu gốc:", diabetes.data.shape)
print("Kích thước nhãn gốc:", diabetes.target.shape)
print("Kích thước tập Train:", X_train.shape, y_train.shape)
print("Kích thước tập Test:", X_test.shape, y_test.shape)

# Bài 4: Quy trình ML hoàn chỉnh - tiền xử lý và Pipeline
print("\n=== Bài 4: Pipeline với chuẩn hóa và mô hình ===")
digits = load_digits()
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42,
    stratify=digits.target,
)

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
print(f"Độ chính xác trên tập Test: {accuracy:.2f}")

