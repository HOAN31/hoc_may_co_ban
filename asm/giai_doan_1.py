import os
import sys
import tempfile

# Hỗ trợ hiển thị tiếng Việt trên Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Tạo thư mục lưu trữ ảnh báo cáo
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "titanic.csv")

print("=" * 80)
print("ASSIGNMENT MÔN HỌC MÁY CƠ BẢN - GIAI ĐOẠN 1: MÔ HÌNH CƠ SỞ (TITANIC)")
print("=" * 80)

# ==============================================================================
# NHIỆM VỤ 1: CHỌN VÀ KHÁM PHÁ DỮ LIỆU (EXPLORATORY DATA ANALYSIS - EDA)
# ==============================================================================
# Theo vở ghi Note 1:
# - Phân loại bài toán: Dự đoán nhãn 'Survived' (0: Thiệt mạng, 1: Sống sót)
#   => Bài toán Phân loại nhị phân (Binary Classification) có giám sát (Supervised Learning).
print("\n" + "=" * 80)
print("1. KHÁM PHÁ DỮ LIỆU (SHAPE, INFO, DESCRIBE)")
print("=" * 80)

df_raw = pd.read_csv(DATA_PATH)

print(f"Kích thước dữ liệu (Shape): {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột")
print("\n5 dòng đầu tiên của tập dữ liệu:")
print(df_raw.head())

print("\n--- Thông tin cấu trúc dữ liệu (Info) ---")
df_raw.info()

print("\n--- Thống kê mô tả các đặc trưng số (Describe) ---")
print(df_raw.describe())

print("\n--- Thống kê giá trị thiếu (Missing values - NaN) trên từng cột ---")
missing_count = df_raw.isnull().sum()
missing_percent = (missing_count / len(df_raw)) * 100
missing_df = pd.DataFrame({"Số lượng thiếu": missing_count, "Tỷ lệ (%)": missing_percent.round(2)})
print(missing_df[missing_df["Số lượng thiếu"] > 0])

# Tỷ lệ phân bố biến mục tiêu (Survived)
target_counts = df_raw["Survived"].value_counts()
print(f"\nPhân bố biến mục tiêu Survived:\n{target_counts}")
print(f"-> Tỷ lệ sống sót: {target_counts[1] / len(df_raw) * 100:.2f}% ({target_counts[1]} người)")
print(f"-> Tỷ lệ thiệt mạng: {target_counts[0] / len(df_raw) * 100:.2f}% ({target_counts[0]} người)")

# ==============================================================================
# NHIỆM VỤ 2: LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# ==============================================================================
# Theo vở ghi Note 2:
# 1. Xử lý giá trị thiếu (NaN):
#    - Xóa cột nếu thiếu quá nhiều (> 50-60%): Cột 'Cabin' thiếu tới 77.1% => Xóa bỏ.
#    - Cột 'Age' thiếu 177 giá trị (19.87%): Điền bằng Trung vị (Median) vì Age có phân phối lệch.
#    - Cột 'Embarked' thiếu 2 giá trị (< 1%): Điền bằng Yếu vị (Mode - giá trị xuất hiện nhiều nhất: 'S').
# 2. Xóa các cột định danh không mang giá trị quy luật học máy:
#    - 'PassengerId', 'Name', 'Ticket' (chuỗi ký tự riêng biệt của từng người).
# 3. Xử lý ngoại lệ (Outliers):
#    - Cột 'Fare' (giá vé) có một số vé VIP lên tới 512$ => áp dụng cắt biên (Winsorization theo 99th percentile).
print("\n" + "=" * 80)
print("2. LÀM SẠCH DỮ LIỆU (XỬ LÝ NAN & OUTLIERS THEO VỞ GHI NOTE 2)")
print("=" * 80)

df_clean = df_raw.copy()

# 1. Xóa các cột không cần thiết hoặc thiếu > 60%
df_clean = df_clean.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
print("- Đã loại bỏ các cột không sử dụng / thiếu quá nhiều: 'PassengerId', 'Name', 'Ticket', 'Cabin'")

# 2. Điền giá trị thiếu cho Age bằng Median
median_age = df_clean["Age"].median()
df_clean["Age"] = df_clean["Age"].fillna(median_age)
print(f"- Đã điền thiếu cột 'Age' bằng Trung vị (Median) = {median_age:.1f} tuổi")

# 3. Điền giá trị thiếu cho Embarked bằng Mode
mode_embarked = df_clean["Embarked"].mode()[0]
df_clean["Embarked"] = df_clean["Embarked"].fillna(mode_embarked)
print(f"- Đã điền thiếu cột 'Embarked' bằng Yếu vị (Mode) = '{mode_embarked}'")

# 4. Điền thiếu Fare (nếu có) bằng Median
median_fare = df_clean["Fare"].median()
df_clean["Fare"] = df_clean["Fare"].fillna(median_fare)

# 5. Xử lý ngoại lệ (Outliers) cho cột 'Fare' bằng phương pháp cắt biên (Winsorization tại percentile 99)
fare_cap = df_clean["Fare"].quantile(0.99)
outlier_fare_count = (df_clean["Fare"] > fare_cap).sum()
df_clean.loc[df_clean["Fare"] > fare_cap, "Fare"] = fare_cap
print(f"- Đã xử lý {outlier_fare_count} điểm ngoại lệ ở cột 'Fare' bằng cắt biên tại mức 99%: {fare_cap:.2f}$")

print("\nKiểm tra lại dữ liệu sau làm sạch (Tổng số NaN còn lại):", df_clean.isnull().sum().sum())

# ==============================================================================
# NHIỆM VỤ 3: TIỀN XỬ LÝ & CHIA TẬP DỮ LIỆU (TRAIN / TEST SPLIT)
# ==============================================================================
# Theo vở ghi Note 2:
# - Chia tập dữ liệu (Train 80% - Test 20%) cố định random_state=42.
# - Rò rỉ dữ liệu (Data Leak): Tuyệt đối không cho máy nhìn thấy tập test trước,
#   phải dùng Pipeline/ColumnTransformer để fit trên Train và transform trên Test.
# - Chuẩn hóa (Scaling): StandardScaler đưa Age, Fare về (Mean=0, Std=1).
# - Mã hóa (Encoding): One-Hot Encoding cho Sex, Embarked, Pclass.
print("\n" + "=" * 80)
print("3. TIỀN XỬ LÝ (MÃ HÓA, CHUẨN HÓA & CHIA TRAIN/TEST)")
print("=" * 80)

# Tách biến đặc trưng X và nhãn mục tiêu y
X = df_clean.drop(columns=["Survived"])
y = df_clean["Survived"]

# Xác định danh sách các cột số và cột phân loại
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

print(f"Đặc trưng số (Numeric): {numeric_features}")
print(f"Đặc trưng phân loại (Categorical): {categorical_features}")

# Chia tập Train (80%) và Test (20%) có phân tầng (stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nKích thước tập Train: {X_train.shape} ({len(X_train)} mẫu)")
print(f"Kích thước tập Test : {X_test.shape} ({len(X_test)} mẫu)")

# Xây dựng ColumnTransformer để tiền xử lý chuẩn mực, chống rò rỉ dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ]
)

# ==============================================================================
# NHIỆM VỤ 4: HUẤN LUYỆN MÔ HÌNH CƠ SỞ (BASE MODEL - LOGISTIC REGRESSION)
# ==============================================================================
# Theo vở ghi Note 4:
# - Logistic Regression dùng hàm Sigmoid để nén tổ hợp tuyến tính thành xác suất [0, 1].
# - Ngưỡng quyết định mặc định là 0.5 (P >= 0.5 => lớp 1: Sống sót).
print("\n" + "=" * 80)
print("4. HUẤN LUYỆN MÔ HÌNH CƠ SỞ (LOGISTIC REGRESSION QUA PIPELINE)")
print("=" * 80)

base_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ]
)

# Huấn luyện trên tập Train
base_model.fit(X_train, y_train)
print("Huấn luyện thành công mô hình cơ sở Logistic Regression!")

# ==============================================================================
# NHIỆM VỤ 5: ĐÁNH GIÁ BAN ĐẦU & TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================
# Theo vở ghi Note 4:
# - Accuracy: Độ chính xác tổng thể.
# - Precision: "Không bắt nhầm" (Trong số người đoán Sống sót, bao nhiêu % sống sót thật).
# - Recall: "Không bỏ sót" (Trong tất cả người sống sót thực tế, mô hình nhận diện được bao nhiêu %).
# - F1-Score: Trung bình điều hòa giữa Precision và Recall.
# - Confusion Matrix: Ma trận 2x2 gồm TP, TN, FP, FN.
# - ROC-AUC: Đo năng lực phân tách tổng thể.
print("\n" + "=" * 80)
print("5. ĐÁNH GIÁ MÔ HÌNH CƠ SỞ TRÊN TẬP TEST")
print("=" * 80)

y_pred = base_model.predict(X_test)
y_prob = base_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("KẾT QUẢ CÁC CHỈ SỐ ĐÁNH GIÁ CƠ BẢN TRÊN TẬP TEST:")
print(f"1. Accuracy (Độ chính xác)  : {acc:.4f} ({acc * 100:.2f}%)")
print(f"2. Precision (Độ chuẩn xác) : {prec:.4f} ('Không bắt nhầm': {prec * 100:.2f}%)")
print(f"3. Recall (Độ thu hồi)      : {rec:.4f} ('Không bỏ sót': {rec * 100:.2f}%)")
print(f"4. F1-Score (Cân bằng)      : {f1:.4f}")
print(f"5. ROC-AUC (Năng lực phân tách): {auc:.4f}")

print("\n--- Báo cáo phân loại chi tiết (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=["Thiệt mạng (0)", "Sống sót (1)"]))

# ------------------------------------------------------------------------------
# MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# ------------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("--- Phân tích Ma trận nhầm lẫn (Confusion Matrix) ---")
print(cm)
print(f"- TN (True Negative - Đoán đúng thiệt mạng) : {tn} người")
print(f"- FP (False Positive - Bắt nhầm sống sót)   : {fp} người")
print(f"- FN (False Negative - Bỏ sót sống sót)     : {fn} người")
print(f"- TP (True Positive - Đoán đúng sống sót)   : {tp} người")

# ==============================================================================
# XUẤT CÁC ĐỒ THỊ TRỰC QUAN HÓA CHO BÁO CÁO GIAI ĐOẠN 1
# ==============================================================================
print("\n" + "=" * 80)
print("XUẤT CÁC BIỂU ĐỒ TRỰC QUAN HÓA (LƯU VÀO THƯ MỤC images/)")
print("=" * 80)

# Đồ thị 1: Phân tích tỷ lệ sống sót theo Giới tính và Hạng vé
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Giới tính vs Sống sót
sex_surv = df_raw.groupby("Sex")["Survived"].mean() * 100
ax1.bar(sex_surv.index, sex_surv.values, color=["lightcoral", "cornflowerblue"], edgecolor="black")
ax1.set_title("Tỷ lệ sống sót theo Giới tính", fontsize=13, fontweight="bold")
ax1.set_ylabel("Tỷ lệ sống sót (%)", fontsize=11)
for i, v in enumerate(sex_surv.values):
    ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax1.set_ylim(0, 100)
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Hạng vé vs Sống sót
pclass_surv = df_raw.groupby("Pclass")["Survived"].mean() * 100
ax2.bar(["Hạng 1", "Hạng 2", "Hạng 3"], pclass_surv.values, color=["gold", "mediumseagreen", "tomato"], edgecolor="black")
ax2.set_title("Tỷ lệ sống sót theo Hạng vé (Pclass)", fontsize=13, fontweight="bold")
ax2.set_ylabel("Tỷ lệ sống sót (%)", fontsize=11)
for i, v in enumerate(pclass_surv.values):
    ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax2.set_ylim(0, 100)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
chart_eda_path = os.path.join(IMAGES_DIR, "1_ty_le_song_sot_eda.png")
plt.savefig(chart_eda_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"1. Đã lưu biểu đồ phân tích EDA tại: {chart_eda_path}")

# Đồ thị 2: Ma trận nhầm lẫn (Confusion Matrix Display)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Thiệt mạng (0)", "Sống sót (1)"])
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("Ma trận nhầm lẫn - Mô hình cơ sở (Logistic Regression)", fontsize=12, fontweight="bold")
chart_cm_path = os.path.join(IMAGES_DIR, "2_confusion_matrix.png")
plt.savefig(chart_cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"2. Đã lưu biểu đồ Ma trận nhầm lẫn tại: {chart_cm_path}")

# Đồ thị 3: Đường cong ROC-AUC
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_estimator(
    base_model,
    X_test,
    y_test,
    name=f"Logistic Regression (AUC = {auc:.3f})",
    ax=ax,
)
ax.plot([0, 1], [0, 1], "k--", label="Đường ngẫu nhiên (AUC = 0.5)")
ax.set_title("Đường cong ROC - Logistic Regression (Titanic)", fontsize=13, fontweight="bold")
ax.set_xlabel("Tỷ lệ dương tính giả (FPR - False Positive Rate)", fontsize=11)
ax.set_ylabel("Tỷ lệ dương tính thật (TPR - Recall / True Positive Rate)", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

chart_roc_path = os.path.join(IMAGES_DIR, "3_roc_curve.png")
plt.savefig(chart_roc_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"3. Đã lưu biểu đồ Đường cong ROC tại: {chart_roc_path}")

print("\n" + "=" * 80)
print("HOÀN THÀNH XUẤT SẮC GIAI ĐOẠN 1 CỦA ASSIGNMENT!")
print("=" * 80)
