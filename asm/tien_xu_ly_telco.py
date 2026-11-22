# -*- coding: utf-8 -*-
"""
BÀI TẬP LỚN / ASSIGNMENT GIAI ĐOẠN 1 - MÔN HỌC MÁY CƠ BẢN (ITA202)
ĐỀ TÀI 11: PHÂN TÍCH HÀNH VI VÀ DỰ BÁO RỜI BỎ DỊCH VỤ CỦA KHÁCH HÀNG (TELCO CUSTOMER CHURN)

Mã nguồn: https://github.com/HOAN31/hoc_may_co_ban
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Cấu hình môi trường và encoding hiển thị tiếng Việt mượt mà
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

def main():
    print("=" * 80)
    print("ASSIGNMENT 1 - HỌC MÁY CƠ BẢN (ITA202)")
    print("ĐỀ TÀI 11: TELCO CUSTOMER CHURN")
    print("=" * 80)

    # -------------------------------------------------------------
    # BƯỚC 1: ĐỌC VÀ KHÁM PHÁ DỮ LIỆU GỐC (EDA)
    # -------------------------------------------------------------
    raw_data_path = os.path.join("asm", "data", "telco_customer_churn.csv")
    if not os.path.exists(raw_data_path):
        raw_data_path = os.path.join("asm", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {raw_data_path}")

    print(f"\n[BƯỚC 1 - EDA] Đọc dữ liệu gốc từ: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    print(f"- Kích thước tập dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"- Danh sách 21 thuộc tính ban đầu:\n  {list(df.columns)}")
    print("\n- Cấu trúc và kiểu dữ liệu:")
    df.info()
    print("\n- Thống kê mô tả:")
    print(df.describe(include="all").transpose().to_string())

    # Thống kê phân phối nhãn mục tiêu Churn
    churn_counts = df["Churn"].value_counts()
    churn_ratio = df["Churn"].value_counts(normalize=True) * 100
    print("\n- Phân phối nhãn mục tiêu (Churn):")
    print(f"  + Không rời mạng (No) : {churn_counts['No']} khách hàng ({churn_ratio['No']:.2f}%)")
    print(f"  + Rời mạng (Yes)       : {churn_counts['Yes']} khách hàng ({churn_ratio['Yes']:.2f}%)")
    print("  => Dữ liệu có sự chênh lệch lớp (~73.5% vs ~26.5%), cần đánh giá bằng Precision, Recall, F1, ROC-AUC thay vì chỉ dùng Accuracy.")

    # -------------------------------------------------------------
    # BƯỚC 2: LOẠI BỎ THUỘC TÍNH KHÔNG MANG GIÁ TRỊ DỰ BÁO
    # -------------------------------------------------------------
    print("\n[BƯỚC 2 - LÀM SẠCH] Xử lý thuộc tính thừa:")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("  -> Đã loại bỏ cột 'customerID' (mã định danh ngẫu nhiên, không chứa quy luật học máy).")

    # -------------------------------------------------------------
    # BƯỚC 3: XỬ LÝ DỮ LIỆU KHUYẾT THIẾU (MISSING VALUES)
    # -------------------------------------------------------------
    print("\n[BƯỚC 3 - XỬ LÝ MISSING] Phát hiện và xử lý dữ liệu khuyết thiếu / sai kiểu:")
    # Cột TotalCharges ban đầu bị đọc là kiểu object/string do có 11 dòng khoảng trắng ' '
    space_mask = df["TotalCharges"] == " "
    num_space = space_mask.sum()
    print(f"  - Phát hiện {num_space} dòng có TotalCharges là khoảng trắng rỗng (' ').")
    if num_space > 0:
        sample_tenure = df.loc[space_mask, "tenure"].unique()
        print(f"  - Khảo sát đặc trưng: Toàn bộ {num_space} khách hàng này đều có tenure = {sample_tenure} (mới đăng ký tháng đầu tiên).")
        print("  -> Ép kiểu sang float bằng pd.to_numeric() và điền giá trị hằng số 0.0 hợp lý nghiệp vụ.")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    print(f"  - Kiểm tra lại sau xử lý: {df.isnull().sum().sum()} giá trị NaN còn lại.")
    print("\n  - Thống kê mô tả các đặc trưng số sau khi làm sạch:")
    print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().to_string())

    # -------------------------------------------------------------
    # BƯỚC 4: MÃ HÓA BIẾN MỤC TIÊU VÀ BIẾN NHỊ PHÂN (ENCODING)
    # -------------------------------------------------------------
    print("\n[BƯỚC 4 - ENCODING] Mã hóa biến mục tiêu Churn và các biến nhị phân:")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    print("  - Churn: 'No' -> 0, 'Yes' -> 1")

    # Mã hóa nhị phân
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"No": 0, "Yes": 1})
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    print(f"  - Đã chuyển đổi sang số 0/1 cho các cột nhị phân: gender, {', '.join(binary_cols)}")

    # -------------------------------------------------------------
    # BƯỚC 5: MÃ HÓA CÁC BIẾN DANH MỤC (ONE-HOT ENCODING)
    # -------------------------------------------------------------
    print("\n[BƯỚC 5 - ONE-HOT ENCODING] Mã hóa các biến danh mục bằng One-Hot Encoding:")
    cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]
    print(f"  - Danh sách 10 biến danh mục định danh: {cat_cols}")
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    print(f"  -> Kích thước DataFrame sau One-Hot Encoding: {df_encoded.shape[0]} dòng, {df_encoded.shape[1]} cột")

    # -------------------------------------------------------------
    # BƯỚC 6: CHIA TẬP, XỬ LÝ NGOẠI LỆ VÀ CHUẨN HÓA
    # -------------------------------------------------------------
    print("\n[BƯỚC 6 - OUTLIER & SCALING] Chặn ngoại lệ theo IQR và chuẩn hóa:")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print(f"  - Đặc trưng số cần chuẩn hóa: {num_cols}")

    X_raw = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Train: {X_train_raw.shape[0]} mẫu (80%) | Test: {X_test_raw.shape[0]} mẫu (20%)")

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    outlier_bounds = {}
    for col in num_cols:
        q1 = X_train_raw[col].quantile(0.25)
        q3 = X_train_raw[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_bounds[col] = (lower_bound, upper_bound)
        train_outliers = ((X_train_raw[col] < lower_bound) | (X_train_raw[col] > upper_bound)).sum()
        test_outliers = ((X_test_raw[col] < lower_bound) | (X_test_raw[col] > upper_bound)).sum()
        X_train[col] = X_train_raw[col].clip(lower=lower_bound, upper=upper_bound)
        X_test[col] = X_test_raw[col].clip(lower=lower_bound, upper=upper_bound)
        print(
            f"  - {col}: IQR [{lower_bound:.2f}, {upper_bound:.2f}], "
            f"phát hiện {train_outliers} điểm Train và {test_outliers} điểm Test (đã capping)."
        )

    scaler_ml = StandardScaler()
    X_train[num_cols] = scaler_ml.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler_ml.transform(X_test[num_cols])
    print("  -> Ngưỡng IQR và StandardScaler chỉ fit trên Train; Test chỉ được transform.")
    for col in num_cols:
        print(f"    + {col:<15}: Mean Train = {X_train[col].mean():.4f}, Std Train = {X_train[col].std():.4f}")

    X_all_clipped = X_raw.copy()
    for col, (lower_bound, upper_bound) in outlier_bounds.items():
        X_all_clipped[col] = X_all_clipped[col].clip(lower=lower_bound, upper=upper_bound)
    df_preprocessed = df_encoded.copy()
    df_preprocessed[num_cols] = scaler_ml.transform(X_all_clipped[num_cols])

    # -------------------------------------------------------------
    # BƯỚC 7: XUẤT TẬP DỮ LIỆU ĐÃ TIỀN XỬ LÝ (LINK 2 NỘP BÀI)
    # -------------------------------------------------------------
    print("\n[BƯỚC 7 - XUẤT FILE] Lưu dữ liệu sạch hoàn thiện (Link 2 nộp bài):")
    os.makedirs(os.path.join("asm", "data"), exist_ok=True)
    output_csv_path = os.path.join("asm", "data", "telco_churn_preprocessed.csv")
    df_preprocessed.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"  -> File CSV kết quả: {output_csv_path}")
    print(f"  -> Quy mô tập dữ liệu nộp: {df_preprocessed.shape[0]} dòng x {df_preprocessed.shape[1]} cột (30 features + 1 target)")

    # -------------------------------------------------------------
    # BƯỚC 8: TRỰC QUAN HÓA DỮ LIỆU & ĐỒ THỊ BOXPLOT/ROC
    # -------------------------------------------------------------
    print("\n[BƯỚC 8 - VISUALIZATION] Vẽ biểu đồ trực quan hóa dữ liệu:")
    os.makedirs(os.path.join("asm", "images"), exist_ok=True)

    # 8.1 Biểu đồ phân phối nhãn Churn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2b5c8f", "#d9534f"]

    axes[0].bar(["Khách hàng ở lại (0)", "Khách hàng rời bỏ (1)"], churn_counts.values, color=colors, width=0.5)
    axes[0].set_title("Phân phối số lượng khách hàng theo nhãn Churn", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số lượng khách hàng", fontsize=11)
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 60, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].pie(churn_counts.values, labels=["Ở lại (No)", "Rời mạng (Yes)"], autopct="%1.1f%%",
                startangle=90, colors=colors, explode=[0, 0.08], textprops={"fontsize": 11, "fontweight": "bold"})
    axes[1].set_title("Tỷ lệ rời bỏ dịch vụ viễn thông (Churn Rate)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    chart1_path = os.path.join("asm", "images", "eda_churn_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()

    # 8.2 Biểu đồ Boxplot khảo sát ngoại lệ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df.copy()

    data_tenure = [df_plot[df_plot["Churn"] == 0]["tenure"], df_plot[df_plot["Churn"] == 1]["tenure"]]
    axes[0].boxplot(data_tenure, tick_labels=["Ở lại (No)", "Rời mạng (Yes)"], patch_artist=True,
                    boxprops=dict(facecolor="#9ecae1", color="#08519c"),
                    medianprops=dict(color="#d9534f", linewidth=2))
    axes[0].set_title("Khảo sát ngoại lệ: Thâm niên tenure theo Churn", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số tháng sử dụng dịch vụ", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    data_charges = [df_plot[df_plot["Churn"] == 0]["MonthlyCharges"], df_plot[df_plot["Churn"] == 1]["MonthlyCharges"]]
    axes[1].boxplot(data_charges, tick_labels=["Ở lại (No)", "Rời mạng (Yes)"], patch_artist=True,
                    boxprops=dict(facecolor="#fcbba1", color="#cb181d"),
                    medianprops=dict(color="#2b5c8f", linewidth=2))
    axes[1].set_title("Khảo sát ngoại lệ: Cước phí hàng tháng theo Churn", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Cước phí hàng tháng ($)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart2_path = os.path.join("asm", "images", "eda_charges_tenure.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()

    # 8.3 Biểu đồ tương quan đặc trưng
    plt.figure(figsize=(10, 8))
    churn_corr = df_preprocessed.corr()["Churn"].drop("Churn").sort_values()
    top_corr = pd.concat([churn_corr.head(8), churn_corr.tail(8)]).drop_duplicates()
    top_colors = ["#2b5c8f" if x < 0 else "#d9534f" for x in top_corr.values]

    plt.barh(top_corr.index, top_corr.values, color=top_colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Hệ số tương quan Pearson giữa các đặc trưng với nhãn Churn", fontsize=12, fontweight="bold")
    plt.xlabel("Hệ số tương quan", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart3_path = os.path.join("asm", "images", "eda_correlation_features.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # BƯỚC 9: HUẤN LUYỆN LOGISTIC REGRESSION
    # -------------------------------------------------------------
    print("\n[BƯỚC 9 - HUẤN LUYỆN BASELINE] Huấn luyện Logistic Regression:")
    # Khởi tạo mô hình Logistic Regression
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)

    # Dự đoán trên tập Test
    y_pred = baseline_model.predict(X_test)
    y_prob = baseline_model.predict_proba(X_test)[:, 1]

    # Tính toán các chỉ số đánh giá
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    print("\n" + "-" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LOGISTIC REGRESSION:")
    print(f"1. Độ chính xác (Accuracy)               : {acc * 100:.2f}%")
    print(f"2. Precision ('Không bắt nhầm')           : {prec:.4f}")
    print(f"3. Recall ('Không bỏ sót')                : {rec:.4f}")
    print(f"4. F1-Score (Cân bằng Precision/Recall)  : {f1:.4f}")
    print(f"5. ROC-AUC Score (Khả năng phân tách)   : {roc:.4f}")
    print("-" * 60)
    print("Ma trận nhầm lẫn (Confusion Matrix):\n", cm)
    print("-" * 60)
    print("Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=["Ở lại (0)", "Rời mạng (1)"]))

    # 8.4 Vẽ đường cong ROC-AUC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#0056b3", lw=2.5, label=f"Đường cong ROC (AUC = {roc:.4f})")
    plt.plot([0, 1], [0, 1], color="#d9534f", lw=1.5, linestyle="--", label="Ngẫu nhiên (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tỷ lệ dương tính giả FPR (False Positive Rate)", fontsize=11)
    plt.ylabel("Tỷ lệ dương tính thật TPR (True Positive Rate / Recall)", fontsize=11)
    plt.title("Đường cong ROC & Chỉ số AUC (Logistic Regression)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_chart_path = os.path.join("asm", "images", "duong_cong_roc_auc.png")
    plt.savefig(roc_chart_path, dpi=300)
    plt.close()
    print(f"  + Đã lưu biểu đồ Đường cong ROC: {roc_chart_path}")

    # 8.5 Xuất ảnh Dashboard tổng hợp minh chứng thực thi (Link 3 nộp bài)
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Bảng tóm tắt quy trình
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis("off")

    summary_text = (
        "BÁO CÁO KẾT QUẢ TIỀN XỬ LÝ DỮ LIỆU VÀ ĐÁNH GIÁ MÔ HÌNH\n"
        "Đề tài 11: Customer Churn Analysis | Dữ liệu: Telco Customer Churn (7,043 dòng x 21 cột gốc)\n"
        + "=" * 98 + "\n"
        "✓ EDA và làm sạch dữ liệu       : Xử lý 11 ô trống của TotalCharges; kiểm tra ngoại lệ.\n"
        "✓ Mã hóa dữ liệu                : Binary Encoding và One-Hot Encoding drop_first=True.\n"
        "✓ Chuẩn hóa và chống Leakage    : Chia 80:20; IQR và StandardScaler chỉ fit trên Train.\n"
        "✓ Mô hình cơ sở                : Logistic Regression với ngưỡng xác suất mặc định 0.5.\n"
        f"✓ Đánh giá mô hình              : Accuracy = {acc*100:.2f}% | Precision = {prec:.4f} ('không bắt nhầm')\n"
        f"                                   Recall = {rec:.4f} ('không bỏ sót') | F1-Score = {f1:.4f} | ROC-AUC = {roc:.4f}\n"
        + "=" * 98 + "\n"
    )
    ax_info.text(0.01, 0.95, summary_text, fontsize=10.5, family="DejaVu Sans", verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#0056b3", linewidth=2))

    # Ma trận nhầm lẫn
    ax_cm = fig.add_subplot(gs[1, 0])
    cax = ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(x=j, y=i, s=f"{cm[i, j]:,}", va="center", ha="center", size="xx-large", weight="bold")
    ax_cm.set_title("Ma trận nhầm lẫn 2x2 (TP, TN, FP, FN)", fontsize=11, fontweight="bold", pad=15)
    ax_cm.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10)
    ax_cm.set_ylabel("Nhãn thực tế (True Label)", fontsize=10)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Ở lại (0)", "Rời mạng (1)"])
    ax_cm.set_yticklabels(["Ở lại (0)", "Rời mạng (1)"])
    fig.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)

    # Đồ thị các chỉ số
    ax_metrics = fig.add_subplot(gs[1, 1])
    metric_names = ["Accuracy", "Precision\n(k bắt nhầm)", "Recall\n(k bỏ sót)", "F1-Score\n(cân bằng)", "ROC-AUC\n(phân tách)"]
    metric_values = [acc, prec, rec, f1, roc]
    bars = ax_metrics.bar(metric_names, metric_values, color=["#007bff", "#28a745", "#ffc107", "#fd7e14", "#20c997"], width=0.5)
    ax_metrics.set_ylim(0, 1.1)
    ax_metrics.set_title("Các chỉ số đánh giá phân loại toàn diện", fontsize=11, fontweight="bold")
    ax_metrics.set_ylabel("Điểm số (0.0 - 1.0)", fontsize=10)
    for bar, val in zip(bars, metric_values):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax_metrics.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    proof_image_path = os.path.join("asm", "images", "ket_qua_chay_tien_xu_ly.png")
    plt.savefig(proof_image_path, dpi=300)
    plt.close()
    print(f"  + Đã xuất ảnh minh chứng kết quả chạy tiền xử lý: {proof_image_path}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
