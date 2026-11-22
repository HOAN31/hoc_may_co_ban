# -*- coding: utf-8 -*-
"""
BÀI TẬP LỚN / ASSIGNMENT GIAI ĐOẠN 1 - MÔN HỌC MÁY CƠ BẢN (ITA202)
ĐỀ TÀI 11: PHÂN TÍCH HÀNH VI VÀ DỰ BÁO RỜI BỎ DỊCH VỤ CỦA KHÁCH HÀNG (TELCO CUSTOMER CHURN)

Yêu cầu Assignment 1:
- Chọn dữ liệu và tiền xử lý dữ liệu trước (để giai đoạn 2 thực hiện Machine Learning)
- Các bước tiền xử lý:
  1. Khám phá dữ liệu (EDA), kiểm tra kích thước, thuộc tính và kiểu dữ liệu
  2. Loại bỏ thuộc tính không mang giá trị dự báo (customerID)
  3. Xử lý giá trị khuyết thiếu / lỗi định dạng (khoảng trắng ở TotalCharges)
  4. Mã hóa biến mục tiêu Churn (Yes/No -> 1/0)
  5. Mã hóa biến phân loại (Binary Encoding & One-Hot Encoding)
  6. Chuẩn hóa đặc trưng số (StandardScaler)
  7. Xuất tập dữ liệu sau tiền xử lý (telco_churn_preprocessed.csv)
  8. Trực quan hóa và xuất biểu đồ kết quả tiền xử lý
  9. Huấn luyện Baseline Model (Logistic Regression) kiểm tra độ tương thích dữ liệu

Tác giả: Sinh viên FPT Polytechnic
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
    confusion_matrix,
    classification_report
)

def main():
    print("=" * 80)
    print("ASSIGNMENT 1 - HỌC MÁY CƠ BẢN (ITA202)")
    print("ĐỀ TÀI 11: TELCO CUSTOMER CHURN - CHỌN DỮ LIỆU & TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 80)

    # -------------------------------------------------------------
    # BƯỚC 1: ĐỌC VÀ KHÁM PHÁ DỮ LIỆU GỐC (EDA)
    # -------------------------------------------------------------
    raw_data_path = os.path.join("asm", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(raw_data_path):
        raw_data_path = os.path.join("asm", "data", "telco_customer_churn.csv")
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {raw_data_path}")

    print(f"\n[BƯỚC 1] Đọc dữ liệu gốc từ: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    print(f"- Kích thước tập dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"- Danh sách 21 thuộc tính ban đầu:\n  {list(df.columns)}")

    # Kiểm tra phân phối nhãn mục tiêu Churn
    churn_counts = df["Churn"].value_counts()
    churn_ratio = df["Churn"].value_counts(normalize=True) * 100
    print("\n- Phân phối nhãn mục tiêu (Churn):")
    print(f"  + Không rời mạng (No) : {churn_counts['No']} khách hàng ({churn_ratio['No']:.2f}%)")
    print(f"  + Rời mạng (Yes)       : {churn_counts['Yes']} khách hàng ({churn_ratio['Yes']:.2f}%)")

    # -------------------------------------------------------------
    # BƯỚC 2: LOẠI BỎ THUỘC TÍNH KHÔNG MANG GIÁ TRỊ DỰ BÁO
    # -------------------------------------------------------------
    print("\n[BƯỚC 2] Xử lý thuộc tính không cần thiết:")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("  -> Đã loại bỏ cột 'customerID' (mã định danh ngẫu nhiên, không có giá trị học máy).")

    # -------------------------------------------------------------
    # BƯỚC 3: PHÁT HIỆN VÀ XỬ LÝ DỮ LIỆU KHUYẾT THIẾU / SAI ĐỊNH DẠNG
    # -------------------------------------------------------------
    print("\n[BƯỚC 3] Phát hiện và xử lý dữ liệu khuyết thiếu / sai kiểu:")
    # Cột TotalCharges ban đầu bị đọc là kiểu object/string do có khoảng trắng ' '
    space_mask = df["TotalCharges"] == " "
    num_space = space_mask.sum()
    print(f"  - Phát hiện {num_space} dòng có TotalCharges là khoảng trắng rỗng (' ').")
    if num_space > 0:
        sample_tenure = df.loc[space_mask, "tenure"].unique()
        print(f"  - Toàn bộ các dòng này đều có tenure = {sample_tenure} (khách hàng mới ký hợp đồng tháng đầu tiên).")
        print("  -> Chuyển đổi TotalCharges sang kiểu float và điền giá trị 0.0 hợp lý về mặt nghiệp vụ.")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    print(f"  - Kiểm tra lại số lượng giá trị null sau khi xử lý: {df.isnull().sum().sum()} giá trị null.")

    # -------------------------------------------------------------
    # BƯỚC 4: MÃ HÓA BIẾN MỤC TIÊU VÀ CÁC BIẾN NHỊ PHÂN
    # -------------------------------------------------------------
    print("\n[BƯỚC 4] Mã hóa biến mục tiêu Churn và các biến nhị phân:")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    print("  - Churn: 'No' -> 0, 'Yes' -> 1")

    # Mã hóa các biến nhị phân (Binary Variables)
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"No": 0, "Yes": 1})
    
    # Mã hóa gender
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    print(f"  - Đã mã hóa nhị phân 0/1 cho các cột: gender, {', '.join(binary_cols)}")

    # -------------------------------------------------------------
    # BƯỚC 5: MÃ HÓA CÁC BIẾN PHÂN LOẠI NHIỀU DANH MỤC (ONE-HOT ENCODING)
    # -------------------------------------------------------------
    print("\n[BƯỚC 5] Mã hóa các biến danh mục bằng One-Hot Encoding (drop_first=True):")
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
    print(f"  - Danh sách biến danh mục: {cat_cols}")
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    print(f"  -> Kích thước sau khi One-Hot Encoding: {df_encoded.shape[0]} dòng, {df_encoded.shape[1]} cột")

    # -------------------------------------------------------------
    # BƯỚC 6: CHUẨN HÓA ĐẶC TRƯNG SỐ (FEATURE SCALING VỚI STANDARDSCALER)
    # -------------------------------------------------------------
    print("\n[BƯỚC 6] Chuẩn hóa đặc trưng số bằng StandardScaler:")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print(f"  - Các đặc trưng số cần chuẩn hóa: {num_cols}")
    
    scaler = StandardScaler()
    df_preprocessed = df_encoded.copy()
    df_preprocessed[num_cols] = scaler.fit_transform(df_preprocessed[num_cols])

    print("  - Thống kê sau khi chuẩn hóa (Mean ~ 0, Std ~ 1):")
    for col in num_cols:
        print(f"    + {col:<15}: Mean = {df_preprocessed[col].mean():.4f}, Std = {df_preprocessed[col].std():.4f}")

    # -------------------------------------------------------------
    # BƯỚC 7: XUẤT TẬP DỮ LIỆU SAU TIỀN XỬ LÝ (LINK 2 NỘP BÀI)
    # -------------------------------------------------------------
    print("\n[BƯỚC 7] Xuất dữ liệu đã tiền xử lý hoàn chỉnh ra file CSV:")
    os.makedirs(os.path.join("asm", "data"), exist_ok=True)
    output_csv_path = os.path.join("asm", "data", "telco_churn_preprocessed.csv")
    df_preprocessed.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"  -> File kết quả đã lưu tại: {output_csv_path}")
    print(f"  -> Kích thước tập dữ liệu hoàn thiện: {df_preprocessed.shape[0]} dòng x {df_preprocessed.shape[1]} cột")

    # Lưu thêm 1 bản copy tên chuẩn telco_customer_churn.csv nếu chưa có
    standard_raw_path = os.path.join("asm", "data", "telco_customer_churn.csv")
    if not os.path.exists(standard_raw_path):
        import shutil
        shutil.copyfile(raw_data_path, standard_raw_path)

    # -------------------------------------------------------------
    # BƯỚC 8: TRỰC QUAN HÓA DỮ LIỆU & XUẤT ẢNH BÁO CÁO (LINK 3 NỘP BÀI)
    # -------------------------------------------------------------
    print("\n[BƯỚC 8] Trực quan hóa dữ liệu và xuất ảnh báo cáo:")
    os.makedirs(os.path.join("asm", "images"), exist_ok=True)

    # 8.1 Biểu đồ phân phối nhãn Churn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2b5c8f", "#d9534f"]

    # Bar chart
    axes[0].bar(["Khách hàng ở lại (0)", "Khách hàng rời bỏ (1)"], churn_counts.values, color=colors, width=0.5)
    axes[0].set_title("Phân phối số lượng khách hàng theo trạng thái Churn", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số lượng khách hàng", fontsize=11)
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Pie chart
    axes[1].pie(churn_counts.values, labels=["Ở lại (No)", "Rời bỏ (Yes)"], autopct="%1.1f%%",
                startangle=90, colors=colors, explode=[0, 0.08], textprops={"fontsize": 11, "fontweight": "bold"})
    axes[1].set_title("Tỷ lệ rời bỏ dịch vụ viễn thông (Churn Rate)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    chart1_path = os.path.join("asm", "images", "eda_churn_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"  + Đã lưu biểu đồ phân phối Churn: {chart1_path}")

    # 8.2 Biểu đồ mối quan hệ giữa tenure, MonthlyCharges và Churn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df.copy()
    
    # Boxplot tenure
    data_tenure = [df_plot[df_plot["Churn"] == 0]["tenure"], df_plot[df_plot["Churn"] == 1]["tenure"]]
    axes[0].boxplot(data_tenure, tick_labels=["Ở lại (No)", "Rời mạng (Yes)"], patch_artist=True,
                    boxprops=dict(facecolor="#9ecae1", color="#08519c"),
                    medianprops=dict(color="#d9534f", linewidth=2))
    axes[0].set_title("Thời gian gắn bó (Tenure - tháng) theo nhóm Churn", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số tháng sử dụng dịch vụ", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Boxplot MonthlyCharges
    data_charges = [df_plot[df_plot["Churn"] == 0]["MonthlyCharges"], df_plot[df_plot["Churn"] == 1]["MonthlyCharges"]]
    axes[1].boxplot(data_charges, tick_labels=["Ở lại (No)", "Rời mạng (Yes)"], patch_artist=True,
                    boxprops=dict(facecolor="#fcbba1", color="#cb181d"),
                    medianprops=dict(color="#2b5c8f", linewidth=2))
    axes[1].set_title("Cước phí hàng tháng (Monthly Charges - USD) theo nhóm Churn", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Cước phí hàng tháng ($)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart2_path = os.path.join("asm", "images", "eda_charges_tenure.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"  + Đã lưu biểu đồ tenure & cước phí: {chart2_path}")

    # 8.3 Biểu đồ tương quan (Correlation with Churn)
    plt.figure(figsize=(10, 8))
    churn_corr = df_preprocessed.corr()["Churn"].drop("Churn").sort_values()
    bar_colors = ["#2b5c8f" if x < 0 else "#d9534f" for x in churn_corr.values]
    
    # Chỉ lấy top 15 đặc trưng có tương quan cao nhất
    top_corr = pd.concat([churn_corr.head(8), churn_corr.tail(8)]).drop_duplicates()
    top_colors = ["#2b5c8f" if x < 0 else "#d9534f" for x in top_corr.values]
    
    plt.barh(top_corr.index, top_corr.values, color=top_colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Top đặc trưng có tương quan mạnh nhất với quyết định rời mạng (Churn)", fontsize=12, fontweight="bold")
    plt.xlabel("Hệ số tương quan Pearson với Churn", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart3_path = os.path.join("asm", "images", "eda_correlation_features.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"  + Đã lưu biểu đồ tương quan đặc trưng: {chart3_path}")

    # -------------------------------------------------------------
    # BƯỚC 9: CHIA TẬP TRAIN/TEST & HUẤN LUYỆN BASELINE LOGISTIC REGRESSION
    # -------------------------------------------------------------
    print("\n[BƯỚC 9] Chia tập dữ liệu Train/Test và huấn luyện mô hình cơ sở (Baseline):")
    X = df_preprocessed.drop(columns=["Churn"])
    y = df_preprocessed["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Tập huấn luyện (Train): {X_train.shape[0]} mẫu (80%)")
    print(f"  - Tập kiểm thử (Test)    : {X_test.shape[0]} mẫu (20%)")

    # Huấn luyện Logistic Regression
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)

    y_pred = baseline_model.predict(X_test)
    y_prob = baseline_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BASELINE (LOGISTIC REGRESSION):")
    print(f"- Độ chính xác (Accuracy)       : {acc * 100:.2f}%")
    print(f"- Precision                     : {prec:.4f}")
    print(f"- Recall                        : {rec:.4f}")
    print(f"- F1-Score                      : {f1:.4f}")
    print(f"- ROC-AUC Score                 : {roc:.4f}")
    print("-" * 60)
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(cm)
    print("-" * 60)
    print("Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=["Ở lại (0)", "Rời mạng (1)"]))

    # 8.4 Xuất ảnh Bảng kết quả chạy tiền xử lý tổng hợp (Dashboard Proof of Execution)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Subplot 1: Bảng thông tin tiền xử lý
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis("off")

    summary_text = (
        "BÁO CÁO KẾT QUẢ THỰC HIỆN TIỀN XỬ LÝ DỮ LIỆU - ASSIGNMENT 1 (ITA202)\n"
        "Đề tài 11: Customer Churn Analysis | Bộ dữ liệu: Telco Customer Churn (blastchar / IBM)\n"
        + "=" * 95 + "\n"
        f"1. Kích thước dữ liệu gốc         : 7,043 dòng, 21 thuộc tính\n"
        f"2. Thuộc tính loại bỏ              : customerID (mã định danh ngẫu nhiên không có tính khái quát)\n"
        f"3. Xử lý giá trị khuyết thiếu      : Cột 'TotalCharges' có 11 khoảng trắng rỗng (' ') do tenure=0 -> Điền 0.0\n"
        f"4. Mã hóa biến mục tiêu Churn      : 'No' -> 0 ({churn_counts['No']} mẫu), 'Yes' -> 1 ({churn_counts['Yes']} mẫu)\n"
        f"5. Mã hóa biến phân loại           : Binary Encoding (gender, Partner...) + One-Hot Encoding (drop_first=True)\n"
        f"6. Chuẩn hóa đặc trưng số          : StandardScaler (Z-score) cho tenure, MonthlyCharges, TotalCharges\n"
        f"7. Kích thước sau tiền xử lý       : {df_preprocessed.shape[0]} dòng, {df_preprocessed.shape[1]} cột (30 features + 1 target)\n"
        f"8. Phân chia dữ liệu               : Train = {X_train.shape[0]} mẫu (80%), Test = {X_test.shape[0]} mẫu (20%) [stratify=y]\n"
        f"9. Đánh giá Baseline (Logistic Reg): Accuracy = {acc*100:.2f}% | F1-Score = {f1:.4f} | ROC-AUC = {roc:.4f}\n"
        + "=" * 95 + "\n"
        "KẾT LUẬN: Dữ liệu đã được làm sạch, chuẩn hóa và đóng gói thành công, sẵn sàng 100% cho Giai đoạn 2."
    )
    ax_info.text(0.01, 0.95, summary_text, fontsize=11, family="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#007bff", linewidth=2))

    # Subplot 2: Ma trận nhầm lẫn Confusion Matrix của Baseline
    ax_cm = fig.add_subplot(gs[1, 0])
    cax = ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(x=j, y=i, s=f"{cm[i, j]:,}", va="center", ha="center", size="xx-large", weight="bold")
    ax_cm.set_title("Ma trận nhầm lẫn (Baseline Logistic Regression)", fontsize=11, fontweight="bold", pad=15)
    ax_cm.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10)
    ax_cm.set_ylabel("Nhãn thực tế (True Label)", fontsize=10)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Ở lại (0)", "Rời mạng (1)"])
    ax_cm.set_yticklabels(["Ở lại (0)", "Rời mạng (1)"])
    fig.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)

    # Subplot 3: Chỉ số hiệu năng trực quan
    ax_metrics = fig.add_subplot(gs[1, 1])
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    metric_values = [acc, prec, rec, f1, roc]
    bars = ax_metrics.bar(metric_names, metric_values, color=["#007bff", "#28a745", "#ffc107", "#fd7e14", "#20c997"], width=0.5)
    ax_metrics.set_ylim(0, 1.1)
    ax_metrics.set_title("Chỉ số hiệu năng mô hình cơ sở (Giai đoạn 1)", fontsize=11, fontweight="bold")
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
    print("HOÀN THÀNH TOÀN BỘ TIỀN XỬ LÝ ASSIGNMENT 1 CHO ĐỀ TÀI 11 THÀNH CÔNG!")
    print("=" * 80)

if __name__ == "__main__":
    main()
