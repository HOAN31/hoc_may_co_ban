# Môn học: Học Máy Cơ Bản (Machine Learning Basics) - ITA202

Kho lưu trữ bài tập thực hành & Đồ án môn **Học máy cơ bản** - FPT Polytechnic.

---

## 🌟 NỘP BÀI ASSIGNMENT 1: ĐỀ TÀI 11 - TELCO CUSTOMER CHURN
> **Yêu cầu nộp bài:** File Word chứa 3 liên kết nộp bài trước 23h59 Chủ nhật.
> File Word chính thức: [**`ASSIGNMENT_1_HOAN31.docx`**](ASSIGNMENT_1_HOAN31.docx) (hoặc trong thư mục [`asm/ASSIGNMENT_1_HOAN31.docx`](asm/ASSIGNMENT_1_HOAN31.docx)).

### 📌 Bảng 3 Liên kết bắt buộc:
1. **Link 1 (Mã nguồn):** [https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/tien_xu_ly_telco.py](https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/tien_xu_ly_telco.py)
2. **Link 2 (Dữ liệu sau tiền xử lý):** [https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/data/telco_churn_preprocessed.csv](https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/data/telco_churn_preprocessed.csv)
3. **Link 3 (Ảnh chụp kết quả chạy tiền xử lý):** [https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/images/ket_qua_chay_tien_xu_ly.png](https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/images/ket_qua_chay_tien_xu_ly.png)

---

## 📁 Cấu trúc thư mục dự án

```text
hoc_may_co_ban/
├── ASSIGNMENT_1_HOAN31.docx            # 📄 File Word chính thức nộp Assignment 1
├── asm/                                # 🌟 Assignment: Dự án Machine Learning hoàn chỉnh
│   ├── ASSIGNMENT_1_HOAN31.docx        # Bản lưu file Word báo cáo trong asm/
│   ├── tien_xu_ly_telco.py             # Code tiền xử lý Đề tài 11 (Telco Customer Churn)
│   ├── tao_file_word_asm1.py           # Script tự động tạo file Word nộp bài
│   ├── data/
│   │   ├── telco_customer_churn.csv    # Dữ liệu gốc Telco Churn (7,043 dòng x 21 cột)
│   │   └── telco_churn_preprocessed.csv# Dữ liệu sau tiền xử lý (7,043 dòng x 31 cột)
│   ├── images/
│   │   ├── ket_qua_chay_tien_xu_ly.png # Ảnh Dashboard kết quả chạy tiền xử lý (Link 3)
│   │   ├── eda_churn_distribution.png  # Biểu đồ phân phối nhãn Churn
│   │   ├── eda_charges_tenure.png      # Phân tích thâm niên & cước phí
│   │   └── eda_correlation_features.png# Biểu đồ tương quan đặc trưng
│   └── README.md                       # Báo cáo chi tiết Assignment 1
├── lab1/                               # Lab 1: NumPy, Pandas & Phân loại bài toán ML
├── lab2/                               # Lab 2: Tiền xử lý dữ liệu & Pipeline
├── lab3/                               # Lab 3: Hồi quy tuyến tính (Linear Regression)
├── lab4/                               # Lab 4: Phân loại – Logistic Regression & Đánh giá
├── lab_pdf/                            # Đề bài các bài Lab và Assignment PDF
├── vo_ghi/                             # Vở ghi chép lý thuyết 4 bài học
└── README.md                           # Tổng quan repository
```

---

## 📌 Danh sách các bài thực hành & Dự án

| Thư mục | Chủ đề | Chi tiết |
| :--- | :--- | :--- |
| **[asm/](asm/)** | **🌟 Đề tài 11: Dự báo khách hàng rời mạng (Telco Churn)** | **Assignment 1:** Làm sạch dữ liệu, xử lý missing value `TotalCharges`, One-Hot Encoding, StandardScaler, xuất tập CSV 7,043 dòng x 31 cột, Baseline Logistic Regression (Accuracy: 80.62%, AUC: 0.8422). |
| **[lab1/](lab1/)** | **Làm quen NumPy, Pandas & Phân loại bài toán ML** | Thao tác mảng/bảng, phân loại Supervised/Unsupervised, thử nghiệm Iris và Digits. |
| **[lab2/](lab2/)** | **Tiền xử lý dữ liệu & Pipeline** | Điền khuyết dữ liệu, xử lý outlier $3\sigma$, chuẩn hóa StandardScaler/MinMaxScaler, Pipeline, LabelEncoder. |
| **[lab3/](lab3/)** | **Hồi quy tuyến tính (Linear Regression)** | Dữ liệu giả lập, đánh giá Diabetes (MAE/MSE/RMSE/$R^2$), trực quan hóa đồ thị, California Housing, Ridge Regression. |
| **[lab4/](lab4/)** | **Phân loại – Logistic Regression** | Phân loại nhị phân Breast Cancer (Accuracy, Precision, Recall, F1, AUC), Ma trận nhầm lẫn (TP/TN/FP/FN), phân loại đa lớp Digits, ROC-AUC. |

---

## 🚀 Hướng dẫn chạy mã nguồn Tiền xử lý dữ liệu

```bash
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# 2. Chạy script tiền xử lý dữ liệu Đề tài 11
python asm/tien_xu_ly_telco.py

# 3. Tạo lại file Word nộp bài (nếu cần chỉnh sửa)
python asm/tao_file_word_asm1.py
```
