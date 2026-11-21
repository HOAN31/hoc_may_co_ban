# Môn học: Học Máy Cơ Bản (Machine Learning Basics)

Kho lưu trữ bài tập thực hành môn **Học máy cơ bản** - FPT Polytechnic.

---

## 📁 Cấu trúc thư mục dự án

```text
hoc_may_co_ban/
├── asm/                                # 🌟 Assignment: Dự án Machine Learning hoàn chỉnh
│   ├── data/titanic.csv                # Tập dữ liệu Titanic
│   ├── images/                         # Các biểu đồ phân tích EDA, Ma trận nhầm lẫn, ROC-AUC
│   ├── giai_doan_1.py                  # Code Giai đoạn 1 (Mô hình cơ sở)
│   ├── BAO_CAO_GIAI_DOAN_1.md          # Báo cáo chi tiết Giai đoạn 1
│   └── README.md                       # Hướng dẫn chi tiết Assignment
├── lab1/
│   ├── lab1.py                         # Mã nguồn Lab 1
│   └── README.md                       # Hướng dẫn chi tiết Lab 1
├── lab2/
│   ├── lab2.py                         # Mã nguồn Lab 2
│   └── README.md                       # Hướng dẫn chi tiết Lab 2
├── lab3/
│   ├── lab3.py                         # Mã nguồn Lab 3
│   ├── bai3_linear_regression.png      # Đồ thị Bài 3 (Lab 3)
│   ├── bai4_california_results.png     # Đồ thị Bài 4 (Lab 3)
│   └── README.md                       # Hướng dẫn chi tiết Lab 3
├── lab4/
│   ├── lab4.py                         # Mã nguồn Lab 4
│   ├── bai3_confusion_matrix.png       # Ma trận nhầm lẫn (Lab 4)
│   ├── bai4_digits_misclassified.png   # 5 mẫu đoán sai (Lab 4)
│   ├── bai5_roc_curve.png              # Đường cong ROC - AUC (Lab 4)
│   └── README.md                       # Hướng dẫn chi tiết Lab 4
├── lab_pdf/                            # Đề bài các bài Lab và Assignment PDF
├── vo_ghi/                             # Vở ghi chép lý thuyết các bài học
├── .gitignore                          # Cấu hình bỏ qua .venv và cache
└── README.md                           # Tổng quan repository
```

---

## 📌 Danh sách các bài thực hành & Dự án

| Thư mục | Chủ đề | Chi tiết |
| :--- | :--- | :--- |
| **[asm/](asm/)** | **🌟 Assignment: Dự án ML Hoàn Chỉnh (Titanic)** | **Giai đoạn 1:** Khám phá dữ liệu, làm sạch NaN/Outliers, One-Hot, StandardScaler, Pipeline Logistic Regression (Accuracy: 80.45%, AUC: 0.8431). |
| **[lab1/](lab1/)** | **Làm quen NumPy, Pandas & Phân loại bài toán ML** | Thao tác mảng/bảng, phân loại Supervised/Unsupervised, thử nghiệm Iris và Digits. |
| **[lab2/](lab2/)** | **Tiền xử lý dữ liệu & Pipeline** | Điền khuyết dữ liệu, xử lý outlier $3\sigma$, chuẩn hóa StandardScaler/MinMaxScaler, Pipeline, LabelEncoder. |
| **[lab3/](lab3/)** | **Hồi quy tuyến tính (Linear Regression)** | Dữ liệu giả lập, đánh giá Diabetes (MAE/MSE/RMSE/$R^2$), trực quan hóa đồ thị, California Housing, Ridge Regression. |
| **[lab4/](lab4/)** | **Phân loại – Logistic Regression** | Phân loại nhị phân Breast Cancer (Accuracy, Precision, Recall, F1, AUC), Ma trận nhầm lẫn (TP/TN/FP/FN), phân loại đa lớp Digits, ROC-AUC. |

---

## 🚀 Cài đặt & Chạy chương trình

### 1. Cài đặt các thư viện cần thiết
```bash
pip install numpy pandas scikit-learn matplotlib
```

### 2. Chạy bài tập hoặc Assignment
```bash
# Chạy Assignment Giai đoạn 1
python asm/giai_doan_1.py

# Chạy Lab 1 -> Lab 4
python lab1/lab1.py
python lab2/lab2.py
python lab3/lab3.py
python lab4/lab4.py
```
