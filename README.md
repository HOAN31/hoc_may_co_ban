# Môn học: Học Máy Cơ Bản (Machine Learning Basics)

Kho lưu trữ bài tập thực hành môn **Học máy cơ bản** - FPT Polytechnic.

---

## 📁 Cấu trúc thư mục

```text
hoc_may_co_ban/
├── lab1/
│   └── lab1.py                         # Bài thực hành Lab 1
├── lab2/
│   └── lab2.py                         # Bài thực hành Lab 2
├── lab3/
│   ├── lab3.py                         # Bài thực hành Lab 3 (Linear Regression)
│   ├── bai3_linear_regression.png      # Biểu đồ Bài 3 (Tập Diabetes)
│   └── bai4_california_results.png     # Biểu đồ Bài 4 (Tập California Housing)
├── .gitignore                          # Cấu hình bỏ qua thư mục ảo .venv và cache
└── README.md                           # Hướng dẫn tổng quan
```

---

## 📌 Nội dung chi tiết các Lab

### 🔹 Lab 1: Giới thiệu Machine Learning & Thư viện cơ bản
- **Bài 1**: Làm quen với thư viện **NumPy** và **Pandas** (xử lý mảng và DataFrame thông tin sinh viên).
- **Bài 2**: Phân loại các bài toán Machine Learning (Supervised vs Unsupervised, Hồi quy vs Phân loại).
- **Bài 3**: Thử nghiệm mô hình **Linear Regression** trên tập dữ liệu Iris.
- **Bài 4**: Khám phá và kiểm tra tập dữ liệu ảnh chữ số viết tay **Digits**.

### 🔹 Lab 2: Tiền xử lý dữ liệu (Data Preprocessing)
- **Bài 1**: Làm sạch dữ liệu, điền giá trị thiếu (mean imputation) và xử lý ngoại lai (outliers) bằng quy tắc 3-sigma.
- **Bài 2**: Chuẩn hóa dữ liệu với **StandardScaler** và **MinMaxScaler**.
- **Bài 3**: Chia tập dữ liệu Train / Test (`train_test_split`) trên bộ dữ liệu Diabetes.
- **Bài 4**: Xây dựng quy trình học máy hoàn chỉnh với **Pipeline** kết hợp chuẩn hóa dữ liệu và mô hình **LogisticRegression**.

### 🔹 Lab 3: Hồi quy tuyến tính (Linear Regression)
- **Bài 1: Xây dựng mô hình Hồi quy tuyến tính**
  - Tạo dữ liệu giả lập có mối quan hệ tuyến tính: $y = 2X + 1 + \text{nhiễu Gauss}$.
  - Huấn luyện mô hình với `LinearRegression` và in ra hệ số góc ($w$) cùng hệ số chặn ($b$).
- **Bài 2: Đánh giá mô hình (Tập dữ liệu Diabetes)**
  - Lấy 1 đặc trưng duy nhất (BMI) để dự đoán mức độ tiến triển của bệnh.
  - Đánh giá bằng các chỉ số: **MAE**, **MSE**, **RMSE**, và **R-squared ($R^2$)** trên tập Test kèm giải thích ý nghĩa.
- **Bài 3: Trực quan hóa kết quả**
  - Dùng Matplotlib vẽ biểu đồ tán xạ (scatter plot) dữ liệu thực tế và đường thẳng hồi quy của mô hình (lưu ảnh `bai3_linear_regression.png`).
- **Bài 4: Quy trình ML hoàn chỉnh (California Housing)**
  - Thực hiện đầy đủ các bước: Load -> Preprocess (StandardScaler) -> Train (LinearRegression qua Pipeline) -> Evaluate (MSE, MAE, $R^2$) -> Visualize kết quả (lưu ảnh `bai4_california_results.png`).
- **Bài 5: Phần mở rộng (Giảng viên giao thêm)**
  - Thử nghiệm mô hình **Ridge Regression (chuẩn hóa $L_2$)** và so sánh hiệu năng.

---

## 🚀 Hướng dẫn chạy

Cài đặt các thư viện cần thiết:
```bash
pip install numpy pandas scikit-learn matplotlib
```

Chạy từng bài Lab:
```bash
# Chạy Lab 1
python lab1/lab1.py

# Chạy Lab 2
python lab2/lab2.py

# Chạy Lab 3
python lab3/lab3.py
```
