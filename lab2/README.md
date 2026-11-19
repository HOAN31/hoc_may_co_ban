# Lab 2: Tiền Xử Lý Dữ Liệu (Data Preprocessing) & Pipeline

---

## 🎯 Mục tiêu
- Biết cách làm sạch dữ liệu, xử lý giá trị bị thiếu (missing values) và giá trị ngoại lai (outliers).
- Áp dụng các kỹ thuật chuẩn hóa dữ liệu: **StandardScaler** ($Z$-score) và **MinMaxScaler** (về khoảng $[0, 1]$).
- Hiểu và thực hiện phân chia dữ liệu Train/Test (`train_test_split`) để đánh giá mô hình khách quan.
- Xây dựng quy trình học máy tự động hóa, khép kín với **Pipeline** trong Scikit-learn.

---

## 📋 Nội dung các bài tập

### Bài 1: Làm sạch dữ liệu & Xử lý ngoại lai (Outliers)
- Điền khuyết giá trị thiếu (`NaN`) bằng giá trị trung bình (`mean`).
- Phát hiện và xử lý điểm ngoại lai (outliers) dựa trên quy tắc thống kê $3\sigma$ (3 độ lệch chuẩn):
  $$|x - \mu| > 3\sigma$$
- Thay thế các giá trị ngoại lai bằng giá trị trung bình của cột.

### Bài 2: Chuẩn hóa dữ liệu (Feature Scaling)
- Sử dụng **StandardScaler**: Đưa dữ liệu về phân phối chuẩn có $\mu = 0$ và $\sigma = 1$:
  $$z = \frac{x - \mu}{\sigma}$$
- Sử dụng **MinMaxScaler**: Đưa dữ liệu về miền giá trị $[0, 1]$:
  $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### Bài 3: Chia tập dữ liệu Train / Test
- Sử dụng tập dữ liệu `load_diabetes` từ Scikit-learn.
- Phân chia theo tỷ lệ **80% Train - 20% Test** với `random_state=42` để đảm bảo tính tái lập.

### Bài 4: Xây dựng quy trình hoàn chỉnh với Pipeline
- Sử dụng tập dữ liệu `load_digits` cho bài toán phân loại đa lớp.
- Đóng gói toàn bộ quy trình gồm:
  1. Chuẩn hóa đặc trưng: `StandardScaler()`
  2. Huấn luyện mô hình: `LogisticRegression(max_iter=1000)`
- Đánh giá độ chính xác (`Accuracy`) trên tập Test.

---

## 🚀 Hướng dẫn chạy

```bash
python lab2.py
```
