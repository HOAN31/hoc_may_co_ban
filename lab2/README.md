# Lab 2: Tiền Xử Lý Dữ Liệu (Data Preprocessing) & Pipeline

Kho lưu trữ và hướng dẫn thực hành **Lab 2: Dữ liệu và Tiền xử lý** (đối chiếu theo đề bài `ITA202-Lab2.pdf` và vở ghi chép bài học `Note2_ITA202.pdf`).

---

## 🎯 Mục tiêu & Lý thuyết cốt lõi (Theo Vở Ghi Note 2)
- **Tầm quan trọng của dữ liệu:** Dữ liệu thô ví như *"Dầu mỏ"* — vô giá trị nếu không được tinh chế. Thời gian tiền xử lý dữ liệu chiếm tới **80%** toàn bộ dự án Machine Learning.
- **Xử lý giá trị thiếu (Missing Values - NaN, None):**
  - Xóa: Dòng nếu thiếu ít, xóa cột nếu thiếu quá nhiều (> 50-60%).
  - Điền khuyết: Dùng `Mean` (trung bình), `Median` (trung vị), `Mode` (yếu vị) hoặc hằng số.
- **Xử lý giá trị ngoại lệ (Outliers):**
  - Xác định bằng biểu đồ Boxplot, quy tắc $3\sigma$ ($|x - \mu| > 3\sigma$) hoặc khoảng tứ phân vị IQR.
  - Xử lý: Loại bỏ, biến đổi log, hoặc cắt biên (Winsorization).
- **Chuẩn hóa (Feature Scaling):**
  - Tránh việc các cột có giá trị lớn (vd: Lương/Thu nhập) lấn át các cột có giá trị nhỏ (vd: Tuổi).
  - **StandardScaler:** Đưa về phân phối chuẩn ($\mu = 0, \sigma = 1$).
  - **MinMaxScaler:** Đưa về khoảng $[0, 1]$.
- **Mã hóa (Encoding):**
  - **Label Encoding:** Gán số nguyên cho dữ liệu có thứ bậc (vd: $S < M < L < XL$).
  - **One-Hot Encoding:** Tạo cột giả $(0, 1)$ cho dữ liệu định danh không có thứ bậc (tránh hiểu lầm về thứ tự).
- **Tránh rò rỉ dữ liệu (Data Leak):** Tuyệt đối không cho máy nhìn thấy tập Test trước. Dữ liệu Test không được lẫn vào quá trình huấn luyện hay fit bộ chuẩn hóa.

---

## 📋 Nội dung các bài tập

### Bài 1: Làm sạch dữ liệu
- Tạo DataFrame chứa giá trị thiếu `NaN` và ngoại lệ.
- Thống kê số lượng giá trị thiếu bằng `isnull().sum()`.
- Điền giá trị thiếu bằng giá trị trung bình (`mean`).
- Phát hiện và xử lý ngoại lệ theo quy tắc $3\sigma$.

### Bài 2: Chuẩn hóa dữ liệu (Scaling)
- Tạo mảng dữ liệu đa thang đo: Cột Tuổi (20-60) và Cột Thu nhập (10tr-100tr).
- Chuẩn hóa bằng `StandardScaler` và `MinMaxScaler` để so sánh sự khác biệt.

### Bài 3: Chia tập dữ liệu Train / Test (Tập Diabetes)
- Sử dụng tập dữ liệu `load_diabetes` từ Scikit-learn.
- Phân chia theo tỷ lệ **80% Train - 20% Test** (`random_state=42`).
- In kích thước tập gốc và 2 tập con.

### Bài 4: Quy trình ML hoàn chỉnh với Pipeline (Tập Digits)
- Sử dụng tập dữ liệu ảnh chữ số viết tay `load_digits`.
- Đóng gói quy trình khép kín bằng `Pipeline(StandardScaler -> LogisticRegression)`.
- Đánh giá trên tập Test: Độ chính xác đạt **`97.22%`**.

### Bài 5: Phần mở rộng (Theo vở ghi Note 2 - Mã hóa LabelEncoder)
- Mã hóa kích thước quần áo `["S", "M", "L", "XL", "M", "S"]` bằng `LabelEncoder`.
- Giữ nguyên trật tự kích thước có thứ bậc ($S < M < L < XL$).

---

## 🚀 Hướng dẫn chạy

```bash
python lab2.py
```
