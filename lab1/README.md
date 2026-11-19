# Lab 1: Làm Quen Với NumPy, Pandas & Phân Loại Bài Toán Machine Learning

---

## 🎯 Mục tiêu
- Nắm vững thao tác cơ bản với thư viện **NumPy** và **Pandas**.
- Phân biệt các nhóm bài toán trong Machine Learning: Supervised (Có giám sát) vs Unsupervised (Không giám sát); Hồi quy (Regression) vs Phân loại (Classification).
- Thử nghiệm mô hình Machine Learning đầu tiên (`LinearRegression`) và khám phá tập dữ liệu thực tế (`Iris`, `Digits`).

---

## 📋 Nội dung các bài tập

### Bài 1: Thao tác với NumPy và Pandas
- Tạo mảng NumPy 1 chiều chứa các số nguyên từ 1 đến 10.
- Tạo DataFrame lưu trữ danh sách sinh viên gồm các trường: `Họ tên`, `Tuổi`, `Điểm toán`.
- Hiển thị thông tin tổng quan bằng `info()` và tính điểm toán trung bình.

### Bài 2: Phân loại các bài toán Machine Learning thực tế
1. **Dự đoán giá nhà dựa trên diện tích:** Supervised Learning (Bài toán Hồi quy - Regression).
2. **Gom nhóm khách hàng theo hành vi mua sắm:** Unsupervised Learning (Bài toán Phân cụm - Clustering).
3. **Phân loại email Spam/không Spam:** Supervised Learning (Bài toán Phân loại - Classification).
4. **Nhận diện chữ viết tay từ hình ảnh:** Supervised Learning (Bài toán Phân loại - Classification).

### Bài 3: Chạy thử Linear Regression trên tập dữ liệu Iris
- Tải tập dữ liệu hoa Iris từ `sklearn.datasets.load_iris`.
- Khởi tạo và huấn luyện mô hình `LinearRegression`.
- Dự đoán và hiển thị kết quả cho 5 mẫu đầu tiên.

### Bài 4: Khám phá tập dữ liệu ảnh chữ số viết tay Digits
- Tải tập dữ liệu `load_digits`.
- Kiểm tra kích thước ma trận đặc trưng `data` và nhãn `target`.
- In ma trận điểm ảnh $8 \times 8$ của mẫu đầu tiên.

---

## 🚀 Hướng dẫn chạy

```bash
python lab1.py
```
