

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LinearRegression


# Bài 1: Làm quen với NumPy và Pandas
print("=== Bài 1: NumPy và Pandas ===")
numbers = np.arange(1, 11)
print("Mảng NumPy từ 1 đến 10:", numbers)

students = pd.DataFrame(
    {
        "Họ tên": ["An", "Bình", "Chi", "Dũng", "Hà"],
        "Tuổi": [20, 21, 19, 22, 20],
        "Điểm toán": [8.5, 7.0, 9.0, 6.5, 8.0],
    }
)
print("\nThông tin 5 sinh viên:")
print(students)
print("\nTổng quan DataFrame:")
students.info()
print(f"\nĐiểm toán trung bình: {students['Điểm toán'].mean():.2f}")

# Bài 2: Phân loại bài toán Machine Learning
print("\n=== Bài 2: Phân loại bài toán Machine Learning ===")
print("1. Dự đoán giá nhà dựa trên diện tích: Supervised (hồi quy).")
print("2. Gom nhóm khách hàng theo hành vi mua sắm: Unsupervised (phân cụm).")
print("3. Phân loại email Spam/không Spam: Supervised (phân loại).")
print("4. Nhận diện chữ viết tay từ hình ảnh: Supervised (phân loại).")

# Bài 3: Chạy thử Linear Regression trên dữ liệu Iris.
# Iris là bài toán phân loại; Linear Regression ở đây chỉ để minh họa
# cách fit và predict, nên đầu ra là số liên tục chứ không phải nhãn lớp.
print("\n=== Bài 3: Linear Regression với Iris ===")
iris = load_iris()
model = LinearRegression()
model.fit(iris.data, iris.target)
predictions = model.predict(iris.data[:5])
print("Kích thước dữ liệu Iris:", iris.data.shape)
print("Nhãn thực tế của 5 mẫu đầu:", iris.target[:5])
print("Giá trị dự đoán của 5 mẫu đầu:", np.round(predictions, 2))

# Bài 4: Thu thập và kiểm tra dữ liệu Digits.
print("\n=== Bài 4: Kiểm tra dữ liệu Digits ===")
digits = load_digits()
print("Kích thước dữ liệu:", digits.data.shape)
print("Kích thước nhãn:", digits.target.shape)
print("Nhãn của mẫu đầu tiên:", digits.target[0])
print("Giá trị pixel của mẫu đầu tiên (ma trận 8x8):")
print(digits.images[0])



