# Assignment: Dự Án Machine Learning Hoàn Chỉnh (Titanic Survival)

Kho lưu trữ mã nguồn, dữ liệu và báo cáo cho bài **Assignment môn Học máy cơ bản (ITA202)** - FPT Polytechnic.

---

## 📁 Cấu trúc thư mục Assignment

```text
asm/
├── data/
│   └── titanic.csv                     # Tập dữ liệu Titanic gốc (891 dòng, 12 cột)
├── images/
│   ├── 1_ty_le_song_sot_eda.png        # Biểu đồ phân tích khám phá dữ liệu (EDA)
│   ├── 2_confusion_matrix.png          # Ma trận nhầm lẫn (Confusion Matrix)
│   └── 3_roc_curve.png                 # Đường cong ROC và chỉ số AUC
├── giai_doan_1.py                      # Mã nguồn hoàn chỉnh thực hiện Giai đoạn 1
├── BAO_CAO_GIAI_DOAN_1.md              # Báo cáo chi tiết kết quả Giai đoạn 1
└── README.md                           # Hướng dẫn tổng quan
```

---

## 📌 Nội dung Giai đoạn 1: Chuẩn bị dữ liệu & Xây dựng mô hình cơ sở

1. **Khám phá dữ liệu (EDA):** `shape`, `info`, `describe`, thống kê tỷ lệ sống sót.
2. **Làm sạch dữ liệu:**
   - Xóa cột `Cabin` (thiếu 77.1%), xóa các cột định danh (`PassengerId`, `Name`, `Ticket`).
   - Điền thiếu `Age` bằng `Median` (28 tuổi).
   - Điền thiếu `Embarked` bằng `Mode` ('S').
   - Xử lý ngoại lệ cột `Fare` bằng phương pháp cắt biên (Winsorization 99th percentile).
3. **Tiền xử lý (Preprocessing):**
   - Mã hóa `OneHotEncoder` cho các cột phân loại (`Sex`, `Embarked`, `Pclass`).
   - Chuẩn hóa `StandardScaler` cho các cột số (`Age`, `Fare`, `SibSp`, `Parch`).
   - Chia tập Train/Test theo tỷ lệ 80/20 có phân tầng nhãn `stratify=y`.
4. **Huấn luyện mô hình cơ sở:**
   - Đóng gói toàn bộ qua `Pipeline(ColumnTransformer -> LogisticRegression)`.
5. **Đánh giá & Trực quan hóa:**
   - **Accuracy:** `80.45%`
   - **Precision:** `79.31%`
   - **Recall:** `66.67%`
   - **F1-Score:** `0.7244`
   - **ROC-AUC:** `0.8431`
   - Phân tích chi tiết Ma trận nhầm lẫn và đường cong ROC.

---

## 🚀 Hướng dẫn chạy

```bash
python asm/giai_doan_1.py
```
