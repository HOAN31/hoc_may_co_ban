# -*- coding: utf-8 -*-
"""
Script tự động tạo file Word (.docx) nộp Assignment 1 môn Học máy cơ bản (ITA202)
Đề tài 11: Phân tích hành vi và dự báo rời bỏ dịch vụ của khách hàng (Telco Customer Churn)
"""

import os
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_hyperlink(paragraph, url, text, color="004B87", underline=True):
    part = paragraph.part
    r_id = part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    hyperlink = parse_xml(
        f'<w:hyperlink xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'r:id="{r_id}"/>'
    )
    new_run = parse_xml(
        f'<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:rPr/>'
        f'<w:t>{text}</w:t>'
        f'</w:r>'
    )
    
    rPr = new_run.find(qn('w:rPr'))
    if color:
        c = parse_xml(f'<w:color xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:val="{color}"/>')
        rPr.append(c)
    if underline:
        u = parse_xml(f'<w:u xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:val="single"/>')
        rPr.append(u)
        
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

def create_word_report():
    doc = Document()

    # Thiết lập lề trang chuẩn (A4: 2cm lề)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # ---------------------------------------------------------
    # TIÊU ĐỀ ĐẦU TRANG
    # ---------------------------------------------------------
    p_header = doc.add_paragraph()
    p_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_school = p_header.add_run("BỘ GIÁO DỤC VÀ ĐÀO TẠO\nTRƯỜNG CAO ĐẲNG FPT POLYTECHNIC\n")
    r_school.font.size = Pt(11)
    r_school.font.bold = True
    r_school.font.color.rgb = RGBColor(100, 100, 100)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("BÁO CÁO BÀI TẬP LỚN / ASSIGNMENT GIAI ĐOẠN 1\n")
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(227, 82, 5) # Màu cam FPT

    r_sub = p_title.add_run("MÔN: HỌC MÁY CƠ BẢN (ITA202)\n")
    r_sub.font.size = Pt(14)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0, 51, 102)

    r_topic = p_title.add_run("ĐỀ TÀI 11: PHÂN TÍCH HÀNH VI VÀ DỰ BÁO RỜI BỎ DỊCH VỤ CỦA KHÁCH HÀNG\n(TELCO CUSTOMER CHURN ANALYSIS)")
    r_topic.font.size = Pt(13)
    r_topic.font.bold = True
    r_topic.font.italic = True
    r_topic.font.color.rgb = RGBColor(43, 92, 143)

    # Đường kẻ ngang ngăn cách
    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_div.paragraph_format.space_after = Pt(10)
    r_div = p_div.add_run("—" * 50)
    r_div.font.color.rgb = RGBColor(180, 180, 180)

    # ---------------------------------------------------------
    # THÔNG TIN SINH VIÊN
    # ---------------------------------------------------------
    tbl_info = doc.add_table(rows=4, cols=2)
    tbl_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Môn học:", "Học máy cơ bản - ITA202"),
        ("Đề tài thực hiện:", "Đề tài 11 - Phân tích & Dự báo khách hàng rời mạng (Telco Churn)"),
        ("Tập dữ liệu:", "Telco Customer Churn (7,043 bản ghi x 21 thuộc tính gốc)"),
        ("Repository GitHub:", "https://github.com/HOAN31/hoc_may_co_ban")
    ]
    for row_idx, (k, v) in enumerate(info_data):
        row = tbl_info.rows[row_idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.width = Inches(2.2)
        cell_v.width = Inches(4.5)
        
        p_k = cell_k.paragraphs[0]
        r_k = p_k.add_run(k)
        r_k.bold = True
        r_k.font.size = Pt(10.5)
        
        p_v = cell_v.paragraphs[0]
        r_v = p_v.add_run(v)
        r_v.font.size = Pt(10.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ---------------------------------------------------------
    # BẢNG TỔNG HỢP 3 LINK BẮT BUỘC NỘP BÀI (HIGHLIGHT ĐẶC BIỆT)
    # ---------------------------------------------------------
    p_link_head = doc.add_paragraph()
    r_lh = p_link_head.add_run("★ BẢNG TỔNG HỢP 3 LIÊN KẾT NỘP BÀI ASSIGNMENT 1")
    r_lh.font.size = Pt(13)
    r_lh.font.bold = True
    r_lh.font.color.rgb = RGBColor(192, 0, 0)

    p_link_desc = doc.add_paragraph()
    p_link_desc.paragraph_format.space_after = Pt(6)
    p_link_desc.add_run("Theo đúng yêu cầu của Giảng viên, dưới đây là 3 liên kết trực tiếp để chấm điểm:")

    table_links = doc.add_table(rows=4, cols=3)
    table_links.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_links.autofit = False

    headers = ["Mục nộp", "Nội dung chi tiết", "Đường dẫn liên kết trực tiếp (Click để mở)"]
    hdr_cells = table_links.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "003366")
        set_cell_margins(hdr_cells[i], 120, 120, 150, 150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(255, 255, 255)

    # Dữ liệu 3 links
    link_rows = [
        (
            "LINK 1\n(Mã nguồn)",
            "Mã nguồn Python thực hiện toàn bộ quy trình tiền xử lý dữ liệu và huấn luyện mô hình cơ sở",
            "https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/tien_xu_ly_telco.py",
            "Xem mã nguồn tại GitHub: asm/tien_xu_ly_telco.py"
        ),
        (
            "LINK 2\n(Dữ liệu sau xử lý)",
            "Tập dữ liệu Telco Customer Churn đã làm sạch, mã hóa và chuẩn hóa hoàn thiện (7,043 dòng x 31 cột)",
            "https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/data/telco_churn_preprocessed.csv",
            "Tải dữ liệu tại GitHub: asm/data/telco_churn_preprocessed.csv"
        ),
        (
            "LINK 3\n(Ảnh chụp kết quả)",
            "Ảnh chụp Dashboard tổng hợp minh chứng kết quả chạy tiền xử lý và độ chính xác mô hình cơ sở",
            "https://github.com/HOAN31/hoc_may_co_ban/blob/main/asm/images/ket_qua_chay_tien_xu_ly.png",
            "Xem ảnh kết quả tại GitHub: asm/images/ket_qua_chay_tien_xu_ly.png"
        )
    ]

    col_widths = [Inches(1.5), Inches(2.5), Inches(2.8)]
    for r_idx, (m_id, m_desc, m_url, m_text) in enumerate(link_rows):
        row = table_links.rows[r_idx + 1]
        for c_idx, w in enumerate(col_widths):
            row.cells[c_idx].width = w
            set_cell_margins(row.cells[c_idx], 100, 100, 120, 120)
            if r_idx % 2 == 1:
                set_cell_background(row.cells[c_idx], "F2F5F8")

        # Cột 1: ID
        p0 = row.cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r0 = p0.add_run(m_id)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.color.rgb = RGBColor(0, 51, 102)

        # Cột 2: Mô tả
        p1 = row.cells[1].paragraphs[0]
        r1 = p1.add_run(m_desc)
        r1.font.size = Pt(9.5)

        # Cột 3: Hyperlink
        p2 = row.cells[2].paragraphs[0]
        add_hyperlink(p2, m_url, m_text, color="0056B3", underline=True)
        p2_sub = row.cells[2].add_paragraph()
        p2_sub.paragraph_format.space_before = Pt(4)
        r2_sub = p2_sub.add_run(f"({m_url})")
        r2_sub.font.size = Pt(8)
        r2_sub.font.color.rgb = RGBColor(120, 120, 120)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # ---------------------------------------------------------
    # PHẦN I: TỔNG QUAN ĐỀ TÀI VÀ TẬP DỮ LIỆU
    # ---------------------------------------------------------
    h1 = doc.add_heading("I. TỔNG QUAN ĐỀ TÀI VÀ TẬP DỮ LIỆU", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.line_spacing = 1.2
    p_intro.add_run(
        "1. Tên đề tài: Đề tài 11 - Phân tích hành vi và dự báo rời bỏ dịch vụ của khách hàng (Customer Churn Analysis).\n"
        "2. Bài toán kinh doanh: Trong ngành viễn thông, chi phí thu hút một khách hàng mới cao gấp 5 đến 7 lần so với "
        "chi phí duy trì một khách hàng hiện hữu. Bài toán đặt ra là phân tích các yếu tố ảnh hưởng đến quyết định hủy dịch vụ "
        "và xây dựng mô hình học máy phân loại nguy cơ rời mạng (Churn: Yes/No) để doanh nghiệp có chính sách chăm sóc kịp thời.\n"
        "3. Nguồn dữ liệu: Tập dữ liệu Telco Customer Churn (IBM Developer / Kaggle blastchar).\n"
        "4. Kích thước tập dữ liệu gốc: 7,043 dòng và 21 cột thuộc tính, bao gồm:\n"
        "   - Cột định danh: customerID.\n"
        "   - Thông tin nhân khẩu học: gender, SeniorCitizen, Partner, Dependents.\n"
        "   - Dịch vụ khách hàng đăng ký: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
        "DeviceProtection, TechSupport, StreamingTV, StreamingMovies.\n"
        "   - Thông tin hợp đồng & thanh toán: tenure (số tháng gắn bó), Contract, PaperlessBilling, PaymentMethod, "
        "MonthlyCharges (cước tháng), TotalCharges (tổng cước tích lũy).\n"
        "   - Biến mục tiêu (Target Label): Churn (No: 5,174 mẫu ~ 73.46%; Yes: 1,869 mẫu ~ 26.54%)."
    )

    # ---------------------------------------------------------
    # PHẦN II: CÁC BƯỚC TIỀN XỬ LÝ DỮ LIỆU CHI TIẾT
    # ---------------------------------------------------------
    h2 = doc.add_heading("II. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)", level=1)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)

    steps = [
        ("Bước 1: Khám phá dữ liệu (EDA)", 
         "Kiểm tra tổng quan 7,043 dòng x 21 cột, thống kê tỷ lệ nhãn Churn. Nhận diện các thuộc tính dạng chuỗi (Object) "
         "cần chuyển đổi và kiểm tra giá trị dị biệt."),
        ("Bước 2: Loại bỏ thuộc tính không mang giá trị dự báo",
         "Cột 'customerID' là mã định danh ngẫu nhiên do hệ thống sinh ra. Thuộc tính này không chứa quy luật hành vi khách hàng, "
         "nếu đưa vào mô hình sẽ gây nhiễu và overfitting. Vì vậy, ta tiến hành loại bỏ (drop) cột này."),
        ("Bước 3: Phát hiện và xử lý lỗi dữ liệu / Missing Values",
         "Cột 'TotalCharges' ban đầu mang kiểu dữ liệu object thay vì float. Kiểm tra phát hiện có 11 dòng dữ liệu chứa ký tự khoảng "
         "trắng ' '. Toàn bộ 11 dòng này đều có thâm niên tenure = 0 (khách hàng mới đăng ký trong tháng đầu tiên nên chưa thanh toán). "
         "Giải pháp: Ép kiểu sang float bằng pd.to_numeric(errors='coerce') và điền giá trị 0.0 hợp lý về mặt nghiệp vụ."),
        ("Bước 4: Mã hóa biến mục tiêu và biến nhị phân (Binary Encoding)",
         "Chuyển đổi nhãn mục tiêu Churn: 'No' -> 0 (khách hàng ở lại), 'Yes' -> 1 (khách hàng rời bỏ).\n"
         "Mã hóa các biến nhị phân có 2 lựa chọn (gender, Partner, Dependents, PhoneService, PaperlessBilling) sang dạng 0 và 1."),
        ("Bước 5: Mã hóa các biến danh mục (One-Hot Encoding)",
         "Với 10 biến phân loại có từ 3 nhóm trở lên (như InternetService, Contract, PaymentMethod...), áp dụng kỹ thuật "
         "One-Hot Encoding với tham số drop_first=True nhằm tránh hiện tượng bẫy biến giả (Dummy Variable Trap) và loại bỏ "
         "tính đa cộng tuyến (Multicollinearity)."),
        ("Bước 6: Chuẩn hóa đặc trưng số (Feature Scaling)",
         "3 biến số liên tục (tenure, MonthlyCharges, TotalCharges) có miền giá trị chênh lệch lớn (ví dụ TotalCharges lên đến hơn 8,000 USD). "
         "Áp dụng StandardScaler để đưa toàn bộ về phân phối chuẩn (Mean = 0, Std = 1), giúp các thuật toán học máy như Logistic Regression, "
         "KNN đạt tốc độ hội tụ nhanh và không bị thiên lệch bởi biến có biên độ lớn."),
        ("Bước 7: Xuất dữ liệu sạch hoàn thiện",
         "Dữ liệu sau khi hoàn tất các bước trên đạt kích thước 7,043 dòng và 31 cột (30 đặc trưng + 1 nhãn mục tiêu). "
         "Được lưu tại file: asm/data/telco_churn_preprocessed.csv và đẩy lên GitHub để lưu trữ.")
    ]

    for title, desc in steps:
        p_step = doc.add_paragraph()
        p_step.paragraph_format.line_spacing = 1.15
        p_step.paragraph_format.space_after = Pt(4)
        r_st = p_step.add_run(f"• {title}: ")
        r_st.bold = True
        r_st.font.color.rgb = RGBColor(0, 51, 102)
        p_step.add_run(desc)

    # ---------------------------------------------------------
    # PHẦN III: KẾT QUẢ THỰC THI & MINH CHỨNG (ẢNH KẾT QUẢ)
    # ---------------------------------------------------------
    h3 = doc.add_heading("III. KẾT QUẢ THỰC NGHIỆM VÀ ẢNH CHỤP MINH CHỨNG", level=1)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)

    p_exp = doc.add_paragraph()
    p_exp.paragraph_format.line_spacing = 1.15
    p_exp.add_run(
        "Để kiểm tra tính hợp lệ và sẵn sàng của tập dữ liệu sau tiền xử lý, dữ liệu đã được chia theo tỷ lệ "
        "80% Train (5,634 dòng) và 20% Test (1,409 dòng) với stratify=y. Sau đó, huấn luyện mô hình cơ sở "
        "Logistic Regression (Giai đoạn 1) và thu được kết quả ấn tượng:"
    )

    # Bảng số liệu mô hình
    tbl_res = doc.add_table(rows=6, cols=3)
    tbl_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_hdr = ["Chỉ số đánh giá", "Kết quả thực nghiệm", "Ý nghĩa chuyên môn"]
    for i, h in enumerate(res_hdr):
        tbl_res.rows[0].cells[i].text = h
        set_cell_background(tbl_res.rows[0].cells[i], "2B5C8F")
        p = tbl_res.rows[0].cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    res_data = [
        ("Độ chính xác (Accuracy)", "80.62%", "Mô hình dự đoán chính xác trên 80% tổng số khách hàng."),
        ("Precision (Rời mạng)", "0.6593", "Trong số các khách hàng bị dự báo rời mạng, có 65.93% thực sự rời bỏ."),
        ("Recall (Rời mạng)", "0.5588", "Bắt trúng được 55.88% tổng số khách hàng có nguy cơ rời mạng."),
        ("F1-Score", "0.6049", "Điểm cân hòa giữa Precision và Recall trên bài toán nhãn lệch."),
        ("ROC-AUC Score", "0.8422", "Khả năng phân tách giữa nhóm ở lại và nhóm rời mạng đạt mức rất tốt (> 0.8).")
    ]

    for idx, (m_name, m_val, m_note) in enumerate(res_data):
        row = tbl_res.rows[idx + 1]
        row.cells[0].paragraphs[0].add_run(m_name).bold = True
        p_val = row.cells[1].paragraphs[0]
        p_val.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_v = p_val.add_run(m_val)
        r_v.bold = True
        r_v.font.color.rgb = RGBColor(192, 0, 0)
        row.cells[2].paragraphs[0].add_run(m_note)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Chèn ảnh chụp kết quả tiền xử lý (Link 3)
    p_img_title1 = doc.add_paragraph()
    p_img_title1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_it1 = p_img_title1.add_run("Hình 1: Ảnh chụp Dashboard minh chứng kết quả chạy tiền xử lý & Đánh giá mô hình (Link 3)")
    r_it1.bold = True
    r_it1.font.italic = True
    r_it1.font.size = Pt(10)

    img_proof_path = os.path.join("asm", "images", "ket_qua_chay_tien_xu_ly.png")
    if os.path.exists(img_proof_path):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_after = Pt(12)
        doc.add_picture(img_proof_path, width=Inches(6.5))

    # Chèn biểu đồ phân phối Churn
    p_img_title2 = doc.add_paragraph()
    p_img_title2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_it2 = p_img_title2.add_run("Hình 2: Phân phối nhãn mục tiêu Churn và các đặc trưng quan trọng")
    r_it2.bold = True
    r_it2.font.italic = True
    r_it2.font.size = Pt(10)

    img_dist_path = os.path.join("asm", "images", "eda_churn_distribution.png")
    if os.path.exists(img_dist_path):
        doc.add_picture(img_dist_path, width=Inches(6.5))

    # ---------------------------------------------------------
    # PHẦN IV: KẾ HOẠCH TRIỂN KHAI GIAI ĐOẠN 2
    # ---------------------------------------------------------
    h4 = doc.add_heading("IV. KẾ HOẠCH TRIỂN KHAI GIAI ĐOẠN 2 (MACHINE LEARNING CHUYÊN SÂU)", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    p_plan = doc.add_paragraph()
    p_plan.paragraph_format.line_spacing = 1.2
    p_plan.add_run(
        "Dựa trên tập dữ liệu đã tiền xử lý hoàn thiện tại Giai đoạn 1, nhóm sẽ thực hiện các nội dung sau ở Giai đoạn 2:\n"
        "1. Thử nghiệm các mô hình học máy nâng cao: K-Nearest Neighbors (KNN), Cây quyết định (Decision Tree), "
        "và Rừng ngẫu nhiên (Random Forest).\n"
        "2. Tinh chỉnh siêu tham số (Hyperparameter Tuning): Ứng dụng GridSearchCV kết hợp K-Fold Cross-Validation (K=5) "
        "để tìm bộ tham số tối ưu (max_depth, n_estimators, n_neighbors...).\n"
        "3. Xử lý mất cân bằng lớp (Imbalanced Data): Khảo sát kỹ thuật điều chỉnh trọng số lớp class_weight='balanced' hoặc SMOTE.\n"
        "4. Đánh giá và so sánh toàn diện: So sánh các mô hình dựa trên Confusion Matrix, ROC-AUC và F1-Score.\n"
        "5. Đóng gói mô hình: Lưu trữ mô hình tối ưu ra file định dạng .pkl sử dụng thư viện joblib để triển khai ứng dụng thực tế."
    )

    # Lời kết
    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(14)
    r_end = p_end.add_run("Người lập báo cáo: Sinh viên thực hiện Assignment\nNgày hoàn thành: Tháng 11/2026")
    r_end.font.italic = True
    r_end.font.color.rgb = RGBColor(100, 100, 100)

    # Lưu file
    output_docx_path = os.path.join("asm", "ASSIGNMENT_1_HOAN31.docx")
    doc.save(output_docx_path)
    print(f"Đã tạo file Word báo cáo thành công tại: {output_docx_path}")

    # Tạo bản sao ra thư mục gốc để người dùng lấy ngay
    root_docx_path = "ASSIGNMENT_1_HOAN31.docx"
    import shutil
    shutil.copyfile(output_docx_path, root_docx_path)
    print(f"Đã sao chép bản nộp ra thư mục gốc: {root_docx_path}")

if __name__ == "__main__":
    create_word_report()
