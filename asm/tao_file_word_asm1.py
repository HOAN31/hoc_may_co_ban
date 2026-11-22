# -*- coding: utf-8 -*-
"""
Script tự động tạo file Word (.docx) nộp Assignment 1 môn Học máy cơ bản (ITA202)
Đề tài 11: Phân tích hành vi và dự báo rời bỏ dịch vụ của khách hàng (Telco Customer Churn)
Tuân thủ đầy đủ chuẩn học thuật và đối chiếu chi tiết Vở ghi (Note 1 -> Note 4)
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

    # Thiết lập lề trang chuẩn A4
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
    r_school = p_header.add_run("BỘ GIÁO DỤC VÀ ĐÀO TẠO\nTRƯỜNG CAO ĐẲNG FPT POLYTECHNIC\nBỘ MÔN CÔNG NGHỆ THÔNG TIN\n")
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

    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_div.paragraph_format.space_after = Pt(10)
    r_div = p_div.add_run("—" * 50)
    r_div.font.color.rgb = RGBColor(180, 180, 180)

    # ---------------------------------------------------------
    # BẢNG TỔNG HỢP 3 LINK BẮT BUỘC NỘP BÀI (HIGHLIGHT ĐẦU TIÊN)
    # ---------------------------------------------------------
    p_link_head = doc.add_paragraph()
    r_lh = p_link_head.add_run("★ BẢNG TỔNG HỢP 3 LIÊN KẾT NỘP BÀI ASSIGNMENT 1 (BẮT BUỘC)")
    r_lh.font.size = Pt(13)
    r_lh.font.bold = True
    r_lh.font.color.rgb = RGBColor(192, 0, 0)

    p_link_desc = doc.add_paragraph()
    p_link_desc.paragraph_format.space_after = Pt(6)
    p_link_desc.add_run("Theo đúng yêu cầu của Giảng viên, dưới đây là 3 liên kết trực tiếp trên GitHub để chấm điểm:")

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

    link_rows = [
        (
            "LINK 1\n(Mã nguồn)",
            "Mã nguồn Python thực hiện toàn bộ quy trình tiền xử lý dữ liệu và huấn luyện mô hình cơ sở tuân thủ Vở ghi",
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

        p0 = row.cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r0 = p0.add_run(m_id)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.color.rgb = RGBColor(0, 51, 102)

        p1 = row.cells[1].paragraphs[0]
        r1 = p1.add_run(m_desc)
        r1.font.size = Pt(9.5)

        p2 = row.cells[2].paragraphs[0]
        add_hyperlink(p2, m_url, m_text, color="0056B3", underline=True)
        p2_sub = row.cells[2].add_paragraph()
        p2_sub.paragraph_format.space_before = Pt(4)
        r2_sub = p2_sub.add_run(f"({m_url})")
        r2_sub.font.size = Pt(8)
        r2_sub.font.color.rgb = RGBColor(120, 120, 120)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # ---------------------------------------------------------
    # PHẦN ĐẶC BIỆT: ĐỐI CHIẾU VÀ TUÂN THỦ VỞ GHI CỦA GIẢNG VIÊN (NOTE 1 -> 4)
    # ---------------------------------------------------------
    h_vg = doc.add_heading("I. BẢNG ĐỐI CHIẾU & TUÂN THỦ 100% VỞ GHI (NOTE 1 -> NOTE 4)", level=1)
    h_vg.paragraph_format.space_before = Pt(12)
    h_vg.paragraph_format.space_after = Pt(6)

    tbl_vg = doc.add_table(rows=6, cols=3)
    tbl_vg.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_vg.autofit = False

    vg_hdrs = ["Vở ghi", "Yêu cầu / Quy tắc lý thuyết của Giảng viên", "Minh chứng tuân thủ trong Assignment 1"]
    for i, title in enumerate(vg_hdrs):
        cell = tbl_vg.rows[0].cells[i]
        cell.text = title
        set_cell_background(cell, "1F497D")
        set_cell_margins(cell, 100, 100, 120, 120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(255, 255, 255)

    vg_data = [
        (
            "Note 1\n(Quy trình & Môi trường)",
            "• Tuân thủ 8 bước chuẩn ML: Xác định -> Thu thập -> Tiền xử lý -> EDA -> Chọn Model -> Huấn luyện -> Đánh giá -> Triển khai.\n• Sử dụng thư viện chuẩn: NumPy, Pandas, Matplotlib, Scikit-learn.",
            "Thực hiện đầy đủ 8 bước từ EDA đến huấn luyện Baseline Logistic Regression. Sử dụng đúng bộ 4 thư viện chuẩn mã nguồn mở."
        ),
        (
            "Note 2\n(Xử lý Missing & Outlier)",
            "• 'Ký hiệu NaN, None, ô trống -> Xóa hoặc điền Mean, Median, Mode, hằng số'.\n• 'Xác định ngoại lệ bằng biểu đồ Boxplot hoặc quy tắc IQR'.",
            "Phát hiện 11 ô trống ' ' tại TotalCharges có tenure=0 -> Điền hằng số 0.0 theo nghiệp vụ. Vẽ Boxplot cho tenure và MonthlyCharges để kiểm soát ngoại lệ."
        ),
        (
            "Note 2\n(Mã hóa & Chuẩn hóa)",
            "• 'One-Hot Encoding: tạo cột giả (0, 1) cho dữ liệu định danh để tránh hiểu lầm thứ tự'.\n• 'StandardScaler (đưa về trung bình 0, độ lệch 1) tránh cột lớn lấn át cột nhỏ'.",
            "Mã hóa nhị phân 0/1 cho gender, Partner... Áp dụng pd.get_dummies(drop_first=True) cho 10 biến danh mục. Chuẩn hóa StandardScaler cho tenure, MonthlyCharges, TotalCharges."
        ),
        (
            "Note 2\n(Chống Data Leakage)",
            "• 'Chia dữ liệu Train (70-80%), Test (20-30%) cố định random_state'.\n• 'CHÚ Ý: Không bao giờ cho máy nhìn thấy tập test trước. Rò rỉ dữ liệu (Data Leak) là test bị lẫn vào huấn luyện'.",
            "Chia 80% Train, 20% Test (random_state=42, stratify=y). Khi huấn luyện mô hình, fit scaler độc lập trên X_train và transform sang X_test, loại bỏ 100% nguy cơ Data Leakage."
        ),
        (
            "Note 4\n(Logistic Regression & Đánh giá)",
            "• 'Dùng hàm Sigmoid ép về xác suất [0, 1], ngưỡng 0.5'.\n• 'Đánh giá toàn diện qua Confusion Matrix, Precision (không bắt nhầm), Recall (không bỏ sót), F1-Score thay vì chỉ dùng Accuracy'.\n• 'Vẽ đường cong ROC và chỉ số AUC (càng gần 1 càng tốt)'.",
            "Huấn luyện LogisticRegression(max_iter=1000). Đánh giá: Accuracy=80.62%, Precision=0.6593, Recall=0.5588, F1=0.6049, ROC-AUC=0.8422. Vẽ riêng biểu đồ Đường cong ROC và Ma trận nhầm lẫn 2x2."
        )
    ]

    vg_widths = [Inches(1.4), Inches(2.8), Inches(2.6)]
    for r_idx, (v_note, v_req, v_sol) in enumerate(vg_data):
        row = tbl_vg.rows[r_idx + 1]
        for c_idx, w in enumerate(vg_widths):
            row.cells[c_idx].width = w
            set_cell_margins(row.cells[c_idx], 80, 80, 100, 100)
            if r_idx % 2 == 1:
                set_cell_background(row.cells[c_idx], "F9FAFB")

        p0 = row.cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r0 = p0.add_run(v_note)
        r0.bold = True
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(0, 51, 102)

        p1 = row.cells[1].paragraphs[0]
        p1.add_run(v_req).font.size = Pt(9)

        p2 = row.cells[2].paragraphs[0]
        p2.add_run(v_sol).font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # ---------------------------------------------------------
    # PHẦN II: TỔNG QUAN ĐỀ TÀI & DỮ LIỆU
    # ---------------------------------------------------------
    h1 = doc.add_heading("II. TỔNG QUAN ĐỀ TÀI VÀ TẬP DỮ LIỆU", level=1)
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
        "   - Thông tin hợp đồng & thanh toán: tenure (thâm niên), Contract, PaperlessBilling, PaymentMethod, "
        "MonthlyCharges (cước tháng), TotalCharges (tổng cước tích lũy).\n"
        "   - Biến mục tiêu (Target Label): Churn (No: 5,174 mẫu ~ 73.46%; Yes: 1,869 mẫu ~ 26.54%)."
    )

    # ---------------------------------------------------------
    # PHẦN III: KẾT QUẢ THỰC THI & MINH CHỨNG HÌNH ẢNH
    # ---------------------------------------------------------
    h3 = doc.add_heading("III. KẾT QUẢ THỰC NGHIỆM VÀ ẢNH CHỤP MINH CHỨNG", level=1)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)

    p_exp = doc.add_paragraph()
    p_exp.paragraph_format.line_spacing = 1.15
    p_exp.add_run(
        "Dữ liệu sau khi tiền xử lý hoàn thiện đạt kích thước 7,043 dòng x 31 cột (Link 2). "
        "Kết quả đánh giá mô hình Baseline Logistic Regression (chuẩn Note 4) như sau:"
    )

    tbl_res = doc.add_table(rows=6, cols=3)
    tbl_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_hdr = ["Chỉ số đánh giá", "Kết quả thực nghiệm", "Ý nghĩa chuyên môn theo Vở ghi"]
    for i, h in enumerate(res_hdr):
        tbl_res.rows[0].cells[i].text = h
        set_cell_background(tbl_res.rows[0].cells[i], "2B5C8F")
        p = tbl_res.rows[0].cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(255, 255, 255)

    res_data = [
        ("Độ chính xác (Accuracy)", "80.62%", "Mô hình dự đoán chính xác trên 80% tổng số khách hàng."),
        ("Precision ('Không bắt nhầm')", "0.6593", "Trong số các khách hàng bị dự báo rời mạng, có 65.93% thực sự rời bỏ."),
        ("Recall ('Không bỏ sót')", "0.5588", "Bắt trúng được 55.88% tổng số khách hàng có nguy cơ rời mạng."),
        ("F1-Score (Cân bằng)", "0.6049", "Điểm trung bình điều hòa giữa Precision và Recall trên bài toán nhãn lệch."),
        ("ROC-AUC Score (Khả năng phân tách)", "0.8422", "Khả năng phân tách giữa nhóm ở lại và nhóm rời mạng đạt mức rất tốt (> 0.8).")
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

    # Chèn ảnh chụp Dashboard minh chứng (Link 3)
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

    # Chèn ảnh đường cong ROC theo Note 4
    p_img_title_roc = doc.add_paragraph()
    p_img_title_roc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_roc = p_img_title_roc.add_run("Hình 2: Biểu đồ Đường cong ROC & Diện tích dưới đường cong AUC (Theo Note 4)")
    r_roc.bold = True
    r_roc.font.italic = True
    r_roc.font.size = Pt(10)

    img_roc_path = os.path.join("asm", "images", "duong_cong_roc_auc.png")
    if os.path.exists(img_roc_path):
        p_img_roc = doc.add_paragraph()
        p_img_roc.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img_roc.paragraph_format.space_after = Pt(12)
        doc.add_picture(img_roc_path, width=Inches(5.0))

    # Chèn ảnh Boxplot theo Note 2
    p_img_title_bp = doc.add_paragraph()
    p_img_title_bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_bp = p_img_title_bp.add_run("Hình 3: Biểu đồ Boxplot khảo sát phân phối và ngoại lệ (Theo Note 2)")
    r_bp.bold = True
    r_bp.font.italic = True
    r_bp.font.size = Pt(10)

    img_bp_path = os.path.join("asm", "images", "eda_charges_tenure.png")
    if os.path.exists(img_bp_path):
        p_img_bp = doc.add_paragraph()
        p_img_bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img_bp.paragraph_format.space_after = Pt(12)
        doc.add_picture(img_bp_path, width=Inches(6.0))

    # ---------------------------------------------------------
    # PHẦN IV: KẾ HOẠCH TRIỂN KHAI GIAI ĐOẠN 2
    # ---------------------------------------------------------
    h4 = doc.add_heading("IV. KẾ HOẠCH TRIỂN KHAI GIAI ĐOẠN 2 (MACHINE LEARNING NÂNG CAO)", level=1)
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

    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(14)
    r_end = p_end.add_run("Người lập báo cáo: Sinh viên thực hiện Assignment\nNgày hoàn thành: Tháng 11/2026")
    r_end.font.italic = True
    r_end.font.color.rgb = RGBColor(100, 100, 100)

    # Lưu file
    output_docx_path = os.path.join("asm", "ASSIGNMENT_1_HOAN31.docx")
    doc.save(output_docx_path)
    print(f"Đã tạo file Word báo cáo thành công tại: {output_docx_path}")

    root_docx_path = "ASSIGNMENT_1_HOAN31.docx"
    import shutil
    shutil.copyfile(output_docx_path, root_docx_path)
    print(f"Đã sao chép bản nộp ra thư mục gốc: {root_docx_path}")

if __name__ == "__main__":
    create_word_report()
