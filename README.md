# AdaFace - Inference Repo

**Mục tiêu**  
Repo này tải mô hình đã huấn luyện sẵn, căn chỉnh khuôn mặt bằng MTCNN và trích xuất vector đặc trưng (embedding).

---

## Tham khảo & Liên kết

- **Bài báo gốc:** [*AdaFace: Quality Adaptive Margin for Face Recognition* (CVPR 2022)](https://arxiv.org/abs/2204.00964)  
  - PDF: [Bản chính thức CVPR](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.pdf)
- **Kho mã gốc:** [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)

---

## Mô tả Pipeline 

Quy trình xử lý trong repo này gồm:

1. **Phát hiện & căn chỉnh khuôn mặt:**  
   Dùng **MTCNN** để phát hiện khuôn mặt và 5 điểm đặc trưng (landmarks), sau đó căn chỉnh về kích thước chuẩn (112×112).

2. **Trích xuất đặc trưng:**  
   Ảnh khuôn mặt sau khi căn chỉnh được đưa qua **mô hình AdaFace đã huấn luyện sẵn** để sinh ra vector embedding 512 chiều.

3. **So sánh embedding:**  
   Dùng **độ tương đồng cosine** để xác thực hoặc nhận dạng khuôn mặt.

4. **Kết quả đầu ra:**  
   Vector embedding và ảnh đã căn chỉnh sẽ được lưu trong thư mục `./results`.

---

## Tổng quan về mô hình

**AdaFace** giới thiệu một hàm mất mát (loss) có **biên (margin) thích ứng theo chất lượng ảnh**.

- Khác với ArcFace (margin cố định), AdaFace dùng **chuẩn vector đặc trưng (feature norm)** để đánh giá chất lượng ảnh.  
- Ảnh chất lượng cao (feature norm lớn) → biên phân tách lớn hơn → phân biệt tốt hơn.  
- Ảnh chất lượng thấp (feature norm nhỏ) → biên nhỏ hơn → giảm nhiễu gradient.

**Cấu hình mô hình**
- Backbone: IR-101 hoặc ResNet-100/50/34/18  
- Kích thước embedding: 512  
- Tham số scale \( s = 64 \)  
- Biên \( m \approx 0.4 \)  
- Hệ số thích ứng \( h \approx 0.33 \)

---

## Dữ liệu huấn luyện & Phân tích

### **Các bộ dữ liệu được sử dụng**
| Tập dữ liệu | Số lượng ảnh | Mô tả |
|--------------|---------------|--------|
| MS1MV2 | ~5.8 triệu | Phiên bản làm sạch của MS-Celeb-1M |
| MS1MV3 | ~5.1 triệu | Phiên bản khác của MS-Celeb-1M |
| WebFace4M | ~4.2 triệu | Tập dữ liệu quy mô lớn, sạch |
| WebFace12M (mở rộng) | ~12 triệu | Dùng trong các phiên bản sau (VD: CVLFace) |

### **Phân loại chất lượng dữ liệu**
- **Chất lượng cao (HQ):** LFW, CFP-FP, CPLFW, AgeDB, CALFW  
  → Ảnh rõ, đủ sáng, điều kiện tốt.                (TAR @ FAR = 1e-4: 94.33% -> 99.82%)
- **Chất lượng trung bình (Mixed):** IJB-B, IJB-C  
  → Bao gồm cả ảnh dễ và khó.                      (TAR @ FAR = 1e-4: 96%)
- **Chất lượng thấp (LQ):** IJB-S, TinyFace  
  → Ảnh giám sát, nhỏ, mờ, góc nghiêng lớn.        (TAR @ FAR = 1e-4: 35% -> 51%)

AdaFace được thiết kế để giữ **hiệu năng ổn định trên cả ảnh chất lượng thấp và cao**.

---

## Hiệu năng (Theo bài báo gốc)

| Dữ liệu huấn luyện | Backbone | Bộ đánh giá | Chỉ số | Kết quả |
|--------------------|-----------|--------------|---------|----------|
| MS1MV2 | ResNet-100 | HQ (LFW, CFP-FP, ...) | Accuracy | 96.72% |
| MS1MV2 | ResNet-100 | LQ (IJB-S) | Rank-1 | 51.66% |
| MS1MV3 | ResNet-100 | HQ | Rank-1 | 70.42% |
| WebFace4M | ResNet-100 | LQ | Rank-1 | 35.05% |


---

## Links tải xuống

### 🔹 **Model huấn luyện sẵn**
| Arch | Dataset    | Link                                                                                         |
|------|------------|----------------------------------------------------------------------------------------------|
| R18  | CASIA-WebFace     | [gdrive](https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view?usp=sharing) |
| R18  | VGGFace2     | [gdrive](https://drive.google.com/file/d/1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv/view?usp=sharing) |
| R18  | WebFace4M     | [gdrive](https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing) |
| R50  | CASIA-WebFace     | [gdrive](https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing) |
| R50  | WebFace4M     | [gdrive](https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing) |
| R50  | MS1MV2     | [gdrive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) |
| R100 | MS1MV2     | [gdrive](https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view?usp=sharing) |
| R100 | MS1MV3     | [gdrive](https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing) |
| R100 | WebFace4M  | [gdrive](https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing) |
| R100 | WebFace12M | [gdrive](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing) |

### 🔹 **Mã gốc**
- [https://github.com/mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)

### 🔹 **Dữ liệu huấn luyện**
- **MS1MV2 / MS1MV3:** [Tham khảo bài báo gốc](https://arxiv.org/abs/2204.00964)
- **WebFace4M:** [Thông tin chính thức](https://arxiv.org/abs/2204.00964)

---

## Cài đặt & Chạy thử

### **Cài đặt**
```bash
git clone https://github.com/Itvc0110/AdaFace.git
cd AdaFace
pip install -r requirements.txt

### **Cài đặt mô hình**

gdown --id 1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI -O ./checkpoints/adaface_ir101_ms1mv3.pth

### **Chạy thử**

### 1. Inference trên hình ảnh tĩnh

python main.py --arch ir_101 \
    --checkpoint_path ./checkpoints/adaface_ir101_ms1mv3.pth \
    --input_dir ./input_images \  # Thư mục chứa hình ảnh đầu vào
    --output_dir ./results/ \     # Thư mục lưu embedding (.npy)
    --batch_size 32               # Điều chỉnh theo nhu cầu

### 2. Xây dựng json DB

python main.py --build_db \
    --employees_dir ./employees \  # Thư mục chứa subdirs nhân viên (e.g., employees/ Tên nhân viên /images/*.jpg)
    --db_path ./employee_db.json \ # File JSON output
    --checkpoint_path ./checkpoints/adaface_ir101_ms1mv3.pth

### 3. Xử lý Video

python main.py --video_path ./input_video.mp4 \  # Đường dẫn video đầu vào
    --db_path ./employee_db.json \               # File DB đã xây dựng
    --output_video_path ./output_video.mp4 \     # Video output với annotation
    --checkpoint_path ./checkpoints/adaface_ir101_ms1mv3.pth \
    --conf_threshold 0.6 \                       # Ngưỡng YOLO confidence
    --cos_threshold 0.6 \                        # Ngưỡng cosine cho matching DB
    --reid_threshold 0.8 \                       # Ngưỡng cosine cho re-ID
    --skip_interval 5                            # Khoảng cách frame thử embedding

