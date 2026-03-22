# Phần Mềm Dịch Ngôn Ngữ Ký Hiệu (Sign Language Translator)

Web app Django nhận dạng ngôn ngữ ký hiệu Mỹ (ASL) từ video camera hoặc file video tải lên.

## Tính năng

- 📹 **Quay camera trực tiếp** — ghi 3 giây rồi tự động gửi lên server phân tích
- 📁 **Tải video lên** — hỗ trợ `.mp4`, `.webm`, `.avi`
- 🤖 **Hai backend model:**
  - **WLASL I3D** (ưu tiên) — model PyTorch pre-trained từ [dxli94/WLASL](https://github.com/dxli94/WLASL), nhận dạng 100 ký hiệu ASL
  - **Local TF/Keras** (fallback) — model TensorFlow/Keras huấn luyện sẵn
- 📊 **Top-3 kết quả** với thanh xác suất trực quan

---

## Cài đặt & Chạy

### Yêu cầu
- Python 3.10+
- Các thư viện trong `requirements.txt`

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Tải trọng số model WLASL (bắt buộc để dùng model tốt nhất)

1. Tải file từ Google Drive:  
   **[WLASL pre-trained weights (.pth.tar)](https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view)**

2. Giải nén — trong thư mục `I3D/archived/` sẽ có các file như `nslt_100.pth.tar`

3. Tạo thư mục và sao chép file vào đúng vị trí:

```bash
mkdir -p models/wlasl
cp nslt_100.pth.tar models/wlasl/
```

> Nếu bạn không có file WLASL, server vẫn chạy được với model TF/Keras (`models/12-14.h5`) nếu file đó có mặt.

### 3. Migrate database

```bash
python manage.py migrate
```

### 4. Chạy server

```bash
python manage.py runserver
```

Truy cập: **http://127.0.0.1:8000**

---

## Chạy bằng Docker

```bash
# Thêm file trọng số vào models/wlasl/ trước (xem bước 2 ở trên)
docker compose up --build
```

Truy cập: **http://localhost:8000**

---

## Cấu trúc thư mục

```
sign_language_web/
├── config/               # Django settings, urls, wsgi
├── translator/
│   ├── templates/
│   │   └── index.html    # Giao diện chính
│   ├── views.py          # Logic xử lý video, gọi model
│   ├── wlasl_model.py    # Kiến trúc I3D + inference WLASL
│   └── ...
├── models/
│   ├── wlasl/            # ← Đặt file .pth.tar vào đây (tải thủ công)
│   ├── wlasl_class_list.txt  # 100 nhãn ASL của WLASL100
│   ├── 12-14.h5          # Model TF/Keras fallback (nếu có)
│   └── labels.npz        # Nhãn cho model TF/Keras
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Biến môi trường (Production)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `SECRET_KEY` | *(dev key)* | Django secret key — **bắt buộc thay đổi khi deploy** |
| `DEBUG` | `1` | Đặt `0` để tắt debug mode khi deploy |

---

## Nguồn model

- **WLASL I3D**: [dxli94/WLASL](https://github.com/dxli94/WLASL) — Li et al., *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*, WACV 2020
