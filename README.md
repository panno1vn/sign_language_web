# Phần Mềm Dịch Ngôn Ngữ Ký Hiệu (Sign Language Translator)

Web app Django nhận dạng ngôn ngữ ký hiệu Mỹ (ASL) từ video camera hoặc file video tải lên.

## Tính năng

- 📹 **Quay camera trực tiếp** — ghi 3 giây rồi tự động gửi lên server phân tích
- 📁 **Tải video lên** — hỗ trợ `.mp4`, `.webm`, `.avi`
- 🤖 **Hai backend model:**
  - **WLASL I3D** (ưu tiên) — model PyTorch pre-trained từ [dxli94/WLASL](https://github.com/dxli94/WLASL), nhận dạng 100 ký hiệu ASL
  - **Local TF/Keras** (fallback) — model TensorFlow/Keras tự huấn luyện
- 📊 **Top-3 kết quả** với thanh xác suất trực quan

---

## 🚀 Chạy Demo Nhanh (5 bước)

### Bước 1 — Clone repo và cài thư viện

```bash
git clone https://github.com/panno1vn/sign_language_web.git
cd sign_language_web
pip install -r requirements.txt
```

### Bước 2 — Tải trọng số model WLASL

> **Đây là bước bắt buộc** — không có model thì app không dịch được.

**Chọn 1 trong 2 cách:**

#### Cách A: Dùng model WLASL I3D (pre-trained, khuyên dùng)

1. Tải file từ Google Drive:  
   👉 **[Tải WLASL pre-trained weights](https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view)**

2. Giải nén — bên trong bạn sẽ thấy cấu trúc **`archived\archived\`** (hai lớp thư mục cùng tên, đây là cấu trúc thực của file zip):

   ```
   archived\
   └── archived\
       ├── asl100\      ← chứa checkpoint 100 từ (khuyên dùng, khớp với wlasl_class_list.txt)
       ├── asl300\      ← checkpoint 300 từ
       ├── 1000\
       └── 2000\
   ```

3. Sao chép **cả thư mục `asl100`** vào `models/wlasl/` trong repo — hoặc chỉ copy file `.pth.tar` bên trong nó:

   **Windows (PowerShell):**
   ```powershell
   # Copy cả thư mục con (cách đơn giản nhất)
   xcopy /E /I "C:\Users\Admin\Downloads\archived\archived\asl100" "models\wlasl\asl100"

   # Hoặc chỉ copy 1 file .pth.tar vào thẳng models\wlasl\
   copy "C:\Users\Admin\Downloads\archived\archived\asl100\*.pth.tar" "models\wlasl\"
   ```

   **Linux / macOS:**
   ```bash
   # Copy cả thư mục con
   cp -r ~/Downloads/archived/archived/asl100 models/wlasl/

   # Hoặc chỉ copy 1 file .pth.tar
   cp ~/Downloads/archived/archived/asl100/*.pth.tar models/wlasl/
   ```

   > ℹ️ App tự động tìm kiếm `.pth.tar` trong **toàn bộ cây thư mục** bên dưới `models/wlasl/`, kể cả thư mục con — bạn không cần phải làm phẳng cấu trúc thư mục.

**Kết quả mong đợi khi chạy server:**  
`[INFO] WLASL I3D model loaded: asl100/nslt_100.pth.tar  (100 classes)`

#### Cách B: Dùng model TF/Keras tự huấn luyện

Nếu bạn đã huấn luyện xong model trên Kaggle và có file `.h5`:

```bash
cp đường/dẫn/tới/model.h5 models/12-14.h5
```

**Kết quả mong đợi khi chạy server:**  
`[INFO] Local TF/Keras model loaded.`

### Bước 3 — Migrate database

```bash
python manage.py migrate
```

### Bước 4 — Chạy server

```bash
python manage.py runserver
```

### Bước 5 — Mở trình duyệt

Truy cập: **http://127.0.0.1:8000**

---

## 🐳 Chạy bằng Docker (tùy chọn)

```bash
# Đặt file model vào models/wlasl/ hoặc models/12-14.h5 trước (xem Bước 2)
docker compose up --build
```

Truy cập: **http://localhost:8000**

---

## ❓ Kiểm tra model đã nạp chưa

Khi chạy `python manage.py runserver`, terminal sẽ in ra một trong các dòng sau:

| Thông báo | Ý nghĩa |
|-----------|---------|
| `[INFO] WLASL I3D model loaded: asl100/nslt_100.pth.tar  (100 classes)` | ✅ WLASL model sẵn sàng |
| `[INFO] Local TF/Keras model loaded.` | ✅ Model Keras sẵn sàng |
| `[INFO] No WLASL weights found in models/wlasl/ — falling back to local model.` | ⚠️ Chưa có WLASL weights |
| `[WARNING] Could not load local TF/Keras model: ...` | ⚠️ Chưa có file `.h5` |

Nếu thấy cả hai cảnh báo, app vẫn chạy nhưng sẽ báo lỗi khi bạn gửi video. Hãy thực hiện Bước 2 trước.

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
│   ├── wlasl/            # ← Đặt file nslt_100.pth.tar vào đây (Cách A)
│   ├── wlasl_class_list.txt  # 100 nhãn ASL của WLASL100
│   ├── 12-14.h5          # ← Đặt model Keras vào đây (Cách B, không có trong repo)
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
