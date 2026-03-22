ĐẶT FILE TRỌNG SỐ WLASL VÀO ĐÂY
==================================

Bạn CẦN đặt file trọng số WLASL (.pth.tar) vào thư mục này để app hoạt động.

BƯỚC 1 — Tải file về:
  https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48

BƯỚC 2 — Giải nén
  Bên trong ZIP có cấu trúc:  archived\archived\asl100\
  Bên trong thư mục asl100 bạn sẽ thấy một số file.

  ★ FILE CẦN DÙNG: file có đuôi .pth.tar  (ví dụ: nslt_100.pth.tar)
  ✗ Các file còn lại (như .json, optimizer state, v.v.) KHÔNG cần copy.

BƯỚC 3 — Copy file .pth.tar vào đây (chọn 1 cách):

  Cách A (copy cả thư mục con, Windows PowerShell):
    xcopy /E /I "C:\Users\...\Downloads\archived\archived\asl100" "models\wlasl\asl100"

  Cách B (copy thẳng file .pth.tar, Windows PowerShell):
    copy "C:\Users\...\Downloads\archived\archived\asl100\nslt_100.pth.tar" "models\wlasl\"

  Cách C (Linux / macOS):
    cp ~/Downloads/archived/archived/asl100/nslt_100.pth.tar models/wlasl/

KẾT QUẢ mong đợi sau khi chạy server:
  [INFO] WLASL I3D model loaded: nslt_100.pth.tar  (100 classes)

Lưu ý:
  - App chỉ đọc file .pth.tar hoặc .pth — các file khác bị bỏ qua hoàn toàn
  - App tự tìm file .pth.tar trong toàn bộ thư mục con bên dưới models/wlasl/
  - Không cần làm phẳng cấu trúc thư mục của archive
  - File .gitkeep trong thư mục này là placeholder, không xóa
