ĐẶT FILE TRỌNG SỐ WLASL VÀO ĐÂY
==================================

Bạn CẦN đặt file trọng số WLASL vào thư mục này để app hoạt động.
App hỗ trợ các định dạng: .pt  /  .pth  /  .pth.tar

BƯỚC 1 — Tải file về:
  https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48

BƯỚC 2 — Giải nén
  Bên trong ZIP có cấu trúc:  archived\archived\asl100\
  Bên trong thư mục asl100 bạn sẽ thấy một số file.

  ★ FILE CẦN DÙNG: file có đuôi .pt  hoặc  .pth  hoặc  .pth.tar
                   (ví dụ: nslt_100.pt  hoặc  nslt_100.pth.tar)
  ✗ Các file còn lại (như .json, optimizer state, v.v.) KHÔNG cần copy.

BƯỚC 3 — Copy file vào đây (chọn 1 cách):

  Cách A (copy cả thư mục con, Windows PowerShell):
    xcopy /E /I "C:\Users\...\Downloads\archived\archived\asl100" "models\wlasl\asl100"

  Cách B (copy thẳng file trọng số, Windows PowerShell):
    copy "C:\Users\...\Downloads\archived\archived\asl100\nslt_100.pt" "models\wlasl\"

  Cách C (Linux / macOS):
    cp ~/Downloads/archived/archived/asl100/nslt_100.pt models/wlasl/

KẾT QUẢ mong đợi sau khi chạy server:
  [INFO] WLASL I3D model loaded: nslt_100.pt  (100 classes)

Lưu ý:
  - App đọc file .pt, .pth, hoặc .pth.tar — các file khác bị bỏ qua hoàn toàn
  - App tự tìm file trong toàn bộ thư mục con bên dưới models/wlasl/
  - Không cần làm phẳng cấu trúc thư mục của archive
  - File .gitkeep trong thư mục này là placeholder, không xóa
