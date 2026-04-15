import json
import numpy as np
import os
import sys

# Cấu hình đường dẫn dựa trên thư mục chứa script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RAW_JSON = os.path.join(PROJECT_ROOT, "data", "raw", "WLASL_v0.3.json")

RAW_NPZ_FILES = [
    os.path.join(PROJECT_ROOT, "data", "raw", "landmarks_v1.npz"),
    os.path.join(PROJECT_ROOT, "data", "raw", "landmarks_v2.npz"),
    os.path.join(PROJECT_ROOT, "data", "raw", "landmarks_v3.npz")
]

# Thư mục đầu ra cho dữ liệu đã lọc.
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(PROCESSED_DIR, 'WLASL_filtered_15plus.json')
OUTPUT_NPZ = os.path.join(PROCESSED_DIR, 'WLASL_filtered_15plus.npz')

MIN_VIDEOS = 15

def run_filter():
    print(f"1. Đang đọc file gốc tại: {RAW_JSON} ...")
    
    # Kiểm tra sự tồn tại của file JSON đầu vào.
    if not os.path.exists(RAW_JSON):
        print(f"❌ LỖI: Không tìm thấy file. Đường dẫn hiện tại đang tìm là: {RAW_JSON}")
        return

    # Xử lý lỗi quyền truy cập khi đọc file.
    try:
        with open(RAW_JSON, 'r', encoding='utf-8') as f:
            wlasl_data = json.load(f)
    except PermissionError:
        print("\n❌ LỖI: WINDOWS TỪ CHỐI QUYỀN TRUY CẬP FILE!")
        print("👉 CÁCH KHẮC PHỤC:")
        print("  1. Hãy nhìn lên các tab của VS Code, nếu có tab WLASL_v0.3.json đang mở -> HÃY TẮT NÓ ĐI.")
        print("  2. Tắt các phần mềm khác (Notepad, Word...) có thể đang mở file này.")
        print("  3. Tắt Terminal hiện tại, mở lại bằng quyền 'Run as Administrator'.")
        return
    except Exception as e:
        print(f"\n❌ LỖI KHÔNG XÁC ĐỊNH KHI ĐỌC FILE: {e}")
        return

    valid_video_ids = set()
    filtered_json_data = []

    # Lọc các gloss có số lượng video không nhỏ hơn ngưỡng.
    for item in wlasl_data:
        if len(item['instances']) >= MIN_VIDEOS:
            filtered_json_data.append(item)
            for instance in item['instances']:
                valid_video_ids.add(instance['video_id'])

    print(f"   -> Đã giữ lại: {len(filtered_json_data)} từ vựng.")
    print(f"   -> Tổng số video cần trích xuất: {len(valid_video_ids)} video.\n")

    # Lưu cấu trúc JSON sau khi lọc.
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(filtered_json_data, f, indent=4)
    print(f"2. Đã lưu file JSON cấu trúc tại: {OUTPUT_JSON}\n")

    print("3. Bắt đầu quét 3 file .npz nặng (Vui lòng đợi 1-2 phút, tốn khá nhiều RAM)...")
    filtered_numpy_data = {}
    found_count = 0

    for npz_path in RAW_NPZ_FILES:
        if not os.path.exists(npz_path):
            print(f"   [CẢNH BÁO] Không tìm thấy: {npz_path}")
            continue
            
        print(f"   - Đang mở: {os.path.basename(npz_path)}...")
        
        try:
            archive = np.load(npz_path)
            # Thu thập ma trận tọa độ cho các video hợp lệ.
            for video_id in archive.files:
                clean_id = video_id.replace('.npy', '') 
                
                if clean_id in valid_video_ids:
                    filtered_numpy_data[clean_id] = archive[video_id]
                    found_count += 1
            archive.close()
        except Exception as e:
            print(f"   ❌ Lỗi khi đọc file {os.path.basename(npz_path)}: {e}")

    print(f"\n4. Đã gom đủ {found_count}/{len(valid_video_ids)} ma trận tọa độ.")
    
    print("5. Đang nén dữ liệu thành 1 file .npz duy nhất...")
    np.savez_compressed(OUTPUT_NPZ, **filtered_numpy_data)
    
    print(f"\n✅ HOÀN TẤT LỌC KAGGLE! File siêu nén đã sẵn sàng tại: {OUTPUT_NPZ}")

if __name__ == "__main__":
    run_filter()