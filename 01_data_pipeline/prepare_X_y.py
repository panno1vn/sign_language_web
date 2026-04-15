import json
import numpy as np
import os

# Cấu hình đường dẫn và tham số tiền xử lý.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
JSON_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "MASTER_DATASET.json")
NPZ_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "MASTER_DATASET.npz")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Số frame chuẩn cho mỗi mẫu sau tiền xử lý.
MAX_FRAMES = 30  

def process_sequences():
    print(f"1. Đang tải MASTER DATASET...")
    if not os.path.exists(JSON_PATH) or not os.path.exists(NPZ_PATH):
        print("❌ LỖI: Không tìm thấy file MASTER_DATASET.")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        master_json = json.load(f)
        
    archive = np.load(NPZ_PATH)
    
    words = [item['gloss'] for item in master_json]
    label_map = {label: num for num, label in enumerate(words)}
    
    X, y = [], []
    missing_count = 0
    inconsistent_shape_count = 0

    print(f"2. Bắt đầu ép chuỗi về {MAX_FRAMES} frames (Padding & Truncating)...")
    
    for item in master_json:
        gloss = item['gloss']
        label_id = label_map[gloss]
        
        for instance in item['instances']:
            vid_id = instance['video_id']
            
            if vid_id in archive:
                data = archive[vid_id] 
                
                # Chuẩn hóa số chiều về dạng (frames, features).
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
                elif data.ndim == 1:
                    data = np.expand_dims(data, axis=0)  # Xử lý trường hợp dữ liệu một chiều.
                    
                frames_count = data.shape[0]
                
                if frames_count == 0:
                    continue
                    
                # Bỏ qua mẫu không khớp số đặc trưng yêu cầu của mô hình.
                if data.shape[1] != 1659:
                    inconsistent_shape_count += 1
                    continue
                
                # Chuẩn hóa độ dài chuỗi theo MAX_FRAMES.
                if frames_count < MAX_FRAMES:
                    # Padding với giá trị 0 khi số frame còn thiếu.
                    pad_amount = MAX_FRAMES - frames_count
                    padded_data = np.pad(data, ((0, pad_amount), (0, 0)), mode='constant')
                    X.append(padded_data)
                
                elif frames_count > MAX_FRAMES:
                    # Cắt giữa chuỗi khi số frame vượt quá MAX_FRAMES.
                    start_idx = (frames_count - MAX_FRAMES) // 2
                    truncated_data = data[start_idx : start_idx + MAX_FRAMES]
                    X.append(truncated_data)
                
                else:
                    # Giữ nguyên khi số frame đã đúng chuẩn.
                    X.append(data)
                    
                y.append(label_id)
            else:
                missing_count += 1

    archive.close()
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n3. Hoàn tất!")
    print(f"   -> Đã bỏ qua {missing_count} video trống.")
    if inconsistent_shape_count > 0:
        print(f"   -> ⚠️ Đã loại bỏ {inconsistent_shape_count} video bị sai cấu trúc (không đủ 1662 điểm tọa độ).")
        
    print(f"   -> Kích thước X (Dữ liệu huấn luyện): {X.shape}") 
    print(f"   -> Kích thước y (Nhãn - Labels): {y.shape}")
    
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
        
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    
    print("\n✅ Đã xuất X.npy, y.npy và label_map.json. SẴN SÀNG ĐỂ BUILD MÔ HÌNH!")

if __name__ == "__main__":
    process_sequences()