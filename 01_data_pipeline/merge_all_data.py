import json
import numpy as np
import os

# Cấu hình đường dẫn dữ liệu đầu vào và đầu ra.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Nguồn dữ liệu Kaggle đã lọc.
KAGGLE_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "WLASL_filtered_15plus.json")
KAGGLE_NPZ = os.path.join(PROJECT_ROOT, "data", "processed", "WLASL_filtered_15plus.npz")

# Nguồn dữ liệu tự thu thập.
SCRAPED_JSON = os.path.join(PROJECT_ROOT, "dataset_final.json")
SCRAPED_DIR = os.path.join(PROJECT_ROOT, "dataset_landmarks") 

# Nguồn dữ liệu lớp idle.
IDLE_DIR = os.path.join(PROJECT_ROOT, "data", "idle_npy")

# Đường dẫn đầu ra cho bộ dữ liệu hợp nhất.
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "MASTER_DATASET.json")
OUTPUT_NPZ = os.path.join(PROJECT_ROOT, "data", "processed", "MASTER_DATASET.npz")

def merge_datasets():
    master_dict = {}  # Chứa ma trận tọa độ: {video_id: array}
    master_json = {}  # Chứa cấu trúc JSON: {gloss: [instances]}

    # Bước 1: Nạp dữ liệu Kaggle đã lọc.
    print("1. Đang nạp dữ liệu Kaggle đã lọc...")
    if os.path.exists(KAGGLE_JSON) and os.path.exists(KAGGLE_NPZ):
        with open(KAGGLE_JSON, 'r', encoding='utf-8') as f:
            kaggle_data = json.load(f)
        
        # Nạp file nén .npz chứa landmark.
        archive = np.load(KAGGLE_NPZ)
        for item in kaggle_data:
            gloss = item['gloss']
            master_json[gloss] = item['instances']
            for instance in item['instances']:
                vid_id = instance['video_id']
                if vid_id in archive:
                    master_dict[vid_id] = archive[vid_id]
        archive.close()
        print(f"   -> Đã nạp {len(master_dict)} video từ Kaggle.")
    else:
        print("   [!] Không thấy dữ liệu Kaggle, bỏ qua.")

    # Bước 2: Nạp dữ liệu tự thu thập.
    print("\n2. Đang nạp dữ liệu từ 1000 video tự cào...")
    if os.path.exists(SCRAPED_JSON):
        with open(SCRAPED_JSON, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        
        count_scraped = 0
        for item in scraped_data:
            gloss = item.get('word')
            vid_id = item.get('id')
            # Bỏ qua bản ghi thiếu landmark_path để tránh lỗi runtime.
            landmark_path = item.get('landmark_path')
            
            if not landmark_path:
                continue
            
            # Chuẩn hóa đường dẫn cho cả Windows và Linux.
            full_path = os.path.join(PROJECT_ROOT, landmark_path.replace('\\', os.sep).replace('/', os.sep))
            
            if os.path.exists(full_path):
                master_dict[vid_id] = np.load(full_path)
                if gloss not in master_json:
                    master_json[gloss] = []
                master_json[gloss].append({
                    "video_id": vid_id, 
                    "split": item.get('split', 'train')
                })
                count_scraped += 1
        print(f"   -> Đã nạp thành công {count_scraped} video tự cào.")
    else:
        print("   [!] Không tìm thấy dataset_final.json.")

    # Bước 3: Nạp dữ liệu lớp idle.
    print("\n3. Đang nạp dữ liệu lớp IDLE...")
    count_idle = 0
    if os.path.exists(IDLE_DIR):
        for file in os.listdir(IDLE_DIR):
            if file.endswith('.npy'):
                vid_id = file.replace('.npy', '')
                full_path = os.path.join(IDLE_DIR, file)
                
                master_dict[vid_id] = np.load(full_path)
                
                if "idle" not in master_json:
                    master_json["idle"] = []
                master_json["idle"].append({"video_id": vid_id, "split": "train"})
                count_idle += 1
        print(f"   -> Đã nạp {count_idle} video lớp Idle.")

    # Bước 4: Lưu kết quả hợp nhất.
    total_videos = len(master_dict)
    total_classes = len(master_json)
    
    print(f"\n4. Đang nén và lưu MASTER_DATASET... (Tổng: {total_classes} lớp, {total_videos} video)")
    
    # Chuyển đổi master_json sang danh sách để lưu JSON.
    final_json_output = [{"gloss": k, "instances": v} for k, v in master_json.items()]
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, indent=4, ensure_ascii=False)
        
    # Lưu file .npz nén để giảm dung lượng lưu trữ.
    np.savez_compressed(OUTPUT_NPZ, **master_dict)
    
    print(f"\n✅ THÀNH CÔNG! Dữ liệu đã được gộp tại:")
    print(f"   - {OUTPUT_JSON}")
    print(f"   - {OUTPUT_NPZ}")

if __name__ == "__main__":
    merge_datasets()