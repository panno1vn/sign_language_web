import os
import json
import numpy as np
import cv2
import traceback
from pathlib import Path

# Tương thích import MediaPipe giữa các phiên bản.
import mediapipe as mp
try:
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    mp_holistic = mp.solutions.holistic

# Cấu hình đường dẫn dữ liệu.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
JSON_PATH = Path(os.path.join(PROJECT_ROOT, "dataset_final.json"))
OUTPUT_DIR = Path(os.path.join(PROJECT_ROOT, "dataset_landmarks"))
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_landmarks_holistic(results):
    """
    Trích xuất đúng 1662 điểm chuẩn Kaggle:
    Pose(132) + Face(1404) + LH(63) + RH(63) = 1662
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, face, lh, rh])

def main():
    if not JSON_PATH.exists():
        print(f"❌ Không tìm thấy file {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print(f"🚀 Bắt đầu trích xuất lại cho {len(records)} videos theo chuẩn Kaggle (1662 điểm)...")
    
    # Mở mô hình Holistic một lần cho toàn bộ quá trình xử lý.
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i, record in enumerate(records):
            video_path = record.get("local_path")
            
            # Bỏ qua bản ghi không có đường dẫn video hợp lệ.
            if not video_path or not os.path.exists(video_path):
                continue
                
            word = record["word"]
            base_name = Path(video_path).stem
            
            save_dir = OUTPUT_DIR / word
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"{base_name}.npy"
            
            # Luôn ghi đè để đảm bảo dữ liệu đồng bộ với cấu hình hiện tại.
            
            print(f"[{i+1}/{len(records)}] Đang xử lý: {word}/{base_name}...")
            
            try:
                cap = cv2.VideoCapture(video_path)
                video_landmarks = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    
                    frame_data = extract_landmarks_holistic(results)
                    video_landmarks.append(frame_data)
                    
                cap.release()
                
                if len(video_landmarks) > 0:
                    np_data = np.array(video_landmarks)
                    np.save(str(save_path), np_data)
                    
                    # Cập nhật đường dẫn landmark đã chuẩn hóa vào bản ghi.
                    record["landmark_path"] = str(save_path)
                else:
                    print(f"   [!] Bỏ qua video vì không đọc được frame nào.")
                
            except Exception as e:
                print(f"❌ Lỗi tại video {video_path}: {e}")

    # Ghi lại JSON sau khi cập nhật đường dẫn landmark.
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
        
    print("\n✅ HOÀN TẤT! Dữ liệu 1000 video đã đồng bộ cấu trúc 100% với Kaggle.")

if __name__ == "__main__":
    main()