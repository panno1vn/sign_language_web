import os
import json
import numpy as np
import cv2
from pathlib import Path

# Tương thích import MediaPipe giữa các phiên bản.
import mediapipe as mp
try:
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    mp_holistic = mp.solutions.holistic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def extract_1659_landmarks(results):
    # Khởi tạo mảng 553 điểm, mỗi điểm gồm 3 tọa độ (x, y, z).
    landmarks = np.zeros((553, 3))
    
    # Pose: 33 điểm, sử dụng các thành phần x/y/z.
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
            
    # Face: 478 điểm, đồng bộ với cấu trúc dữ liệu huấn luyện.
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            landmarks[33 + i] = [lm.x, lm.y, lm.z]
            
    # Left hand: 21 điểm.
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[33 + 478 + i] = [lm.x, lm.y, lm.z]
            
    # Right hand: 21 điểm.
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[33 + 478 + 21 + i] = [lm.x, lm.y, lm.z]
            
    return landmarks  # Đầu ra chuẩn: (553, 3).

def run():
    print("🚀 BẮT ĐẦU ĐỒNG BỘ CẤU TRÚC (1659 điểm) VỚI KAGGLE...")
    
    # Bật refine_face_landmarks để lấy đầy đủ 478 điểm khuôn mặt.
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
        
        # Phần 1: Chuẩn hóa lớp idle.
        idle_dir = os.path.join(PROJECT_ROOT, "data", "custom_videos")
        idle_out = os.path.join(PROJECT_ROOT, "data", "idle_npy")
        os.makedirs(idle_out, exist_ok=True)
        
        print("\n👉 1. Đang ép chuẩn các video IDLE...")
        if os.path.exists(idle_dir):
            idle_videos = [f for f in os.listdir(idle_dir) if f.endswith(('.mp4', '.avi'))]
            for vid in idle_videos:
                cap = cv2.VideoCapture(os.path.join(idle_dir, vid))
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = holistic.process(img)
                    frames.append(extract_1659_landmarks(res))
                cap.release()
                if frames:
                    np.save(os.path.join(idle_out, vid.split('.')[0] + ".npy"), np.array(frames))
            print(f"   ✅ Xong {len(idle_videos)} video Idle.")

        # Phần 2: Chuẩn hóa dữ liệu video tự thu thập.
        scraped_json_path = os.path.join(PROJECT_ROOT, "dataset_final.json")
        scraped_out_dir = os.path.join(PROJECT_ROOT, "dataset_landmarks")
        os.makedirs(scraped_out_dir, exist_ok=True)
        
        print("\n👉 2. Đang ép chuẩn 1000 video TỰ CÀO (Vui lòng đợi 15-30 phút)...")
        if os.path.exists(scraped_json_path):
            with open(scraped_json_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            for i, record in enumerate(records):
                vid_path = record.get('local_path')
                if not vid_path or not os.path.exists(vid_path): continue
                
                word = record['word']
                vid_id = Path(vid_path).stem
                save_dir = os.path.join(scraped_out_dir, word)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{vid_id}.npy")
                
                cap = cv2.VideoCapture(vid_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = holistic.process(img)
                    frames.append(extract_1659_landmarks(res))
                cap.release()
                
                if frames:
                    np.save(save_path, np.array(frames))
                    record['landmark_path'] = str(Path("dataset_landmarks") / word / f"{vid_id}.npy")
                
                if (i+1) % 50 == 0:
                    print(f"   - Đã xử lý {i+1}/{len(records)} video...")
                    
            with open(scraped_json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=4)
                
    print("\n🎉 HOÀN TẤT ĐỒNG BỘ! TẤT CẢ DỮ LIỆU ĐÃ SẴN SÀNG.")

if __name__ == '__main__':
    run()