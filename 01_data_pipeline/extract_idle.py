import cv2
import numpy as np
import os

# Import tường minh để giảm khác biệt giữa các phiên bản MediaPipe.
import mediapipe as mp
try:
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    # Fallback cho môi trường không hỗ trợ đường dẫn import ưu tiên.
    mp_holistic = mp.solutions.holistic

# Cấu hình đường dẫn đầu vào và đầu ra.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "custom_videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "idle_npy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_landmarks(results):
    """
    Trích xuất và chuẩn hóa tọa độ y hệt cấu trúc của MuteMotion Dataset.
    Tổng cộng: 132 (Pose) + 1404 (Face) + 63 (LH) + 63 (RH) = 1662 điểm/frame
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, face, lh, rh])

def process_idle_videos():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.startswith('idle_') and f.endswith(('.mp4', '.avi'))]
    
    if not video_files:
        print(f"❌ Không tìm thấy video idle nào trong {VIDEO_DIR}.")
        return
        
    print(f"🚀 Tìm thấy {len(video_files)} video. Bắt đầu trích xuất MediaPipe...")
    
    # Khởi tạo Holistic một lần cho toàn bộ vòng lặp xử lý video.
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for idx, video_name in enumerate(video_files):
            video_path = os.path.join(VIDEO_DIR, video_name)
            video_id = video_name.split('.')[0] 
            
            cap = cv2.VideoCapture(video_path)
            video_landmarks = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                frame_data = extract_landmarks(results)
                video_landmarks.append(frame_data)
                
            cap.release()
            
            npy_path = os.path.join(OUTPUT_DIR, f"{video_id}.npy")
            np_data = np.array(video_landmarks)
            np.save(npy_path, np_data)
            
            print(f"[{idx+1}/{len(video_files)}] Đã xuất {video_id}.npy -> Shape: {np_data.shape}")
            
    print("\n✅ HOÀN TẤT! Tất cả các file tọa độ đã sẵn sàng.")
    print("👉 Hãy chạy 'python merge_all_data.py' để HỢP NHẤT DỮ LIỆU.")

if __name__ == "__main__":
    process_idle_videos()