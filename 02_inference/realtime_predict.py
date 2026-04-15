import cv2
import numpy as np
import os
import json
from tensorflow.keras.models import load_model

# Tương thích import MediaPipe giữa các phiên bản.
import mediapipe as mp
try:
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    mp_holistic = mp.solutions.holistic

# Cấu hình đường dẫn model và nhãn.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_action_model.h5")  # Cập nhật nếu tên model thay đổi.
LABEL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "label_map.json")

def extract_1659_landmarks(results):
    """Trích xuất 1659 điểm gốc, sau đó GỌT NGAY LẬP TỨC xuống 225 điểm"""
    landmarks = np.zeros((553, 3))
    
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            landmarks[33 + i] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[33 + 478 + i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[33 + 478 + 21 + i] = [lm.x, lm.y, lm.z]
            
    # Chuyển về vector 1659 chiều.
    flat_landmarks = landmarks.flatten()
    
    # Giữ Pose (99 chiều) và Hands (126 chiều), tổng 225 chiều.
    pose_data = flat_landmarks[0:99]
    hands_data = flat_landmarks[1533:1659]
    
    return np.concatenate([pose_data, hands_data])

def main():
    print("1. Đang nạp mô hình AI và Từ điển...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        print("❌ LỖI: Không tìm thấy model hoặc label_map.json!")
        return

    model = load_model(MODEL_PATH)
    
    with open(LABEL_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
        
    # Đảo chiều ánh xạ từ id sang nhãn.
    actions = {v: k for k, v in label_map.items()}
    
    print("2. Mở Webcam... (Bấm 'Q' để thoát)")
    
    sequence = []
    current_action = "Waiting..."
    confidence = 0.0
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Lật khung hình để hiển thị theo dạng gương.
            frame = cv2.flip(frame, 1)
            image, results = frame.copy(), None
            
            # Trích xuất landmark bằng MediaPipe.
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = holistic.process(img_rgb)
            
            # Vẽ pose và hand landmarks để quan sát kết quả tracking.
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Trích xuất vector đặc trưng 225 chiều.
            keypoints = extract_1659_landmarks(results)
            sequence.append(keypoints)
            
            # Giữ 30 frame gần nhất cho cửa sổ dự đoán.
            sequence = sequence[-30:]
            
            # Dự đoán hành động.
            if len(sequence) == 30:
                # Chuẩn hóa input theo shape model: (1, 30, 225).
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                # Chọn lớp có xác suất cao nhất.
                best_match_id = np.argmax(res)
                confidence = res[best_match_id]
                
                # Chỉ hiển thị kết quả khi độ tin cậy lớn hơn ngưỡng.
                if confidence > 0.50:
                    current_action = actions[best_match_id]
                else:
                    # Gán idle khi độ tin cậy chưa đủ.
                    current_action = "idle" 

            # Hiển thị kết quả dự đoán trên khung hình.
            cv2.rectangle(image, (0, 0), (640, 50), (245, 117, 16), -1)
            cv2.putText(image, f"Action: {current_action}", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Conf: {confidence*100:.1f}%", (450, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('MuteMotion AI', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()