import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse

# Đặt đường dẫn tuyệt đối để không bao giờ lỗi
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', '12-14.h5')
LABELS_DICT_PATH = os.path.join(BASE_DIR, 'models', 'labels.npz')

# Tải Model
model = tf.keras.models.load_model(MODEL_PATH)

# Tải Từ điển FastText Vector
try:
    LABELS_DICT = np.load(LABELS_DICT_PATH, allow_pickle=True)
except FileNotFoundError:
    print("!!! LỖI: Chưa tìm thấy file labels.npz trong thư mục models !!!")
    LABELS_DICT = {}

# --- CẤU HÌNH MEDIAPIPE (Giữ nguyên từ Kaggle) ---
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
                 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150,
                 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246,
                 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310,
                 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377,
                 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
                 466, 468, 473]

HAND_NUM = len(filtered_hand)
POSE_NUM = len(filtered_pose)
FACE_NUM = len(filtered_face)
NUM_LANDMARKS = HAND_NUM * 2 + POSE_NUM + FACE_NUM

# Giữ nguyên các list này ở trên
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
                 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150,
                 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246,
                 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310,
                 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377,
                 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
                 466, 468, 473]

HAND_NUM = len(filtered_hand)
POSE_NUM = len(filtered_pose)

# TẠO RA BỘ CHỈ MỤC BỊ LỖI Y HỆT TRÊN KAGGLE ĐỂ ÉP MODEL HIỂU
BUGGY_INDICES = (
        [x for x in filtered_hand] +
        [x + HAND_NUM for x in filtered_hand] +
        [x + HAND_NUM * 2 for x in filtered_pose] +
        [x + HAND_NUM * 2 + POSE_NUM for x in filtered_face]  # Điểm mù gây ra lỗi
)


def get_frame_landmarks(frame, hands_mp, pose_mp, face_mesh_mp):
    # 1. Khởi tạo mảng FULL 543 điểm của MediaPipe (giống cấu trúc V3)
    full_landmarks = np.zeros((42 + 33 + 468, 3))

    def get_hands(f):
        results = hands_mp.process(f)
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Lưu ý: Tay trái/phải trên web có thể bị ngược so với Dataset
                if results.multi_handedness[i].classification[0].index == 0:
                    full_landmarks[:21, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                else:
                    full_landmarks[21:42, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    def get_pose(f):
        results = pose_mp.process(f)
        if results.pose_landmarks:
            # Ghi vào đúng vị trí của Pose trong mảng 543
            full_landmarks[42:75, :] = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])

    def get_face(f):
        results = face_mesh_mp.process(f)
        if results.multi_face_landmarks:
            # Ghi vào đúng vị trí của Face trong mảng 543
            full_landmarks[75:, :] = np.array([(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark])

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(get_hands, frame)
        executor.submit(get_pose, frame)
        executor.submit(get_face, frame)

    # 2. CẮT MẢNG THEO ĐÚNG CÔNG THỨC LỖI TRÊN KAGGLE (Kích thước sẽ ra đúng 180)
    return full_landmarks[BUGGY_INDICES, :]


# ... (Các thư viện và load model giữ nguyên)

# TẠO LẠI ĐÚNG BỘ CHỈ MỤC LỖI TRÊN KAGGLE
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
                 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150,
                 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246,
                 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310,
                 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377,
                 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
                 466, 468, 473]

BUGGY_INDICES = (
        [x for x in range(21)] +
        [x + 21 for x in range(21)] +
        [x + 42 for x in [11, 12, 13, 14, 15, 16]] +
        [x + 48 for x in filtered_face]
)


def extract_features_from_video(video_path, target_length=200, pad_value=-100):
    # Dùng Holistic nguyên khối y như thư viện V3
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(video_path)
    video_landmarks = []

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)

            # 1. Tái tạo lại mảng 543 điểm chuẩn của Holistic
            full_landmarks = np.zeros((543, 3))

            if results.pose_landmarks:
                full_landmarks[0:33, :] = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
            if results.face_landmarks:
                full_landmarks[33:501, :] = np.array([(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark])
            if results.left_hand_landmarks:
                full_landmarks[501:522, :] = np.array(
                    [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark])
            if results.right_hand_landmarks:
                full_landmarks[522:543, :] = np.array(
                    [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark])

            # 2. Ép dùng bộ lọc lỗi của Kaggle để tạo ra "đĩa súp" quen thuộc cho Model
            frame_landmarks = full_landmarks[BUGGY_INDICES, :]
            video_landmarks.append(frame_landmarks)

    cap.release()

    if len(video_landmarks) == 0:
        return np.full((1, target_length, 180, 3), pad_value, dtype=float)

    video_landmarks = np.array(video_landmarks)

    if len(video_landmarks) > target_length:
        processed_data = video_landmarks[:target_length]
    else:
        pad_length = target_length - len(video_landmarks)
        processed_data = np.pad(
            video_landmarks,
            ((0, pad_length), (0, 0), (0, 0)),
            mode='constant',
            constant_values=pad_value
        )

    return np.expand_dims(processed_data, axis=0)


# ... (Phần code cosine_similarity và def index(request) giữ nguyên y hệt)

# Y hệt code tính điểm trên Kaggle
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def index(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video = request.FILES['video_file']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        uploaded_file_url = fs.path(filename)

        try:
            features = extract_features_from_video(uploaded_file_url)

            # 1. Model nhả ra vector 300 chiều
            prediction_vector = model.predict(features)[0]

            best_label = "Không xác định"
            max_similarity = -1.0  # Cosine Similarity chạy từ -1 đến 1

            # 2. Duyệt qua file npz để tìm từ có Vector giống với Model nhất
            for label in LABELS_DICT.keys():
                word_vector = LABELS_DICT[label]
                sim_score = cosine_similarity(prediction_vector, word_vector)

                if sim_score > max_similarity:
                    max_similarity = sim_score
                    best_label = label

            return JsonResponse({'status': 'success', 'result': f"{best_label}"})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
        finally:
            if os.path.exists(uploaded_file_url):
                os.remove(uploaded_file_url)

    # Đã sửa lại đường dẫn cho chuẩn với ảnh của bạn
    return render(request, 'index.html')