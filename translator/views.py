import json
import os

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse

# --- ĐƯỜNG DẪN CÁC FILE MODEL ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Ưu tiên model mới (best_model.keras / best_model.h5 + label_map.json),
# nếu chưa có thì dùng model cũ (12-14.h5 + labels.npz) làm fallback.
_MODEL_KERAS = os.path.join(BASE_DIR, 'models', 'best_model.keras')
_MODEL_H5_NEW = os.path.join(BASE_DIR, 'models', 'best_model.h5')
_MODEL_H5_OLD = os.path.join(BASE_DIR, 'models', '12-14.h5')
_LABEL_MAP_JSON = os.path.join(BASE_DIR, 'models', 'label_map.json')
_LABELS_NPZ = os.path.join(BASE_DIR, 'models', 'labels.npz')

# --- CHỌN VÀ TẢI MODEL ---
if os.path.exists(_MODEL_KERAS):
    MODEL_PATH = _MODEL_KERAS
elif os.path.exists(_MODEL_H5_NEW):
    MODEL_PATH = _MODEL_H5_NEW
else:
    MODEL_PATH = _MODEL_H5_OLD

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Đã tải model: {MODEL_PATH}")

# --- CHỌN VÀ TẢI NHÃN ---
# Chế độ mới: label_map.json (index → từ), dự đoán bằng argmax.
# Chế độ cũ: labels.npz (FastText vectors), dự đoán bằng cosine similarity.
USE_LABEL_MAP = os.path.exists(_LABEL_MAP_JSON)

if USE_LABEL_MAP:
    with open(_LABEL_MAP_JSON, encoding='utf-8') as f:
        label_to_index = json.load(f)  # {word: index} as saved by train.py
    # Invert to {index: word} for argmax lookup
    LABEL_INDEX_MAP = {int(v): k for k, v in label_to_index.items()}
    LABELS_DICT = {}
    print(f"✅ Đã tải label_map.json: {len(LABEL_INDEX_MAP)} từ vựng")
else:
    LABEL_INDEX_MAP = {}
    try:
        LABELS_DICT = np.load(_LABELS_NPZ, allow_pickle=True)
        print("✅ Đã tải labels.npz (chế độ cosine similarity)")
    except FileNotFoundError:
        print("!!! LỖI: Không tìm thấy labels.npz trong thư mục models !!!")
        LABELS_DICT = {}

# --- CẤU HÌNH MEDIAPIPE ---
# Số frame cố định (phải khớp với F_AVG khi huấn luyện)
F_AVG = 48

# Số tọa độ mỗi landmark (x, y, z)
_COORDS = 3
# Số landmark: pose=33, left hand=21, right hand=21
_POSE_N = 33
_HAND_N = 21
# Kích thước đặc trưng mỗi frame: (33+21+21)*3 = 225
FEATURE_DIM = (_POSE_N + _HAND_N * 2) * _COORDS


def _extract_holistic_frame(results):
    """
    Trích xuất đặc trưng từ một frame theo định dạng khớp với script huấn luyện:
    np.concatenate((pose, lh, rh), axis=0).flatten() → vector FEATURE_DIM chiều.
    """
    pose = np.zeros((_POSE_N, _COORDS), dtype=np.float32)
    lh = np.zeros((_HAND_N, _COORDS), dtype=np.float32)
    rh = np.zeros((_HAND_N, _COORDS), dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array(
            [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )
    if results.left_hand_landmarks:
        lh = np.array(
            [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        )
    if results.right_hand_landmarks:
        rh = np.array(
            [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        )

    return np.concatenate((pose, lh, rh), axis=0).flatten()


def extract_features_from_video(video_path, target_length=F_AVG):
    """
    Đọc video, trích xuất đặc trưng từng frame qua MediaPipe Holistic,
    rồi pad/trim về target_length frame.
    Trả về mảng shape (1, target_length, FEATURE_DIM).
    """
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    frames_feat = []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            frames_feat.append(_extract_holistic_frame(results))

    cap.release()

    if len(frames_feat) == 0:
        return np.zeros((1, target_length, FEATURE_DIM), dtype=np.float32)

    frames_feat = np.array(frames_feat, dtype=np.float32)  # (N, FEATURE_DIM)

    if len(frames_feat) >= target_length:
        frames_feat = frames_feat[:target_length]
    else:
        pad = np.tile(frames_feat[[-1]], (target_length - len(frames_feat), 1))
        frames_feat = np.concatenate((frames_feat, pad), axis=0)

    return frames_feat[np.newaxis, ...]  # (1, target_length, FEATURE_DIM)


def _cosine_similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def index(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video = request.FILES['video_file']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        uploaded_file_url = fs.path(filename)

        try:
            features = extract_features_from_video(uploaded_file_url)
            prediction = model.predict(features)[0]

            if USE_LABEL_MAP:
                # Chế độ mới: lấy nhãn có xác suất cao nhất
                best_idx = int(np.argmax(prediction))
                best_label = LABEL_INDEX_MAP.get(best_idx, "Không xác định")
            else:
                # Chế độ cũ: cosine similarity với FastText vectors
                best_label = "Không xác định"
                max_sim = -1.0
                for label, word_vector in LABELS_DICT.items():
                    sim = _cosine_similarity(prediction, word_vector)
                    if sim > max_sim:
                        max_sim = sim
                        best_label = label

            return JsonResponse({'status': 'success', 'result': best_label})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
        finally:
            if os.path.exists(uploaded_file_url):
                os.remove(uploaded_file_url)

    return render(request, 'index.html')
