import os
import cv2
import numpy as np
import mediapipe as mp
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse

# Đặt đường dẫn tuyệt đối để không bao giờ lỗi
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', '12-14.h5')
LABELS_DICT_PATH = os.path.join(BASE_DIR, 'models', 'labels.npz')

# ── WLASL (I3D) model ────────────────────────────────────────────────────
# Pre-trained weights from: https://github.com/dxli94/WLASL
# Download: https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48
# Place checkpoint at: models/wlasl/nslt_100.pth.tar  (or nslt_300.pth.tar, etc.)
WLASL_WEIGHTS_DIR = os.path.join(BASE_DIR, 'models', 'wlasl')
WLASL_CLASS_LIST_PATH = os.path.join(BASE_DIR, 'models', 'wlasl_class_list.txt')

WLASL_MODEL = None
WLASL_LABELS = None
ACTIVE_MODEL = 'none'

# Try to load the WLASL I3D model (requires weights to be downloaded separately)
try:
    from translator.wlasl_model import load_wlasl_model, load_class_list, predict_wlasl

    if os.path.isdir(WLASL_WEIGHTS_DIR):
        _candidates = sorted(
            [f for f in os.listdir(WLASL_WEIGHTS_DIR) if f.endswith('.pth.tar') or f.endswith('.pth')],
            key=lambda f: int(''.join(filter(str.isdigit, f)) or '0')
        )
        if _candidates:
            _weights_path = os.path.join(WLASL_WEIGHTS_DIR, _candidates[0])
            WLASL_LABELS = load_class_list(WLASL_CLASS_LIST_PATH)
            WLASL_MODEL = load_wlasl_model(_weights_path, num_classes=len(WLASL_LABELS))
            ACTIVE_MODEL = 'wlasl'
            print(f"[INFO] WLASL I3D model loaded: {_candidates[0]}  ({len(WLASL_LABELS)} classes)")
        else:
            print("[INFO] No WLASL weights found in models/wlasl/ — falling back to local model.")
    else:
        print("[INFO] models/wlasl/ directory not found — falling back to local model.")
except Exception as _e:
    print(f"[WARNING] Could not load WLASL model: {_e}")

# ── Fallback: existing TensorFlow / Keras model ──────────────────────────
if ACTIVE_MODEL == 'none':
    try:
        import tensorflow as tf

        _tf_model = tf.keras.models.load_model(MODEL_PATH)
        _labels_dict = np.load(LABELS_DICT_PATH, allow_pickle=True)
        ACTIVE_MODEL = 'local'
        print("[INFO] Local TF/Keras model loaded.")
    except Exception as _e:
        _tf_model = None
        _labels_dict = {}
        print(f"[WARNING] Could not load local TF/Keras model: {_e}")
else:
    _tf_model = None
    _labels_dict = {}

# --- CẤU HÌNH MEDIAPIPE (Giữ nguyên từ Kaggle) ---
# Danh sách các điểm face được lọc dùng khi huấn luyện
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
                 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150,
                 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246,
                 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310,
                 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377,
                 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
                 466, 468, 473]

# Bộ chỉ mục tái tạo đúng cấu trúc dữ liệu đã dùng khi huấn luyện trên Kaggle
BUGGY_INDICES = (
        list(range(21)) +
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
            # ── WLASL I3D model path ──────────────────────────────────────
            if ACTIVE_MODEL == 'wlasl':
                results = predict_wlasl(WLASL_MODEL, uploaded_file_url, WLASL_LABELS)
                if results is None:
                    return JsonResponse({'status': 'error',
                                         'message': 'Không thể đọc video.'})
                top_label, top_prob = results[0]
                top3 = [{'label': lbl, 'prob': round(prob * 100, 1)}
                        for lbl, prob in results]
                return JsonResponse({
                    'status': 'success',
                    'result': top_label,
                    'confidence': round(top_prob * 100, 1),
                    'top3': top3,
                    'model': 'WLASL I3D',
                })

            # ── Local TF/Keras model path ─────────────────────────────────
            if ACTIVE_MODEL == 'local' and _tf_model is not None:
                features = extract_features_from_video(uploaded_file_url)
                prediction_vector = _tf_model.predict(features)[0]

                best_label = "Không xác định"
                max_similarity = -1.0
                for label in _labels_dict.keys():
                    sim = cosine_similarity(prediction_vector, _labels_dict[label])
                    if sim > max_similarity:
                        max_similarity = sim
                        best_label = label

                return JsonResponse({
                    'status': 'success',
                    'result': best_label,
                    'confidence': round(float(max_similarity) * 100, 1),
                    'top3': [{'label': best_label,
                               'prob': round(float(max_similarity) * 100, 1)}],
                    'model': 'Local (TF/Keras)',
                })

            return JsonResponse({'status': 'error',
                                  'message': 'Chưa có model nào được tải. '
                                             'Vui lòng tải file trọng số WLASL '
                                             'vào thư mục models/wlasl/.'})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
        finally:
            if os.path.exists(uploaded_file_url):
                os.remove(uploaded_file_url)

    return render(request, 'index.html', {'active_model': ACTIVE_MODEL})