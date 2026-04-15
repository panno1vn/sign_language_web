import os
import cv2
import torch
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse

# Import file kiến trúc mạng I3D 
try:
    from .pytorch_i3d import InceptionI3d
except ImportError:
    print("!!! CẢNH BÁO: Chưa tìm thấy file pytorch_i3d.py !!!")


# cấu hình path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS_PATH = os.path.join(BASE_DIR, 'models', 'wlasl', 'wlasl_class_list.txt')

WEIGHTS_FILENAME = 'FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt' 
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'wlasl', 'asl100', WEIGHTS_FILENAME)

# dung cho file nst100
NUM_CLASSES = 100 

# tai tu dien label
def load_wlasl_labels(txt_path, num_classes):
    labels = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    word = " ".join(parts[1:]) 
                    labels.append(word)
                else:
                    labels.append(line.strip())
        
        # WLASL100 dung 100 tu dau tien trong file text
        labels = labels[:num_classes] 
        print(f"[*] Đã tải thành công {len(labels)} từ vựng.")
        return labels
    except FileNotFoundError:
        print(f"!!! LỖI: Chưa tìm thấy file {txt_path} !!!")
        return []

WLASL_LABELS = load_wlasl_labels(LABELS_PATH, NUM_CLASSES)

# deploy model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang sử dụng thiết bị tính toán: {device}")

model = None
if os.path.exists(MODEL_PATH) and WLASL_LABELS:
    try:
        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() 
        print(f"[*] Tải mô hình WLASL-{NUM_CLASSES} thành công!")
    except Exception as e:
        print(f"!!! LỖI KHI TẢI MÔ HÌNH: {e} !!!")
else:
    print(f"!!! KHÔNG TÌM THẤY FILE TẠ TẠI: {MODEL_PATH} !!!")

# handle video input
def extract_frames_from_video(video_path, target_frames=64):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Center Crop
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w // 2) - (min_dim // 2)
        start_y = (h // 2) - (min_dim // 2)
        frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # Resize về chuẩn Inception (224x224)
        frame = cv2.resize(frame, (224, 224))
        
        # Chuẩn hóa về [-1, 1]
        frame = (frame / 255.0) * 2 - 1 
        frames.append(frame)
        
    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((224, 224, 3))] * target_frames
    elif len(frames) < target_frames:
        padding = [frames[-1]] * (target_frames - len(frames))
        frames.extend(padding)
    elif len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    # [Channel, Time, Height, Width]
    frames_np = np.array(frames).transpose(3, 0, 1, 2) 
    frames_tensor = torch.from_numpy(frames_np).unsqueeze(0).float()
    
    return frames_tensor.to(device)

#API prediction
def index(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video = request.FILES['video_file']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        uploaded_file_url = fs.path(filename)

        try:
            if model is None:
                raise Exception("Mô hình chưa được khởi tạo. Vui lòng kiểm tra lại file weights.")

            # Trích xuất Tensor từ Video
            video_tensor = extract_frames_from_video(uploaded_file_url, target_frames=64)

            with torch.no_grad():
                predictions = model(video_tensor)
                # Lấy logits từ kết quả (nếu trả về dạng tuple)
                logits = predictions[0] if isinstance(predictions, tuple) else predictions
                
                # SỬA LỖI Ở ĐÂY: Gộp chiều thời gian (dim=2) bằng Max Pooling
                # Biến ma trận từ [1, 100, số_khung_hình] thành [1, 100]
                pooled_logits = torch.max(logits, dim=2)[0]
            
            # Tra cứu vị trí index có điểm số cao nhất trong 100 từ
            predicted_index = torch.argmax(pooled_logits[0]).item()
            best_label = WLASL_LABELS[predicted_index] if WLASL_LABELS else "Lỗi từ điển"

            return JsonResponse({'status': 'success', 'result': best_label})

        except Exception as e:
            print(f"Lỗi hệ thống: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)})
        finally:
            if os.path.exists(uploaded_file_url):
                os.remove(uploaded_file_url)

    return render(request, 'index.html')