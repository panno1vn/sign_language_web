import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
KAGGLE_NPZ = os.path.join(PROJECT_ROOT, "data", "processed", "WLASL_filtered_15plus.npz")

archive = np.load(KAGGLE_NPZ)
first_video_id = archive.files[0]
data = archive[first_video_id]

print(f"Video ID: {first_video_id}")
print(f"Kích thước gốc của Kaggle: {data.shape}")