import os
import json
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- TỰ ĐỘNG DÒ ĐƯỜNG DẪN KAGGLE ---
master_data_path = ""
for root, dirs, files in os.walk('/kaggle/input'):
    if 'npy_arrays' in dirs:
        master_data_path = os.path.join(root, 'npy_arrays')
        break

if master_data_path != "":
    print(f"✅ Đã tự động dò trúng đường dẫn: {master_data_path}")
else:
    raise ValueError("❌ KAGGLE CHƯA NHẬN DATA! Bạn hãy nhìn sang cột bên phải xem đã Add Data thành công chưa nhé.")

# --- LẤY DANH SÁCH TỪ VỰNG ---
all_words = np.sort(np.array(os.listdir(os.path.join(master_data_path, 'Train'))))

# Dùng TOÀN BỘ từ vựng có trong dataset (tối đa ~2000 chữ)
NUM_CLASSES = len(all_words)
words = all_words[:NUM_CLASSES]

label_map = {label: num for num, label in enumerate(words)}
print(f"🔥 Huấn luyện với {NUM_CLASSES} từ vựng ({len(words)} lớp)")

with open('/kaggle/working/label_map.json', 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# HELPER: làm phẳng mảng nếu là 3D (frames, landmarks, coords) → (frames, features)
# Dữ liệu MediaPipe có thể được lưu theo cả hai định dạng; hàm này xử lý cả hai.
# ---------------------------------------------------------------------------
def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Converts (frames, landmarks, coords) → (frames, landmarks*coords).
    2-D arrays are returned unchanged."""
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    return arr


# --- XÁC ĐỊNH KÍCH THƯỚC ĐẦU VÀO TỪ MỘT MẪU DỮ LIỆU ---
def _probe_feature_dim(data_path, split, words_list):
    """Reads one real sample to determine the total flat feature size per frame."""
    for word in words_list:
        word_path = os.path.join(data_path, split, word)
        lh_folder = os.path.join(word_path, "lh_keypoints")
        rh_folder = os.path.join(word_path, "rh_keypoints")
        pose_folder = os.path.join(word_path, "pose_keypoints")
        if not os.path.exists(lh_folder):
            continue
        for seq in os.listdir(lh_folder):
            try:
                lh = _ensure_2d(np.load(os.path.join(lh_folder, seq)))
                rh = _ensure_2d(np.load(os.path.join(rh_folder, seq)))
                pose = _ensure_2d(np.load(os.path.join(pose_folder, seq)))
                feature_dim = pose.shape[1] + lh.shape[1] + rh.shape[1]
                print(f"📐 Kích thước mẫu: pose={pose.shape}, lh={lh.shape}, rh={rh.shape} → feature_dim={feature_dim}")
                return feature_dim
            except Exception:
                continue
    raise RuntimeError("Không thể đọc bất kỳ mẫu dữ liệu nào để xác định số chiều đặc trưng.")


FEATURE_DIM = _probe_feature_dim(master_data_path, 'Train', words)
F_AVG = 48  # Số frame cố định cho mỗi chuỗi
print(f"📐 FEATURE_DIM={FEATURE_DIM}, F_AVG={F_AVG}")


# --- GENERATOR TIẾT KIỆM BỘ NHỚ CHO DATASET LỚN ---
class SignSequenceGenerator(keras.utils.Sequence):
    """
    Batch generator that loads .npy keypoint files on demand.
    Handles both 2-D (frames, features) and 3-D (frames, landmarks, coords) arrays.
    Set augment=True to add small Gaussian noise during training.
    """

    def __init__(self, words_list, label_map, f_avg, feature_dim,
                 batch_size=32, shuffle=True, augment=False, samples=None):
        self.words_list = words_list
        self.label_map = label_map
        self.f_avg = f_avg
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        # Accept a pre-built sample list (used for val/test splits)
        self.samples = samples if samples is not None else []
        if samples is None:
            raise ValueError("Pass a pre-built samples list via the 'samples' parameter.")
        self.on_epoch_end()

    @staticmethod
    def collect_samples(data_path, split, words_list, label_map):
        """Scans the directory tree and returns a list of (lh, rh, pose, label) tuples."""
        samples = []
        for word in words_list:
            word_path = os.path.join(data_path, split, word)
            lh_folder = os.path.join(word_path, "lh_keypoints")
            rh_folder = os.path.join(word_path, "rh_keypoints")
            pose_folder = os.path.join(word_path, "pose_keypoints")
            if not os.path.exists(lh_folder):
                continue
            for seq in os.listdir(lh_folder):
                lh_path = os.path.join(lh_folder, seq)
                rh_path = os.path.join(rh_folder, seq)
                pose_path = os.path.join(pose_folder, seq)
                if os.path.exists(rh_path) and os.path.exists(pose_path):
                    samples.append((lh_path, rh_path, pose_path, label_map[word]))
        return samples

    def __len__(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = np.zeros((len(batch), self.f_avg, self.feature_dim), dtype=np.float32)
        y_batch = np.zeros((len(batch), len(self.words_list)), dtype=np.float32)

        load_errors = 0
        for i, (lh_path, rh_path, pose_path, label) in enumerate(batch):
            try:
                # Load và làm phẳng: xử lý cả 2D (frames, feat) và 3D (frames, lm, coord)
                res_lh = _ensure_2d(np.load(lh_path).astype(np.float32))
                res_rh = _ensure_2d(np.load(rh_path).astype(np.float32))
                res_pose = _ensure_2d(np.load(pose_path).astype(np.float32))

                res_lh = self._pad_or_trim(res_lh)
                res_rh = self._pad_or_trim(res_rh)
                res_pose = self._pad_or_trim(res_pose)

                seq = np.concatenate((res_pose, res_lh, res_rh), axis=1)  # (F_AVG, FEATURE_DIM)

                if self.augment:
                    # Thêm nhiễu Gaussian nhỏ để tránh overfitting
                    seq += np.random.normal(0, 0.005, seq.shape).astype(np.float32)

                X_batch[i] = seq
                y_batch[i, label] = 1.0
            except Exception as exc:
                load_errors += 1
                if load_errors == 1:
                    # In lần đầu gặp lỗi để dễ debug
                    print(f"⚠️  Lỗi load mẫu (lh={lh_path}): {exc}")

        if load_errors > len(batch) // 2:
            print(f"❌ Cảnh báo: {load_errors}/{len(batch)} mẫu trong batch {idx} bị lỗi!")

        return X_batch, y_batch

    def _pad_or_trim(self, arr: np.ndarray) -> np.ndarray:
        """arr must be 2-D: (frames, features). Pads with last frame or trims."""
        n = arr.shape[0]
        if n == 0:
            return np.zeros((self.f_avg, arr.shape[1]), dtype=np.float32)
        if n >= self.f_avg:
            return arr[:self.f_avg, :]
        pad = np.tile(arr[[-1], :], (self.f_avg - n, 1))
        return np.concatenate((arr, pad), axis=0)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)


# --- CHUẨN BỊ DỮ LIỆU ---
print("📂 Đang quét danh sách file Train...")
all_train_samples = SignSequenceGenerator.collect_samples(master_data_path, 'Train', words, label_map)

print("📂 Đang quét danh sách file Test...")
all_test_samples = SignSequenceGenerator.collect_samples(master_data_path, 'Test', words, label_map)

# Tách validation từ training (20%)
np.random.seed(42)
np.random.shuffle(all_train_samples)
val_split = int(len(all_train_samples) * 0.2)
val_samples = all_train_samples[:val_split]
train_samples = all_train_samples[val_split:]

print(f"✅ Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(all_test_samples)} mẫu")
if len(train_samples) == 0:
    raise RuntimeError("Không tìm thấy mẫu Train nào! Kiểm tra lại đường dẫn và cấu trúc thư mục.")

train_gen = SignSequenceGenerator(
    words, label_map, F_AVG, FEATURE_DIM,
    batch_size=32, shuffle=True, augment=True, samples=train_samples
)
val_gen = SignSequenceGenerator(
    words, label_map, F_AVG, FEATURE_DIM,
    batch_size=32, shuffle=False, augment=False, samples=val_samples
)
test_gen = SignSequenceGenerator(
    words, label_map, F_AVG, FEATURE_DIM,
    batch_size=32, shuffle=False, augment=False, samples=all_test_samples
)

# Kiểm tra sanity: lấy 1 batch và in shape + giá trị
X_sample, y_sample = train_gen[0]
print(f"🔍 Batch sanity check: X={X_sample.shape}, y={y_sample.shape}")
print(f"   X max={X_sample.max():.4f}, X min={X_sample.min():.4f}, X mean={X_sample.mean():.4f}")
print(f"   y sum per sample (should be 1): {y_sample.sum(axis=1)[:5]}")
zero_x = (X_sample.std(axis=(1, 2)) == 0).sum()
if zero_x > 0:
    print(f"⚠️  {zero_x}/{len(X_sample)} mẫu trong batch đầu có input hằng số (toàn 0) - kiểm tra lại .npy files!")


# --- KIẾN TRÚC MÔ HÌNH CHO DATASET LỚN ---
def build_model(seq_len, feature_dim, num_classes):
    inputs = keras.Input(shape=(seq_len, feature_dim))

    # Encoder Bidirectional LSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.0)
    )(inputs)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.0)
    )(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.0)
    )(x)

    # Classifier head
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)


model = build_model(F_AVG, FEATURE_DIM, NUM_CLASSES)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['categorical_accuracy']
)

# --- CALLBACKS ---
checkpoint_path = '/kaggle/working/best_model.keras'
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
]

# --- HUẤN LUYỆN ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=callbacks
)

# --- ĐÁNH GIÁ ---
print("\n📊 Đánh giá trên tập Test...")
loss, acc = model.evaluate(test_gen)
print(f"\n🎯 ĐỘ CHÍNH XÁC TEST: {acc * 100:.2f}%")

# Lưu thêm bản .h5 để tương thích với web cũ nếu cần
model.save('/kaggle/working/best_model.h5')

# --- ĐÓNG GÓI FILE ZIP ---
print("📦 Đang đóng gói file ZIP để triển khai lên Web...")
with zipfile.ZipFile('/kaggle/working/Web_Deployment_Files.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for fname in ['best_model.h5', 'best_model.keras', 'label_map.json']:
        fpath = f'/kaggle/working/{fname}'
        if os.path.exists(fpath):
            zipf.write(fpath, arcname=fname)
            print(f"  ✅ Đã thêm {fname}")

print("🚀 THÀNH CÔNG! File Web_Deployment_Files.zip đã sẵn sàng để tải về và triển khai!")

