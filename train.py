import os
import json
import zipfile
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

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


# --- XÁC ĐỊNH KÍCH THƯỚC ĐẦU VÀO TỪ MỘT MẪU DỮ LIỆU ---
def _probe_feature_dim(data_path, split, words_list):
    """Đọc một mẫu để xác định số chiều đặc trưng mỗi frame."""
    for word in words_list:
        word_path = os.path.join(data_path, split, word)
        lh_folder = os.path.join(word_path, "lh_keypoints")
        rh_folder = os.path.join(word_path, "rh_keypoints")
        pose_folder = os.path.join(word_path, "pose_keypoints")
        if not os.path.exists(lh_folder):
            continue
        for seq in os.listdir(lh_folder):
            try:
                lh = np.load(os.path.join(lh_folder, seq))
                rh = np.load(os.path.join(rh_folder, seq))
                pose = np.load(os.path.join(pose_folder, seq))
                sample = np.concatenate((pose[:1], lh[:1], rh[:1]), axis=1)
                return sample.shape[1]
            except Exception:
                continue
    raise RuntimeError("Không thể đọc bất kỳ mẫu dữ liệu nào để xác định số chiều đặc trưng.")


FEATURE_DIM = _probe_feature_dim(master_data_path, 'Train', words)
F_AVG = 48  # Số frame cố định cho mỗi chuỗi
print(f"📐 Số chiều đặc trưng mỗi frame: {FEATURE_DIM}, Số frame: {F_AVG}")


# --- GENERATOR TIẾT KIỆM BỘ NHỚ CHO DATASET LỚN ---
class SignSequenceGenerator(keras.utils.Sequence):
    """
    Generator load dữ liệu theo batch, tiết kiệm RAM khi dataset có
    hàng chục nghìn video với hàng nghìn từ vựng.
    """

    def __init__(self, data_path, split, words_list, label_map, f_avg, feature_dim, batch_size=32, shuffle=True):
        self.data_path = data_path
        self.split = split
        self.words_list = words_list
        self.label_map = label_map
        self.f_avg = f_avg
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = self._collect_samples()
        self.on_epoch_end()

    def _collect_samples(self):
        samples = []
        for word in self.words_list:
            word_path = os.path.join(self.data_path, self.split, word)
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
                    samples.append((lh_path, rh_path, pose_path, self.label_map[word]))
        return samples

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = np.zeros((len(batch), self.f_avg, self.feature_dim), dtype=np.float32)
        y_batch = np.zeros((len(batch), len(self.words_list)), dtype=np.float32)

        for i, (lh_path, rh_path, pose_path, label) in enumerate(batch):
            try:
                res_lh = np.load(lh_path).astype(np.float32)
                res_rh = np.load(rh_path).astype(np.float32)
                res_pose = np.load(pose_path).astype(np.float32)

                res_lh = self._pad_or_trim(res_lh)
                res_rh = self._pad_or_trim(res_rh)
                res_pose = self._pad_or_trim(res_pose)

                X_batch[i] = np.concatenate((res_pose, res_lh, res_rh), axis=1)
                y_batch[i, label] = 1.0
            except Exception:
                pass

        return X_batch, y_batch

    def _pad_or_trim(self, arr):
        n = arr.shape[0]
        if n == 0:
            return np.zeros((self.f_avg, arr.shape[1]), dtype=np.float32)
        if n >= self.f_avg:
            return arr[:self.f_avg, :]
        # Lặp frame cuối để đủ độ dài
        pad = np.tile(arr[[-1], :], (self.f_avg - n, 1))
        return np.concatenate((arr, pad), axis=0)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)


# --- CHUẨN BỊ DỮ LIỆU ---
print("📂 Đang quét danh sách file Train...")
train_gen = SignSequenceGenerator(
    master_data_path, 'Train', words, label_map, F_AVG, FEATURE_DIM, batch_size=32, shuffle=True
)

print("📂 Đang quét danh sách file Test...")
test_gen = SignSequenceGenerator(
    master_data_path, 'Test', words, label_map, F_AVG, FEATURE_DIM, batch_size=32, shuffle=False
)

# Tách validation từ training (20%)
train_samples = train_gen.samples
np.random.seed(42)
np.random.shuffle(train_samples)
val_split = int(len(train_samples) * 0.2)
val_samples = train_samples[:val_split]
train_samples = train_samples[val_split:]
train_gen.samples = train_samples

val_gen = SignSequenceGenerator(
    master_data_path, 'Train', words, label_map, F_AVG, FEATURE_DIM, batch_size=32, shuffle=False
)
val_gen.samples = val_samples

print(f"✅ Train: {len(train_gen.samples)} mẫu | Val: {len(val_gen.samples)} mẫu | Test: {len(test_gen.samples)} mẫu")


# --- KÍCH HOẠT MIXED PRECISION (Tăng tốc GPU, giảm VRAM) ---
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("⚡ Mixed Precision float16 đã kích hoạt.")
except Exception:
    print("⚠️ Mixed Precision không khả dụng, dùng float32.")


# --- KIẾN TRÚC MÔ HÌNH CHO DATASET LỚN (2000 LỚP) ---
def build_model(seq_len, feature_dim, num_classes):
    inputs = keras.Input(shape=(seq_len, feature_dim))

    # Encoder Bidirectional LSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True)
    )(inputs)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True)
    )(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64)
    )(x)
    x = keras.layers.Dropout(0.3)(x)

    # Classifier head
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    # Output phải là float32 để tương thích mixed precision
    outputs = keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    return model


model = build_model(F_AVG, FEATURE_DIM, NUM_CLASSES)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
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
