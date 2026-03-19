# =============================================================================
#  TRAIN.PY  –  Huấn luyện mô hình nhận dạng ngôn ngữ ký hiệu trên Kaggle
#  Dán toàn bộ file này vào một ô code Kaggle rồi bấm Run là xong.
# =============================================================================

# ── THƯ VIỆN ─────────────────────────────────────────────────────────────────
import os
import json
import zipfile
import numpy as np
from tensorflow import keras

# ── CẤU HÌNH (chỉnh ở đây nếu cần) ──────────────────────────────────────────
BATCH_SIZE  = 32    # số mẫu mỗi batch
F_AVG       = 48    # số frame cố định cho mỗi chuỗi (pad/trim về F_AVG)
EPOCHS      = 100   # số epoch tối đa (EarlyStopping sẽ dừng sớm nếu không tiến bộ)
VAL_RATIO   = 0.2   # tỉ lệ validation tách từ tập Train
LR          = 0.001 # learning rate ban đầu
WORKING_DIR = '/kaggle/working'   # thư mục lưu kết quả

# ── NGƯỠNG CHẤT LƯỢNG MÔ HÌNH ────────────────────────────────────────────────
#  Sau khi train xong, script tự động so sánh kết quả với các ngưỡng này:
#
#  categorical_accuracy (trên tập Test)
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  ≥ 85 %  →  ✅  XUẤT SẮC   – model sẵn sàng deploy lên web            │
#  │  70–85 % →  ⚠️  KHÁ TỐT   – dùng được, nên train thêm nếu có thể     │
#  │  50–70 % →  ⚠️  TRUNG BÌNH – cần thêm data hoặc tăng số epoch         │
#  │  < 50 %  →  ❌  CHƯA ĐẠT  – kiểm tra lại data/pipeline               │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  loss (CategoricalCrossentropy + label_smoothing=0.1 nên loss không về 0)
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  ≤ 0.5   →  ✅  TỐT                                                    │
#  │  0.5–1.5 →  ⚠️  CHẤP NHẬN ĐƯỢC                                        │
#  │  > 1.5   →  ❌  CAO – model chưa hội tụ tốt                            │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  Lưu ý: đây là ngưỡng thực tế cho bài toán nhiều lớp (~100-2000 từ).
#  Model dùng thực tế trên web nên đạt tối thiểu 70 % accuracy trên Test.

ACC_EXCELLENT = 0.85   # ≥ 85 % → xuất sắc
ACC_GOOD      = 0.70   # ≥ 70 % → khá tốt / chấp nhận được
ACC_FAIR      = 0.50   # ≥ 50 % → trung bình
LOSS_GOOD     = 0.50   # ≤ 0.5  → loss tốt
LOSS_OK       = 1.50   # ≤ 1.5  → loss chấp nhận được

# ── TÌM ĐƯỜNG DẪN DATA TỰ ĐỘNG ───────────────────────────────────────────────
# Script tự quét /kaggle/input để tìm thư mục npy_arrays (không cần sửa đường dẫn tay).
master_data_path = ""
for _root, _dirs, _files in os.walk('/kaggle/input'):
    if 'npy_arrays' in _dirs:
        master_data_path = os.path.join(_root, 'npy_arrays')
        break

if master_data_path:
    print(f"✅ Tìm thấy data tại: {master_data_path}")
else:
    raise ValueError(
        "❌ Không tìm thấy thư mục 'npy_arrays'!\n"
        "   → Hãy kiểm tra lại phần Add Data bên phải của Kaggle."
    )

# ── DANH SÁCH TỪ VỰNG & LABEL MAP ───────────────────────────────────────────
words     = np.sort(np.array(os.listdir(os.path.join(master_data_path, 'Train'))))
NUM_CLASSES = len(words)
label_map   = {w: i for i, w in enumerate(words)}
print(f"🔥 Tổng số từ vựng: {NUM_CLASSES} lớp")

with open(os.path.join(WORKING_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)


# ── HÀM TIỆN ÍCH ─────────────────────────────────────────────────────────────

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Chuyển mảng 3D (frames, landmarks, coords) → 2D (frames, landmarks*coords).
    Mảng đã là 2D thì giữ nguyên.
    Lý do: MediaPipe đôi khi lưu .npy dạng 3D, dạng 2D, hoặc cả hai lẫn lộn.
    Nếu không flatten trước, np.concatenate sẽ cho ra mảng 3D và gán vào
    X_batch[i] sẽ lỗi → toàn bộ batch cho giá trị 0 → model không học được gì."""
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    return arr


def probe_feature_dim(data_path: str, split: str, words_list) -> int:
    """Đọc một mẫu thực tế để tự xác định tổng số chiều đặc trưng mỗi frame."""
    for word in words_list:
        base = os.path.join(data_path, split, word)
        lh_dir   = os.path.join(base, 'lh_keypoints')
        rh_dir   = os.path.join(base, 'rh_keypoints')
        pose_dir = os.path.join(base, 'pose_keypoints')
        if not os.path.exists(lh_dir):
            continue
        for seq_file in os.listdir(lh_dir):
            try:
                lh   = ensure_2d(np.load(os.path.join(lh_dir,   seq_file)))
                rh   = ensure_2d(np.load(os.path.join(rh_dir,   seq_file)))
                pose = ensure_2d(np.load(os.path.join(pose_dir, seq_file)))
                dim  = pose.shape[1] + lh.shape[1] + rh.shape[1]
                print(f"📐 pose{pose.shape}  lh{lh.shape}  rh{rh.shape}  →  FEATURE_DIM={dim}")
                return dim
            except Exception:
                continue
    raise RuntimeError("Không thể đọc bất kỳ mẫu nào để xác định FEATURE_DIM.")


FEATURE_DIM = probe_feature_dim(master_data_path, 'Train', words)
print(f"📐 FEATURE_DIM={FEATURE_DIM}, F_AVG={F_AVG}")


# ── GENERATOR (load từng batch, tiết kiệm RAM) ────────────────────────────────

class SignSequenceGenerator(keras.utils.Sequence):
    """Generator tải file .npy theo batch.
    • Tự động flatten mảng 3D MediaPipe sang 2D.
    • augment=True: thêm nhiễu Gaussian nhỏ lên keypoints (chỉ dùng khi train).
    """

    def __init__(self, samples, words_list, f_avg, feature_dim,
                 batch_size=32, shuffle=False, augment=False):
        self.samples     = list(samples)
        self.words_list  = words_list
        self.f_avg       = f_avg
        self.feature_dim = feature_dim
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.augment     = augment
        self.on_epoch_end()

    # ------------------------------------------------------------------
    @staticmethod
    def collect(data_path: str, split: str, words_list, label_map) -> list:
        """Quét thư mục và trả về danh sách (lh_path, rh_path, pose_path, label)."""
        samples = []
        for word in words_list:
            base     = os.path.join(data_path, split, word)
            lh_dir   = os.path.join(base, 'lh_keypoints')
            rh_dir   = os.path.join(base, 'rh_keypoints')
            pose_dir = os.path.join(base, 'pose_keypoints')
            if not os.path.exists(lh_dir):
                continue
            for seq_file in os.listdir(lh_dir):
                lh_p   = os.path.join(lh_dir,   seq_file)
                rh_p   = os.path.join(rh_dir,   seq_file)
                pose_p = os.path.join(pose_dir, seq_file)
                if os.path.exists(rh_p) and os.path.exists(pose_p):
                    samples.append((lh_p, rh_p, pose_p, label_map[word]))
        return samples

    # ------------------------------------------------------------------
    def __len__(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))

    def __getitem__(self, idx):
        batch   = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = np.zeros((len(batch), self.f_avg, self.feature_dim), dtype=np.float32)
        y_batch = np.zeros((len(batch), len(self.words_list)),          dtype=np.float32)

        errors = 0
        for i, (lh_p, rh_p, pose_p, label) in enumerate(batch):
            try:
                lh   = self._pad(ensure_2d(np.load(lh_p).astype(np.float32)))
                rh   = self._pad(ensure_2d(np.load(rh_p).astype(np.float32)))
                pose = self._pad(ensure_2d(np.load(pose_p).astype(np.float32)))

                seq = np.concatenate((pose, lh, rh), axis=1)   # (F_AVG, FEATURE_DIM)

                if self.augment:
                    seq += np.random.normal(0, 0.005, seq.shape).astype(np.float32)

                X_batch[i] = seq
                y_batch[i, label] = 1.0

            except Exception as e:
                errors += 1
                if errors <= 3:              # in tối đa 3 lỗi đầu để dễ debug
                    print(f"⚠️  Lỗi load file: {lh_p}\n   → {e}")

        if errors > len(batch) // 2:
            print(f"❌ Batch {idx}: {errors}/{len(batch)} mẫu bị lỗi! "
                  f"Các mẫu lỗi sẽ là vector 0 và KHÔNG đóng góp cho việc học. "
                  f"Kiểm tra lại tính toàn vẹn của các file .npy.")

        return X_batch, y_batch

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        """Pad bằng frame cuối hoặc trim về đúng F_AVG frames."""
        n = arr.shape[0]
        if n == 0:
            return np.zeros((self.f_avg, arr.shape[1]), dtype=np.float32)
        if n >= self.f_avg:
            return arr[:self.f_avg]
        pad = np.tile(arr[[-1]], (self.f_avg - n, 1))
        return np.concatenate((arr, pad), axis=0)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)


# ── CHUẨN BỊ DỮ LIỆU ─────────────────────────────────────────────────────────
print("\n📂 Quét file Train...")
all_train = SignSequenceGenerator.collect(master_data_path, 'Train', words, label_map)

print("📂 Quét file Test...")
all_test  = SignSequenceGenerator.collect(master_data_path, 'Test',  words, label_map)

if len(all_train) == 0:
    raise RuntimeError("Không tìm thấy mẫu Train nào! Kiểm tra cấu trúc thư mục.")

# Tách validation 20% từ Train
np.random.seed(42)
np.random.shuffle(all_train)
n_val        = int(len(all_train) * VAL_RATIO)
val_samples  = all_train[:n_val]
train_samples = all_train[n_val:]
print(f"✅ Train: {len(train_samples)}  |  Val: {len(val_samples)}  |  Test: {len(all_test)} mẫu")

train_gen = SignSequenceGenerator(train_samples, words, F_AVG, FEATURE_DIM,
                                  batch_size=BATCH_SIZE, shuffle=True,  augment=True)
val_gen   = SignSequenceGenerator(val_samples,   words, F_AVG, FEATURE_DIM,
                                  batch_size=BATCH_SIZE, shuffle=False, augment=False)
test_gen  = SignSequenceGenerator(all_test,      words, F_AVG, FEATURE_DIM,
                                  batch_size=BATCH_SIZE, shuffle=False, augment=False)

# Kiểm tra batch đầu tiên – giúp phát hiện sớm dữ liệu bị lỗi
X0, y0 = train_gen[0]
print(f"\n🔍 Batch sanity-check: X{X0.shape}  y{y0.shape}")
print(f"   X  max={X0.max():.4f}  min={X0.min():.4f}  mean={X0.mean():.4f}")
print(f"   y  sum mỗi hàng (phải = 1): {y0.sum(axis=1)[:8]}")
n_zero = (X0.std(axis=(1, 2)) == 0).sum()
if n_zero:
    pct = n_zero / len(X0) * 100
    msg = f"⚠️  {n_zero}/{len(X0)} ({pct:.0f}%) mẫu có input toàn hằng số → kiểm tra lại file .npy!"
    if pct > 10:
        raise RuntimeError(msg + "\n   Quá nhiều mẫu lỗi – training sẽ không hội tụ. Dừng lại để bạn sửa data trước.")
    print(msg)


# ── KIẾN TRÚC MÔ HÌNH ────────────────────────────────────────────────────────
def build_model(seq_len: int, feature_dim: int, num_classes: int) -> keras.Model:
    inp = keras.Input(shape=(seq_len, feature_dim))

    # 3 lớp Bidirectional LSTM (dropout nằm trong LSTM để ổn định hơn)
    x = keras.layers.Bidirectional(
            keras.layers.LSTM(256, return_sequences=True, dropout=0.3))(inp)
    x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = keras.layers.Bidirectional(
            keras.layers.LSTM(64,  dropout=0.2))(x)

    # Classifier head
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    out = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inp, out)


model = build_model(F_AVG, FEATURE_DIM, NUM_CLASSES)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
    # label_smoothing giúp tránh overfit khi có nhiều lớp (~2000)
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['categorical_accuracy'],
)


# ── CALLBACKS ────────────────────────────────────────────────────────────────
ckpt_path = os.path.join(WORKING_DIR, 'best_model.keras')
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
]


# ── HUẤN LUYỆN ───────────────────────────────────────────────────────────────
print("\n🚀 Bắt đầu huấn luyện...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ── ĐÁNH GIÁ ─────────────────────────────────────────────────────────────────
def print_verdict(acc: float, loss: float) -> None:
    """In kết quả đánh giá kèm nhận xét chất lượng dựa trên các ngưỡng đã định."""
    sep = "=" * 60
    print("\n" + sep)
    print("📊  KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST".center(60))
    print(sep)
    print(f"  categorical_accuracy : {acc  * 100:.2f} %")
    print(f"  loss                 : {loss:.4f}")
    print("-" * 60)

    # ── Nhận xét accuracy ──────────────────────────────────────────
    if acc >= ACC_EXCELLENT:
        acc_msg = "✅  XUẤT SẮC   – model sẵn sàng deploy lên web"
    elif acc >= ACC_GOOD:
        acc_msg = "⚠️   KHÁ TỐT   – dùng được, nên train thêm nếu có thể"
    elif acc >= ACC_FAIR:
        acc_msg = "⚠️   TRUNG BÌNH – cần thêm data hoặc tăng số epoch"
    else:
        acc_msg = "❌  CHƯA ĐẠT  – kiểm tra lại data / pipeline"

    # ── Nhận xét loss ──────────────────────────────────────────────
    if loss <= LOSS_GOOD:
        loss_msg = "✅  TỐT"
    elif loss <= LOSS_OK:
        loss_msg = "⚠️   CHẤP NHẬN ĐƯỢC"
    else:
        loss_msg = "❌  CAO – model chưa hội tụ tốt"

    print(f"  Accuracy : {acc_msg}")
    print(f"  Loss     : {loss_msg}")
    print("-" * 60)
    print("  Ngưỡng tham khảo:")
    print("    accuracy  ≥ 85 % → Xuất sắc  |  ≥ 70 % → Khá tốt  |  ≥ 50 % → Trung bình")
    print("    loss      ≤ 0.5  → Tốt        |  ≤ 1.5  → Chấp nhận được")
    print("    (loss dùng CategoricalCrossentropy + label_smoothing=0.1, không về 0)")
    print("=" * 60 + "\n")


print("\n📊 Đánh giá trên tập Test...")
loss, acc = model.evaluate(test_gen, verbose=1)
print_verdict(acc, loss)


# ── LƯU MÔ HÌNH ──────────────────────────────────────────────────────────────
h5_path = os.path.join(WORKING_DIR, 'best_model.h5')
model.save(h5_path)
print(f"💾 Đã lưu model: {h5_path}")


# ── ĐÓNG GÓI FILE ZIP ────────────────────────────────────────────────────────
zip_path = os.path.join(WORKING_DIR, 'Web_Deployment_Files.zip')
print(f"\n📦 Đóng gói {zip_path} ...")
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for fname in ['best_model.h5', 'best_model.keras', 'label_map.json']:
        fpath = os.path.join(WORKING_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, arcname=fname)
            print(f"   ✅ {fname}")

print("\n🎉 XONG! Tải file Web_Deployment_Files.zip về máy để triển khai lên Web.")

