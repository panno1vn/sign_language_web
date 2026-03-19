# =============================================================================
#  TRAIN.PY  –  Huấn luyện mô hình nhận dạng ngôn ngữ ký hiệu trên Kaggle
#  Dán toàn bộ file này vào một ô code Kaggle rồi bấm Run là xong.
# =============================================================================

# ── THƯ VIỆN ─────────────────────────────────────────────────────────────────
import math
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

# Số mẫu train tối thiểu trên mỗi lớp để mô hình có thể học được.
# Dưới ngưỡng này model gần như chắc chắn không hội tụ.
MIN_SAMPLES_PER_CLASS        = 5    # dưới mức này → cảnh báo nghiêm trọng
RECOMMENDED_SAMPLES_PER_CLASS = 20  # khuyến nghị để model học tốt

# Số mẫu train tối thiểu để GIỮ một lớp (lọc tự động – Lựa chọn 2).
# Các lớp có ÍT HƠN con số này trong tập train sẽ bị loại hoàn toàn.
# • Đặt = 0 để tắt lọc (dùng toàn bộ từ vựng – không khuyến nghị khi ít data).
FILTER_MIN_SAMPLES = RECOMMENDED_SAMPLES_PER_CLASS

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


# ── KIỂM TRA ĐỦ DỮ LIỆU ─────────────────────────────────────────────────────
def check_data_sufficiency(samples: list, num_classes: int) -> None:
    """Kiểm tra liệu số mẫu trên mỗi lớp có đủ để mô hình học được không.

    Lý do cần kiểm tra:
    • loss ban đầu của mô hình ngẫu nhiên ≈ ln(num_classes).
    • Nếu sau nhiều epoch loss vẫn gần giá trị này, nghĩa là mô hình đang
      đoán mò – nguyên nhân số 1 là quá ít mẫu mỗi lớp.
    • Cần ≥ MIN_SAMPLES_PER_CLASS mẫu/lớp để bắt đầu học;
      ≥ RECOMMENDED_SAMPLES_PER_CLASS mẫu/lớp để học tốt.
    """
    counts: dict = {}
    for *_, label in samples:
        counts[label] = counts.get(label, 0) + 1

    total          = len(samples)
    avg_per_class  = total / max(num_classes, 1)
    below_min      = sum(1 for c in counts.values() if c < MIN_SAMPLES_PER_CLASS)
    missing        = num_classes - len(counts)      # lớp có 0 mẫu
    random_loss    = math.log(max(num_classes, 1))  # loss khi đoán mò

    sep = "=" * 65
    print(f"\n{sep}")
    print("📊  KIỂM TRA ĐỦ DỮ LIỆU TRƯỚC KHI TRAIN".center(65))
    print(sep)
    print(f"  Tổng số lớp          : {num_classes}")
    print(f"  Tổng mẫu train       : {total:,}")
    print(f"  Trung bình / lớp     : {avg_per_class:.1f} mẫu")
    print(f"  Lớp < {MIN_SAMPLES_PER_CLASS!s:<2} mẫu        : {below_min + missing} / {num_classes}")
    print(f"  Loss đoán ngẫu nhiên : ~{random_loss:.2f}  (= ln({num_classes}))")
    print(f"  Loss cần đạt         : ≤ {LOSS_OK} (chấp nhận)  |  ≤ {LOSS_GOOD} (tốt)")
    print("-" * 65)

    if avg_per_class < MIN_SAMPLES_PER_CLASS:
        needed_total = num_classes * RECOMMENDED_SAMPLES_PER_CLASS
        max_classes  = total // RECOMMENDED_SAMPLES_PER_CLASS
        print(f"  ❌  CẢNH BÁO NGHIÊM TRỌNG:")
        print(f"      Trung bình chỉ {avg_per_class:.1f} mẫu/lớp – model gần như CHẮC CHẮN")
        print(f"      sẽ KHÔNG HỘI TỤ và chỉ đoán mò (accuracy ≈ 0%).")
        print(f"      Để giải quyết, chọn một trong hai:")
        print(f"        1️⃣  Thu thập thêm data:")
        print(f"            Cần tổng ~{needed_total:,} mẫu (= {num_classes} lớp × {RECOMMENDED_SAMPLES_PER_CLASS} mẫu/lớp)")
        print(f"            (hiện tại còn thiếu ~{max(0, needed_total - total):,} mẫu)")
        print(f"        2️⃣  Giảm số lớp xuống ≤ {max_classes}")
        print(f"            (giữ lại các lớp có nhiều mẫu nhất)")
    elif avg_per_class < RECOMMENDED_SAMPLES_PER_CLASS:
        print(f"  ⚠️   CẢNH BÁO: chỉ có {avg_per_class:.1f} mẫu/lớp – thấp hơn khuyến nghị.")
        print(f"      Model có thể học kém. Khuyến nghị ≥ {RECOMMENDED_SAMPLES_PER_CLASS} mẫu/lớp.")
        print(f"      Cần thêm ~{max(0, num_classes * RECOMMENDED_SAMPLES_PER_CLASS - total):,} mẫu nữa.")
    else:
        print(f"  ✅  Số mẫu/lớp ổn ({avg_per_class:.1f} ≥ {RECOMMENDED_SAMPLES_PER_CLASS}).")

    print(sep + "\n")


check_data_sufficiency(train_samples, NUM_CLASSES)


# ── LỌC LỚP TỰ ĐỘNG (LỰẠA CHỌN 2) ─────────────────────────────────────────
def filter_top_classes(train_samp: list, val_samp: list, test_samp: list,
                       current_words: np.ndarray, filter_min: int):
    """Giữ lại chỉ những lớp có đủ ≥ filter_min mẫu trong tập train.

    • Các lớp dưới ngưỡng bị loại khỏi train, val VÀ test.
    • Nhãn được đánh lại về 0…K-1 (sắp xếp giảm dần theo số mẫu train).
    • label_map.json được ghi lại tự động.

    Trả về: (train_new, val_new, test_new, words_new, label_map_new, num_classes_new)
    """
    counts: dict = {}
    for lh, rh, pose, lbl in train_samp:
        counts[lbl] = counts.get(lbl, 0) + 1

    kept = sorted(
        [lbl for lbl, cnt in counts.items() if cnt >= filter_min],
        key=lambda l: counts[l],
        reverse=True,
    )

    if not kept:
        max_cnt = max(counts.values(), default=0)
        raise RuntimeError(
            f"FILTER_MIN_SAMPLES={filter_min} quá cao – không còn lớp nào sau khi lọc!\n"
            f"  Lớp có nhiều mẫu nhất chỉ có {max_cnt} mẫu train.\n"
            f"  Hãy giảm FILTER_MIN_SAMPLES xuống ≤ {max_cnt}."
        )

    old_to_new = {old: new for new, old in enumerate(kept)}
    kept_set   = frozenset(kept)

    def _remap(samples):
        return [(lh, rh, pose, old_to_new[lbl])
                for lh, rh, pose, lbl in samples
                if lbl in kept_set]

    words_new = np.array([current_words[lbl] for lbl in kept])
    lmap_new  = {w: i for i, w in enumerate(words_new)}
    return _remap(train_samp), _remap(val_samp), _remap(test_samp), words_new, lmap_new, len(kept)


if FILTER_MIN_SAMPLES > 0:
    _sep = "=" * 65
    print(f"\n{_sep}")
    print(f"{'✂️   LỌC LỚP TỰ ĐỘNG  (FILTER_MIN_SAMPLES = ' + str(FILTER_MIN_SAMPLES) + ')':^65}")
    print(_sep)
    print(f"  Trước lọc : {NUM_CLASSES:,} lớp  |  {len(train_samples):,} mẫu train")
    train_samples, val_samples, all_test, words, label_map, NUM_CLASSES = \
        filter_top_classes(train_samples, val_samples, all_test, words, FILTER_MIN_SAMPLES)
    _avg  = len(train_samples) / max(NUM_CLASSES, 1)
    _rloss = math.log(max(NUM_CLASSES, 1))
    print(f"  Sau lọc   : {NUM_CLASSES:,} lớp  |  {len(train_samples):,} mẫu train"
          f"  |  {len(val_samples):,} val  |  {len(all_test):,} test")
    print(f"  Trung bình / lớp   : {_avg:.1f} mẫu")
    print(f"  Loss ngẫu nhiên    : ~{_rloss:.2f}  (= ln({NUM_CLASSES}))")
    with open(os.path.join(WORKING_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  💾 label_map.json đã cập nhật ({NUM_CLASSES} từ)")
    if _avg >= RECOMMENDED_SAMPLES_PER_CLASS:
        print(f"  ✅ Dữ liệu sau lọc đủ điều kiện để model học tốt!")
    else:
        print(f"  ⚠️  Sau lọc vẫn chỉ {_avg:.1f} mẫu/lớp – cân nhắc tăng thêm data.")
    print(_sep + "\n")

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
class QualityMonitorCallback(keras.callbacks.Callback):
    """In một dòng nhận xét chất lượng sau mỗi epoch dựa trên val_accuracy và val_loss.

    Sau epoch thứ 5, nếu val_loss vẫn ≥ 90% của loss đoán ngẫu nhiên (= ln(NUM_CLASSES)),
    tức là model chưa học được gì, callback sẽ in một cảnh báo chi tiết một lần.

    Ví dụ output:
      [Epoch  5] val_acc=  0.56 % ❌ CHƯA ĐẠT   |  val_loss=7.0610  ❌ CAO
      ⚠️  CẢNH BÁO – MODEL KHÔNG HỌC ĐƯỢC SAU 5 EPOCH: ...
      [Epoch 10] val_acc= 62.4 % ⚠️  TRUNG BÌNH  |  val_loss=1.1200 ⚠️  CHẤP NHẬN ĐƯỢC
      [Epoch 20] val_acc= 88.1 % ✅ XUẤT SẮC      |  val_loss=0.3450 ✅ TỐT
    """

    # Sau bao nhiêu epoch không cải thiện mới in cảnh báo "không học được"
    _NOT_LEARNING_CHECK_EPOCH = 5

    def __init__(self):
        super().__init__()
        # Loss kỳ vọng khi mô hình đoán đều tất cả các lớp = ln(NUM_CLASSES)
        self._random_loss       = math.log(max(NUM_CLASSES, 1))
        self._warned_not_learning = False

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        logs = logs or {}
        # Only report when validation metrics are present (i.e. val_gen was provided).
        # Falling back to training metrics would produce misleading "val_acc" labels.
        if 'val_categorical_accuracy' not in logs:
            return
        acc  = logs['val_categorical_accuracy']
        loss = logs.get('val_loss', float('inf'))

        if acc >= ACC_EXCELLENT:
            acc_tag = "✅ XUẤT SẮC   "
        elif acc >= ACC_GOOD:
            acc_tag = "⚠️  KHÁ TỐT   "
        elif acc >= ACC_FAIR:
            acc_tag = "⚠️  TRUNG BÌNH"
        else:
            acc_tag = "❌ CHƯA ĐẠT  "

        if loss <= LOSS_GOOD:
            loss_tag = "✅ TỐT"
        elif loss <= LOSS_OK:
            loss_tag = "⚠️  CHẤP NHẬN ĐƯỢC"
        else:
            loss_tag = "❌ CAO"

        print(f"  [Epoch {epoch + 1:3d}]"
              f"  val_acc={acc * 100:6.2f} %  {acc_tag}"
              f"  |  val_loss={loss:.4f}  {loss_tag}")

        # ── Phát hiện mô hình không học được ──────────────────────────────
        # Nếu sau N epoch val_loss vẫn ≥ 90% loss ngẫu nhiên → in cảnh báo một lần.
        if (not self._warned_not_learning
                and epoch + 1 >= self._NOT_LEARNING_CHECK_EPOCH
                and loss >= self._random_loss * 0.90):
            self._warned_not_learning = True
            avg_spc = len(train_samples) / max(NUM_CLASSES, 1)
            needed  = NUM_CLASSES * RECOMMENDED_SAMPLES_PER_CLASS
            print(f"\n  ⚠️  CẢNH BÁO – MODEL KHÔNG HỌC ĐƯỢC SAU {epoch + 1} EPOCH:")
            print(f"      val_loss={loss:.4f}  ≈  loss đoán ngẫu nhiên"
                  f" ({self._random_loss:.2f} = ln({NUM_CLASSES}))")
            print(f"      Model vẫn đang đoán mò! Nguyên nhân thường gặp:")
            print(f"        1️⃣  Quá ít data: hiện ~{avg_spc:.1f} mẫu/lớp"
                  f" (cần ≥ {RECOMMENDED_SAMPLES_PER_CLASS})")
            print(f"            → Cần thêm ~{max(0, needed - len(train_samples)):,} mẫu,"
                  f" hoặc giảm xuống ≤ {len(train_samples) // RECOMMENDED_SAMPLES_PER_CLASS} lớp")
            print(f"        2️⃣  File .npy bị lỗi (NaN / toàn số 0 / sai shape)")
            print(f"        3️⃣  label_map không khớp với tên thư mục\n")


ckpt_path = os.path.join(WORKING_DIR, 'best_model.keras')
callbacks = [
    QualityMonitorCallback(),
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

