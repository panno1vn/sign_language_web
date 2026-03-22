"""
WLASL (Word-Level American Sign Language) I3D model integration.

Pre-trained source: https://github.com/dxli94/WLASL

Download pre-trained weights:
    https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view?usp=sharing
    Extract and place the .pth.tar files under: models/wlasl/
    Expected path (WLASL100): models/wlasl/nslt_100.pth.tar

Class list:
    The word list must match the order used during WLASL training.
    Obtain nslt_100.json from the WLASL repository (code/train_test_split/)
    and run scripts/build_class_list.py to regenerate models/wlasl_class_list.txt.
    A pre-built list is included in models/wlasl_class_list.txt.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── I3D Architecture ──────────────────────────────────────────────────────
# Adapted from:
# https://github.com/dxli94/WLASL/blob/master/code/I3D/pytorch_i3d.py


class MaxPool3dSamePadding(nn.MaxPool3d):
    """MaxPool3d with 'same' padding (mirrors TensorFlow behaviour)."""

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (_, _, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        x = F.pad(x, (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_t // 2, pad_t - pad_t // 2,
        ))
        return super().forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, activation_fn=F.relu,
                 use_batch_norm=True, use_bias=False, name='unit_3d'):
        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias,
        )
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (_, _, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        x = F.pad(x, (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_t // 2, pad_t - pad_t // 2,
        ))
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        self.b0 = Unit3D(in_channels, out_channels[0], kernel_shape=[1, 1, 1],
                         padding=0, name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels, out_channels[1], kernel_shape=[1, 1, 1],
                          padding=0, name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels, out_channels[3], kernel_shape=[1, 1, 1],
                          padding=0, name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels, out_channels[5], kernel_shape=[1, 1, 1],
                          padding=0, name=name + '/Branch_3/Conv3d_0b_1x1')

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture (Carreira & Zisserman, CVPR 2017).

    As used in WLASL: https://github.com/dxli94/WLASL
    """

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1',
        'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c',
        'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d',
        'Mixed_4e', 'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b',
        'Mixed_5c', 'Logits', 'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d',
                 in_channels=3, dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        self.end_points = {}

        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels, 64, [7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3),
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding([1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(64, 64, [1, 1, 1], padding=0, name=name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(64, 192, [3, 3, 3], padding=1, name=name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding([1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding([3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(480, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(512, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(512, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(512, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(528, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding([2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(832, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(832, [384, 192, 384, 48, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            self.build(); return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(1024, self._num_classes, [1, 1, 1], padding=0,
                             activation_fn=None, use_batch_norm=False,
                             use_bias=True, name='logits')
        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(1024, self._num_classes, [1, 1, 1], padding=0,
                             activation_fn=None, use_batch_norm=False,
                             use_bias=True, name='logits')

    def build(self):
        for k in self.end_points:
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x.mean(dim=2)


# ── Video Preprocessing ───────────────────────────────────────────────────

def load_rgb_frames(video_path, num_frames=64):
    """Sample *num_frames* evenly-spaced RGB frames from *video_path*.

    Returns a float32 array of shape (T, H, W, 3) normalised to [-1, 1].
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = (
        list(range(total))
        if total <= num_frames
        else np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    arr = np.array(frames, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # → [-1, 1]

    while len(arr) < num_frames:
        arr = np.concatenate([arr, arr[-1:]], axis=0)

    return arr[:num_frames]


def preprocess_video(video_path, num_frames=64):
    """Return a (1, 3, T, 224, 224) float tensor ready for I3D inference."""
    frames = load_rgb_frames(video_path, num_frames)
    if frames is None:
        return None
    tensor = torch.from_numpy(frames.transpose(3, 0, 1, 2)).unsqueeze(0)
    return tensor


# ── Model Loading ─────────────────────────────────────────────────────────

def load_wlasl_model(weights_path, num_classes):
    """Load an I3D model from a WLASL .pth.tar checkpoint."""
    model = InceptionI3d(num_classes=num_classes, in_channels=3)
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_class_list(class_list_path):
    """Return an ordered list of class label strings from a plain-text file.

    Lines beginning with ``#`` and blank lines are ignored.
    """
    with open(class_list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')]


# ── Inference ─────────────────────────────────────────────────────────────

def predict_wlasl(model, video_path, class_labels, num_frames=64, top_k=3):
    """Run I3D inference and return top-k (label, probability) pairs.

    Args:
        model: Loaded InceptionI3d model.
        video_path: Path to the input video file.
        class_labels: Ordered list of class label strings.
        num_frames: Number of frames to sample (default 64).
        top_k: Number of top predictions to return.

    Returns:
        List of (label, probability) tuples, highest probability first.
        Returns None if video cannot be processed.
    """
    tensor = preprocess_video(video_path, num_frames)
    if tensor is None:
        return None

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(class_labels[i], float(probs[i])) for i in top_indices]
