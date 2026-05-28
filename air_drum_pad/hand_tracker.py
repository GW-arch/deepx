"""손 랜드마크 추적: CPU(MediaPipe) / NPU(DX-RT + .dxnn) 공통 인터페이스."""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np

_tools_dir = str(Path(__file__).resolve().parent / "tools")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)


# --- MediaPipe-compatible view types (duck typing for strike_detector / main) ---


class _Cls:
    __slots__ = ("label", "score")

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = float(score)


class _Handedness:
    __slots__ = ("classification",)

    def __init__(self, label: str, score: float) -> None:
        self.classification = [_Cls(label, score)]


class _Lm:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _HandLms:
    __slots__ = ("landmark",)

    def __init__(self, landmarks21: Sequence[_Lm]) -> None:
        self.landmark = landmarks21


@dataclass
class HandTrackingResult:
    """MediaPipe `process` 결과와 동일 필드명."""

    multi_hand_landmarks: list[Any]
    multi_handedness: list[Any]


@runtime_checkable
class HandTracker(Protocol):
    def process(self, rgb: np.ndarray) -> HandTrackingResult: ...

    def close(self) -> None: ...


class _LandmarkSmoother:
    """Per-hand EMA (exponential moving average) landmark filter.

    Reduces INT8 quantization jitter from NPU models while preserving
    fast movements (adaptive alpha: fast motion → less smoothing).
    """

    def __init__(
        self,
        alpha: float = 0.15,
        velocity_scale: float = 8.0,
        max_alpha: float = 0.75,
    ) -> None:
        self._alpha = alpha
        self._vel_scale = velocity_scale
        self._max_alpha = max(alpha, min(1.0, max_alpha))
        # Keyed by a stable hand-side bucket → (21, 3) array
        self._prev: dict[int, np.ndarray] = {}

    def smooth(self, hand_idx: int, lm21: tuple) -> tuple:
        """Apply EMA to a tuple of 21 _Lm, return smoothed tuple."""
        cur = np.array([[l.x, l.y, l.z] for l in lm21], dtype=np.float64)
        prev = self._prev.get(hand_idx)
        if prev is None:
            self._prev[hand_idx] = cur.copy()
            return lm21

        # Adaptive per-landmark alpha: stationary points stay heavily smoothed,
        # while an actively moving fingertip remains responsive.  A single
        # global alpha lets one moving hand/finger reduce smoothing for every
        # landmark, making the supposedly still skeleton visibly wiggle.
        speed = np.linalg.norm(cur[:, :2] - prev[:, :2], axis=1)
        alpha = np.clip(
            self._alpha + speed * self._vel_scale,
            self._alpha,
            self._max_alpha,
        )

        smoothed = prev + alpha[:, None] * (cur - prev)
        self._prev[hand_idx] = smoothed.copy()
        return tuple(
            _Lm(float(smoothed[j, 0]), float(smoothed[j, 1]), float(smoothed[j, 2]))
            for j in range(smoothed.shape[0])
        )

    def clear(self) -> None:
        self._prev.clear()


class _LandmarkCorrector:
    """Apply a small learned xy correction to NPU landmarks.

    Correction JSON schema (version 1):
      {
        "type": "affine_xy",
        "labels": {
          "Right": [
            {"matrix": [[a, b, c], [d, e, f]], "n": 90},  # lm0
            ...
          ],
          "Left": [...]
        }
      }

    x' = a*x + b*y + c,  y' = d*x + e*y + f
    """

    def __init__(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Landmark correction JSON not found: {path}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("type") != "affine_xy":
            raise ValueError(f"Unsupported landmark correction type: {raw.get('type')!r}")
        labels = raw.get("labels", {})
        if not isinstance(labels, dict) or not labels:
            raise ValueError("Landmark correction JSON must contain a non-empty 'labels' object")
        self.path = str(p)
        self.metadata = raw
        self._labels: dict[str, list[np.ndarray]] = {}
        for label, transforms in labels.items():
            if not isinstance(transforms, list) or len(transforms) < 21:
                raise ValueError(f"Correction for label {label!r} must contain at least 21 transforms")
            mats: list[np.ndarray] = []
            for i, item in enumerate(transforms[:21]):
                mat = np.asarray(item.get("matrix"), dtype=np.float64)
                if mat.shape != (2, 3):
                    raise ValueError(
                        f"Correction matrix for label={label!r}, landmark={i} must be 2x3"
                    )
                mats.append(mat)
            self._labels[str(label)] = mats

    def apply(self, lms_list: list[_HandLms], handed: list[_Handedness]) -> list[_HandLms]:
        corrected: list[_HandLms] = []
        for i, hlm in enumerate(lms_list):
            label = ""
            if i < len(handed):
                label = str(handed[i].classification[0].label)
            mats = self._labels.get(label) or self._labels.get("__all__")
            if not mats:
                corrected.append(hlm)
                continue
            new_lms: list[_Lm] = []
            for j, lm in enumerate(hlm.landmark):
                if j < len(mats):
                    m = mats[j]
                    x = float(m[0, 0] * lm.x + m[0, 1] * lm.y + m[0, 2])
                    y = float(m[1, 0] * lm.x + m[1, 1] * lm.y + m[1, 2])
                    new_lms.append(_Lm(x, y, lm.z))
                else:
                    new_lms.append(_Lm(lm.x, lm.y, lm.z))
            corrected.append(_HandLms(tuple(new_lms)))
        return corrected


# --- CPU: MediaPipe ---


class MediapipeHandTracker:
    def __init__(
        self,
        *,
        max_num_hands: int,
        model_complexity: int,
    ) -> None:
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb: np.ndarray) -> HandTrackingResult:
        res = self._hands.process(rgb)
        # MediaPipe assumes mirrored/selfie input → flip labels for raw camera
        raw_handed = list(res.multi_handedness or [])
        flipped: list[Any] = []
        for h in raw_handed:
            c = h.classification[0]
            new_label = "Right" if c.label == "Left" else "Left"
            flipped.append(_Handedness(new_label, c.score))
        return HandTrackingResult(
            multi_hand_landmarks=list(res.multi_hand_landmarks or []),
            multi_handedness=flipped,
        )

    def close(self) -> None:
        self._hands.close()


# --- NPU: DX-RT (dx_engine.InferenceEngine) ---


def _default_dxnn_layout() -> dict[str, Any]:
    return {
        "input": {
            "color_order": "rgb",
            "tensor_layout": "auto",
            "tensor_name": None,
            "normalize": {"mode": "scale_255", "dtype": "float32"},
        },
        "inference": {
            "dual_horizontal_halves": False,
        },
        "outputs": {
            "landmarks_tensor_index": 0,
            "layout": "flat_xyZ",
            "coordinate_space": "normalized",
            "points_per_hand": 21,
            "max_hands": 2,
            "hand_order": "as_model",
        },
        "handedness": {"mode": "wrist_x_screen"},
        "confidence": {"tensor_index": None, "threshold": 0.25},
    }


def _load_layout(path: Optional[str]) -> dict[str, Any]:
    base = _default_dxnn_layout()
    if not path or not str(path).strip():
        return base
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    user = json.loads(p.read_text(encoding="utf-8"))
    # shallow merge top keys
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merged = {**base[k], **v}
            base[k] = merged
        else:
            base[k] = v
    return base


def _ch_dim(s: Sequence[Any], idx: int) -> int:
    v = s[idx]
    if v in (None, -1):
        return -1
    return int(v)


def _infer_nhwc(
    shape: Sequence[Any],
    override_hw: Optional[tuple[int, int]],
    inp_cfg: dict[str, Any],
) -> tuple[int, int, str]:
    """Return (H, W, 'nhwc'|'nchw') from 4D shape [N,C,H,W] or [N,H,W,C]."""
    if override_hw is not None:
        h, w = int(override_hw[0]), int(override_hw[1])
        if h <= 0 or w <= 0:
            raise ValueError("layout input height/width must be positive")
        s = list(shape)
        if len(s) != 4:
            raise ValueError(f"Expected 4D input shape, got {shape}")
        tl = str(inp_cfg.get("tensor_layout", "auto")).lower()
        if tl in ("nchw", "nhwc"):
            return h, w, tl
        a1, a3 = _ch_dim(s, 1), _ch_dim(s, 3)
        if a1 in (1, 3) and a3 not in (1, 3):
            return h, w, "nchw"
        if a3 in (1, 3):
            return h, w, "nhwc"
        raise ValueError(
            f"동적 입력 shape {shape} 에서 NCHW/NHWC를 알 수 없습니다. "
            f"dxnn_layout.json 의 input.tensor_layout 을 \"nchw\" 또는 \"nhwc\" 로 지정하세요."
        )

    s = [int(x) if x not in (None, -1) else -1 for x in shape]
    if len(s) != 4:
        raise ValueError(f"Expected 4D input shape, got {shape}")
    n, a, b, c = s
    _ = n
    if a in (1, 3) and c not in (1, 3):
        # NCHW
        _, ch, h, w = s
        _ = ch
        if h <= 0 or w <= 0:
            raise ValueError(
                f"Invalid NCHW spatial dims in {shape}; "
                f"set layout input.height / input.width when using dynamic shapes."
            )
        return h, w, "nchw"
    if c in (1, 3):
        _, h, w, ch = s
        _ = ch
        if h <= 0 or w <= 0:
            raise ValueError(
                f"Invalid NHWC spatial dims in {shape}; "
                f"set layout input.height / input.width when using dynamic shapes."
            )
        return h, w, "nhwc"
    raise ValueError(f"Cannot infer NCHW vs NHWC from shape {shape}")


def _square_pad_rgb(rgb: np.ndarray, fill: int = 0) -> np.ndarray:
    """H×W×3 → max(H,W) 정사각에 가운데 패딩."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    h, w = rgb.shape[:2]
    s = max(h, w)
    out = np.full((s, s, 3), fill, dtype=np.uint8)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = rgb
    return out


def _prepare_rgb_patch(
    rgb: np.ndarray,
    out_h: int,
    out_w: int,
    color_order: str,
    *,
    square_pad: bool = False,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    src = _square_pad_rgb(rgb) if square_pad else rgb
    patch = cv2.resize(src, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if color_order.lower() == "bgr":
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    return patch


def _to_model_tensor(
    patch_hwc_uint8: np.ndarray,
    layout: str,
    dtype: np.dtype,
    normalize: dict[str, Any],
) -> np.ndarray:
    x = patch_hwc_uint8.astype(np.float32)
    mode = str(normalize.get("mode", "scale_255"))
    if mode == "scale_255":
        x = x * (1.0 / 255.0)
    elif mode == "none":
        pass
    elif mode == "mean_std":
        mean = np.array(normalize.get("mean", [0, 0, 0]), dtype=np.float32)
        std = np.array(normalize.get("std", [1, 1, 1]), dtype=np.float32)
        x = (x - mean) / std
    else:
        raise ValueError(f"Unknown normalize.mode: {mode}")

    if layout == "nhwc":
        t = x.astype(dtype, copy=False)
        return np.expand_dims(t, axis=0)
    if layout == "nchw":
        chw = np.transpose(x, (2, 0, 1)).astype(dtype, copy=False)
        return np.expand_dims(chw, axis=0)
    raise ValueError(layout)


class DxnnHandTracker:
    """
    DX-RT로 컴파일된 손 랜드마크 .dxnn 실행.

    MediaPipe `hand_landmark_{lite,full}.tflite` → ONNX → DX-COM 한 모델은 **손 1개**만 출력합니다.
    `--max-hands 2` 이고 레이아웃 `inference.dual_horizontal_halves` 가 true이면
    화면 좌·우 절반을 각각 224 입력으로 돌려 **근사 양손**을 냅니다(손바닥 검출 모델 없음).

    레이아웃: `models/dxnn_layout.mediapipe_hand_lite.json` 참고.
    """

    def __init__(
        self,
        model_path: str,
        *,
        layout_path: Optional[str],
        max_hands: int,
    ) -> None:
        try:
            from dx_engine import InferenceEngine
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "NPU 백엔드는 DEEPX DX-RT Python 패키지(dx_engine)가 필요합니다. "
                "https://github.com/DEEPX-AI/dx_rt 의 python_package 를 빌드·설치하세요."
            ) from e

        self._layout = _load_layout(layout_path)
        self._max_hands_app = max(1, int(max_hands))
        self._per_run_max = max(
            1, min(int(self._layout["outputs"].get("max_hands", 2)), self._max_hands_app)
        )
        self._ie: Any = InferenceEngine(model_path)
        if int(self._ie.get_input_tensor_count()) != 1:
            raise RuntimeError(
                "NPU 백엔드는 입력 텐서 1개짜리 .dxnn 만 지원합니다. "
                "다중 입력은 전처리 퓨전 ONNX로 합친 뒤 컴파일하세요."
            )
        self._in_info = self._ie.get_input_tensors_info()
        if not self._in_info:
            raise RuntimeError("모델에 입력 텐서 정보가 없습니다.")
        self._shape = list(self._in_info[0]["shape"])
        self._in_dtype = self._in_info[0]["dtype"]
        inp_cfg = self._layout.get("input", {})
        oh = inp_cfg.get("height", inp_cfg.get("input_height", None))
        ow = inp_cfg.get("width", inp_cfg.get("input_width", None))
        override = None
        if oh is not None and ow is not None:
            override = (int(oh), int(ow))
        self._h, self._w, self._layout_hw = _infer_nhwc(self._shape, override, inp_cfg)
        self._dual_halves = bool(self._layout.get("inference", {}).get("dual_horizontal_halves", False))
        self._square_pad = bool(inp_cfg.get("square_pad", False))
        self._smoother = _LandmarkSmoother(alpha=0.15, velocity_scale=8.0, max_alpha=0.75)

    def _effective_input_layout(self, inp_cfg: dict[str, Any]) -> str:
        """
        DX-COM으로 float NCHW ONNX를 빌드해도 입력이 UINT8 [1,H,W,3] NHWC 로 바뀌는 경우가 많음.
        JSON에 ONNX 시절 tensor_layout: nchw 가 남아 있으면 NHWC 텐서에 CHW 순으로 넣어
        힙 손상·malloc 오류가 날 수 있어, 실제 .dxnn shape 기준으로 보정한다.
        """
        tl = str(inp_cfg.get("tensor_layout", "auto")).lower()
        sh = list(self._shape)
        if len(sh) == 4:
            a1, a3 = _ch_dim(sh, 1), _ch_dim(sh, 3)
            if a3 in (1, 3, 4) and a1 not in (1, 3, 4):
                return "nhwc"
            if a1 in (1, 3, 4) and a3 not in (1, 3, 4):
                return "nchw"
        if tl in ("nhwc", "nchw"):
            return tl
        if tl == "auto":
            return self._layout_hw
        raise ValueError(tl)

    def _build_input(self, rgb_region: np.ndarray) -> np.ndarray:
        inp_cfg = self._layout.get("input", {})
        color_order = str(inp_cfg.get("color_order", "rgb"))
        patch = _prepare_rgb_patch(
            rgb_region, self._h, self._w, color_order, square_pad=self._square_pad
        )
        layout = self._effective_input_layout(inp_cfg)

        if np.issubdtype(self._in_dtype, np.integer):
            p = patch.astype(self._in_dtype, copy=False)
            if layout == "nhwc":
                out = np.expand_dims(p, axis=0)
            else:
                chw = np.transpose(p, (2, 0, 1)).astype(self._in_dtype, copy=False)
                out = np.expand_dims(chw, axis=0)
            return np.ascontiguousarray(out)

        normalize = inp_cfg.get("normalize", {"mode": "scale_255", "dtype": "float32"})
        want_dt = str(normalize.get("dtype", "float32"))
        np_dt = np.dtype(want_dt)
        inp = _to_model_tensor(
            patch,
            layout,
            np_dt,
            normalize if isinstance(normalize, dict) else {},
        )
        if inp.dtype != self._in_dtype:
            inp = inp.astype(self._in_dtype, copy=False)
        return np.ascontiguousarray(inp)

    def process(self, rgb: np.ndarray) -> HandTrackingResult:
        if self._dual_halves and self._max_hands_app >= 2:
            h, w = rgb.shape[:2]
            left = rgb[:, : w // 2]
            right = rgb[:, w // 2 :]
            parts: list[tuple[str, HandTrackingResult]] = []
            parts.append(("Right", self._infer_region(left, x_scale=0.5, x_bias=0.0)))
            parts.append(("Left", self._infer_region(right, x_scale=0.5, x_bias=0.5)))
            lms: list[_HandLms] = []
            hnd: list[_Handedness] = []
            for lab, sub in parts:
                if not sub.multi_hand_landmarks:
                    continue
                lms.append(sub.multi_hand_landmarks[0])
                sc = float(sub.multi_handedness[0].classification[0].score)
                hnd.append(_Handedness(lab, sc))
            # Apply EMA smoothing to reduce NPU INT8 jitter
            for i, hlm in enumerate(lms):
                smoothed = self._smoother.smooth(i, hlm.landmark)
                lms[i] = _HandLms(smoothed)
            return HandTrackingResult(lms, hnd)
        result = self._infer_region(rgb, x_scale=1.0, x_bias=0.0)
        # Apply EMA smoothing for single-region path
        for i, hlm in enumerate(result.multi_hand_landmarks):
            smoothed = self._smoother.smooth(i, hlm.landmark)
            result.multi_hand_landmarks[i] = _HandLms(smoothed)
        return result

    def _infer_region(
        self,
        rgb: np.ndarray,
        *,
        x_scale: float,
        x_bias: float,
    ) -> HandTrackingResult:
        inp = self._build_input(rgb)
        outs: List[np.ndarray] = self._ie.run([inp])
        return self._postprocess(
            outs,
            x_scale=x_scale,
            x_bias=x_bias,
            region_hw=(int(rgb.shape[0]), int(rgb.shape[1])),
        )

    def _landmarks_flat_to_norm(
        self,
        flat: np.ndarray,
        *,
        region_hw: tuple[int, int],
        ocfg: dict[str, Any],
    ) -> np.ndarray:
        """
        모델이 (x,y,z)를 224 패치 픽셀처럼보내는 경우, square_pad + resize 역변환으로
        원본 region RGB (H,W) 기준 MediaPipe식 정규화 좌표로 맞춘다.
        """
        co = str(ocfg.get("coordinate_space", "normalized")).lower()
        if co in ("normalized", "norm", "mp"):
            return flat
        if co not in ("letterbox_patch_pixels", "patch_pixels", "input_pixels"):
            raise ValueError(
                f"Unknown outputs.coordinate_space: {co!r} "
                f"(use normalized | letterbox_patch_pixels)"
            )
        ih, iw = int(region_hw[0]), int(region_hw[1])
        mh, mw = int(self._h), int(self._w)
        pph = int(ocfg.get("points_per_hand", 21))
        need = pph * 3
        work = np.asarray(flat[:need], dtype=np.float32).copy()
        if self._square_pad:
            s_side = max(ih, iw)
            x0 = (s_side - iw) // 2
            y0 = (s_side - ih) // 2
            inv_w = 1.0 / float(max(mw, 1))
            inv_h = 1.0 / float(max(mh, 1))
            inv_iw = 1.0 / float(max(iw, 1))
            inv_ih = 1.0 / float(max(ih, 1))
            for j in range(pph):
                lx, ly, lz = float(work[j * 3]), float(work[j * 3 + 1]), float(work[j * 3 + 2])
                px = lx * inv_w * float(s_side)
                py = ly * inv_h * float(s_side)
                ox = px - float(x0)
                oy = py - float(y0)
                work[j * 3] = ox * inv_iw
                work[j * 3 + 1] = oy * inv_ih
                work[j * 3 + 2] = lz * inv_h
        else:
            inv_w = 1.0 / float(max(mw, 1))
            inv_h = 1.0 / float(max(mh, 1))
            inv_iw = 1.0 / float(max(iw, 1))
            inv_ih = 1.0 / float(max(ih, 1))
            for j in range(pph):
                lx, ly, lz = float(work[j * 3]), float(work[j * 3 + 1]), float(work[j * 3 + 2])
                work[j * 3] = (lx * inv_w * float(iw)) * inv_iw
                work[j * 3 + 1] = (ly * inv_h * float(ih)) * inv_ih
                work[j * 3 + 2] = lz * inv_h
        return work

    def _postprocess(
        self,
        outs: List[np.ndarray],
        *,
        x_scale: float = 1.0,
        x_bias: float = 0.0,
        region_hw: tuple[int, int] = (1, 1),
    ) -> HandTrackingResult:
        ocfg = self._layout.get("outputs", {})
        li = int(ocfg.get("landmarks_tensor_index", 0))
        if li < 0 or li >= len(outs):
            raise IndexError(f"landmarks_tensor_index {li} out of range, n_out={len(outs)}")
        flat_raw = np.asarray(outs[li]).astype(np.float32).reshape(-1)
        pph = int(ocfg.get("points_per_hand", 21))
        need = pph * 3
        flat = self._landmarks_flat_to_norm(flat_raw, region_hw=region_hw, ocfg=ocfg)
        if flat.size % need != 0:
            raise ValueError(
                f"출력 길이 {flat.size} 가 손당 {need} 배수가 아님 — dxnn_layout.json 에서 "
                f"points_per_hand / landmarks_tensor_index 를 모델에 맞게 수정하세요."
            )
        nh_all = flat.size // need
        nh = min(nh_all, self._per_run_max)

        conf_cfg = self._layout.get("confidence", {})
        ct = conf_cfg.get("tensor_index", None)
        thr = float(conf_cfg.get("threshold", 0.25))
        scores: Optional[np.ndarray] = None
        if ct is not None:
            ci = int(ct)
            if 0 <= ci < len(outs):
                scores = np.asarray(outs[ci]).astype(np.float32).reshape(-1)

        hands_work: list[tuple[int, _HandLms]] = []
        for hi in range(nh):
            sc = (
                float(scores[hi])
                if scores is not None and hi < scores.size
                else 1.0
            )
            if scores is not None and sc < thr:
                continue
            chunk = flat[hi * need : (hi + 1) * need]
            lm21 = tuple(
                _Lm(
                    float(chunk[j * 3]) * x_scale + x_bias,
                    float(chunk[j * 3 + 1]),
                    float(chunk[j * 3 + 2]),
                )
                for j in range(pph)
            )
            hands_work.append((hi, _HandLms(lm21)))

        handed_mode = str(self._layout.get("handedness", {}).get("mode", "wrist_x_screen"))
        lms_list: list[_HandLms] = []
        handed: list[_Handedness] = []

        if not hands_work:
            return HandTrackingResult([], [])

        if handed_mode == "wrist_x_screen":
            hands_work.sort(key=lambda t: t[1].landmark[0].x)
            for si, (orig_hi, hlm) in enumerate(hands_work):
                if len(hands_work) == 1:
                    x0 = hlm.landmark[0].x
                    lab = "Right" if x0 < 0.5 else "Left"
                else:
                    lab = "Right" if si == 0 else "Left"
                sc = (
                    float(scores[orig_hi])
                    if scores is not None and orig_hi < scores.size
                    else 0.99
                )
                lms_list.append(hlm)
                handed.append(_Handedness(lab, sc))
        else:
            for orig_hi, hlm in hands_work:
                x0 = hlm.landmark[0].x
                lab = "Right" if x0 < 0.5 else "Left"
                sc = (
                    float(scores[orig_hi])
                    if scores is not None and orig_hi < scores.size
                    else 0.9
                )
                lms_list.append(hlm)
                handed.append(_Handedness(lab, sc))

        return HandTrackingResult(multi_hand_landmarks=lms_list, multi_handedness=handed)

    def close(self) -> None:
        if hasattr(self, "_ie") and self._ie is not None:
            self._ie.dispose()
            self._ie = None


# --- CPU TFLite hand landmark (same interface as DxnnHandTracker) ---


class TFLiteHandLandmark:
    """Run hand_landmark_lite.tflite on CPU via TFLite runtime.

    Accepts a 224×224 RGB patch, returns HandTrackingResult with 21 landmarks
    in normalised [0,1] coordinates (same contract as DxnnHandTracker).
    """

    def __init__(self, model_path: str, *, max_hands: int = 1) -> None:
        try:
            import tensorflow as tf
            _Interpreter = tf.lite.Interpreter
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                _Interpreter = tflite.Interpreter
            except ImportError:
                raise ImportError(
                    "TFLiteHandLandmark requires tensorflow or tflite_runtime."
                )
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"Hand landmark TFLite not found: {model_path}")
        self._interp = _Interpreter(model_path=str(p))
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._outs = self._interp.get_output_details()
        self._max_hands = max(1, int(max_hands))

    def process(self, rgb: np.ndarray) -> HandTrackingResult:
        """Run inference on a 224×224×3 RGB patch (uint8 or float).

        Returns landmarks in normalised [0,1] coordinates within the patch.
        """
        h, w = rgb.shape[:2]
        patch = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR) if (h, w) != (224, 224) else rgb
        inp_dtype = self._inp["dtype"]
        if inp_dtype == np.float32:
            tensor = patch.astype(np.float32) / 255.0
        else:
            tensor = patch.astype(inp_dtype)
        tensor = np.expand_dims(tensor, axis=0)
        self._interp.set_tensor(self._inp["index"], tensor)
        self._interp.invoke()

        # Output[0] = Identity [1,63] = 21 × (x,y,z) in patch pixel coords
        lm_raw = self._interp.get_tensor(self._outs[0]["index"]).astype(np.float32).flatten()
        # Output[1] = Identity_1 [1,1] = hand presence score (hand_flag)
        hand_flag = float(self._interp.get_tensor(self._outs[1]["index"]).flatten()[0])

        if hand_flag < 0.5:
            return HandTrackingResult([], [])

        # Convert pixel coords → normalised [0,1]
        lm21 = []
        for j in range(21):
            x = float(lm_raw[j * 3 + 0]) / 224.0
            y = float(lm_raw[j * 3 + 1]) / 224.0
            z = float(lm_raw[j * 3 + 2]) / 224.0
            lm21.append(_Lm(x, y, z))
        hlm = _HandLms(tuple(lm21))
        wx = hlm.landmark[0].x
        lab = "Right" if wx < 0.5 else "Left"
        return HandTrackingResult([hlm], [_Handedness(lab, hand_flag)])

    def close(self) -> None:
        self._interp = None


# --- Full pipeline: Palm detection + Hand landmark (CPU TFLite + optional NPU) ---


class FullNpuHandsTracker:
    """Palm detection → ROI warp → Hand landmark.

    Palm 은 .dxnn (NPU) 또는 TFLite (CPU) 중 하나로 실행.
    Hand landmark 는 DxnnHandTracker (.dxnn NPU) 또는 TFLiteHandLandmark (.tflite CPU).
    """

    def __init__(
        self,
        *,
        palm_tflite_path: Optional[str] = None,
        palm_dxnn_path: Optional[str] = None,
        hand_dxnn_path: Optional[str] = None,
        hand_tflite_path: Optional[str] = None,
        hand_layout_path: Optional[str],
        max_hands: int,
        palm_score_thresh: float = 0.5,
        palm_redetect_every: int = 0,
        async_palm: bool = False,
        landmark_correction_path: Optional[str] = None,
    ) -> None:
        from palm_decode import generate_ssd_anchors
        from palm_letterbox import rgb_uint8_to_palm_input_tensor  # noqa: F811

        self._max_hands = max(1, int(max_hands))
        self._palm_backend: str  # "dxnn" | "tflite"
        # .dxnn 양자화 → score 저하 보정: TFLite 기본 0.5, .dxnn 기본 0.3
        self._palm_score_thresh: float

        # --- Palm model init ---
        if palm_dxnn_path:
            # NPU palm via dx_engine
            p = Path(palm_dxnn_path)
            if not p.is_file():
                raise FileNotFoundError(f"Palm .dxnn not found: {palm_dxnn_path}")
            from dx_engine import InferenceEngine
            self._palm_ie: Any = InferenceEngine(str(p))
            self._palm_backend = "dxnn"
            self._palm_score_thresh = palm_score_thresh if palm_score_thresh != 0.5 else 0.3
            self._palm_intr = None
        elif palm_tflite_path:
            # CPU palm via TFLite
            p = Path(palm_tflite_path)
            if not p.is_file():
                raise FileNotFoundError(f"Palm TFLite not found: {palm_tflite_path}")
            try:
                import tensorflow as tf
                _Interpreter = tf.lite.Interpreter
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite
                    _Interpreter = tflite.Interpreter
                except ImportError:
                    raise ImportError(
                        "FullNpuHandsTracker 의 palm detection (TFLite) 에 tensorflow 또는 "
                        "tflite_runtime 이 필요합니다."
                    )
            self._palm_intr = _Interpreter(model_path=str(p))
            self._palm_intr.allocate_tensors()
            self._palm_inp = self._palm_intr.get_input_details()[0]
            self._palm_outs = self._palm_intr.get_output_details()
            self._palm_backend = "tflite"
            self._palm_score_thresh = palm_score_thresh
            self._palm_ie = None
        else:
            raise ValueError("palm_dxnn_path 또는 palm_tflite_path 중 하나를 지정하세요.")

        self._anchors = generate_ssd_anchors()

        # Hand landmark model: prefer TFLite (CPU) if given, else .dxnn (NPU)
        if hand_tflite_path:
            self._hand_tracker = TFLiteHandLandmark(
                hand_tflite_path,
                max_hands=1,
            )
        elif hand_dxnn_path:
            self._hand_tracker = DxnnHandTracker(
                hand_dxnn_path,
                layout_path=hand_layout_path,
                max_hands=1,  # one hand per ROI
            )
        else:
            raise ValueError("hand_dxnn_path 또는 hand_tflite_path 중 하나를 지정하세요.")

        # Stash callables for runtime (avoid repeated imports)
        self._rgb_to_palm = rgb_uint8_to_palm_input_tensor
        self._decode = None  # lazy import
        self._roi_fn = None

        # --- Tracking state: skip palm when previous landmarks are good ---
        # Each entry: (center_x_px, center_y_px, roi_size_px, rotation_rad)
        self._prev_rois: list[tuple[float, float, float, float]] = []
        self._palm_skip_count: int = 0
        self._PALM_REDETECT_EVERY: int = max(0, int(palm_redetect_every))
        self._async_palm = bool(async_palm)
        self._palm_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="air-drum-palm")
            if self._async_palm
            else None
        )
        self._palm_future: Future[tuple[np.ndarray, float]] | None = None
        # Public per-frame timing snapshot for benchmark tools.
        self.last_profile: dict[str, Any] = {}
        self._smoother = _LandmarkSmoother(alpha=0.18, velocity_scale=8.0, max_alpha=0.75)
        # Fixed demo prior: two hands stay on left/right screen sides and face
        # roughly the same direction.  This is not a retrained palm model; it is
        # a post-detection stabilizer that prevents one side from stealing a
        # duplicate detection and damps palm-ROI rotation/center jitter.
        self._fixed_two_hand_prior = self._max_hands >= 2
        self._stable_palm_rois: dict[int, tuple[float, float, float, float]] = {}
        self._landmark_corrector = (
            _LandmarkCorrector(landmark_correction_path)
            if landmark_correction_path and str(landmark_correction_path).strip()
            else None
        )

    def _ensure_imports(self) -> None:
        if self._decode is None:
            from palm_decode import decode_palm_tensors
            from palm_roi import (
                extract_hand_roi,
                inverse_landmark_transform,
                palm_detection_to_roi,
                warp_roi_affine,
            )
            self._decode = decode_palm_tensors
            self._extract_roi = extract_hand_roi
            self._inv_lm = inverse_landmark_transform
            self._palm_det_to_roi = palm_detection_to_roi
            self._warp_roi = warp_roi_affine

    def _run_palm(self, rgb: np.ndarray):
        """Run palm detection, return (K, 19) detections."""
        tensor, meta = self._rgb_to_palm(rgb)

        if self._palm_backend == "dxnn":
            # DX-COM compiled to NHWC uint8 [0,255] — convert from [0,1] float tensor
            t_u8 = (tensor[0] * 255.0).clip(0, 255).astype(np.uint8)  # (192,192,3)
            t_u8 = np.expand_dims(t_u8, axis=0)  # (1,192,192,3)
            raw_outputs: list[np.ndarray] = self._palm_ie.run([t_u8])
        else:
            # TFLite path
            inp_dtype = self._palm_inp["dtype"]
            if inp_dtype in (np.float32, "float32"):
                tensor = tensor.astype(np.float32)
            elif inp_dtype in (np.uint8, "uint8"):
                tensor = (tensor * 255.0).clip(0, 255).astype(np.uint8)
            self._palm_intr.set_tensor(self._palm_inp["index"], tensor)
            self._palm_intr.invoke()
            raw_outputs = [self._palm_intr.get_tensor(o["index"]) for o in self._palm_outs]

        dets = self._decode(
            raw_outputs, self._anchors,
            letterbox_meta=meta,
            score_thresh=self._palm_score_thresh,
        )
        return dets

    def _run_palm_timed(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        dets = self._run_palm(rgb)
        return dets, (time.perf_counter() - t0) * 1000.0

    def _select_palm_detections(self, dets: np.ndarray) -> np.ndarray:
        """Apply a fixed left/right two-hand prior to palm detections.

        For the PANDA demo posture, hands are expected to remain one per screen
        side.  Keeping only the best detection on each side reduces duplicate
        boxes on the moving hand from being interpreted as the stationary hand.
        """
        if dets.shape[0] == 0:
            return dets
        if not self._fixed_two_hand_prior:
            order = np.argsort(-dets[:, 0])
            return dets[order[: self._max_hands]]

        from palm_decode import DET_XMAX_IDX, DET_XMIN_IDX

        selected: list[np.ndarray] = []
        used: set[int] = set()
        for side in (0, 1):
            side_candidates: list[tuple[float, int]] = []
            for idx, det in enumerate(dets):
                cx_norm = (float(det[DET_XMIN_IDX]) + float(det[DET_XMAX_IDX])) * 0.5
                if (cx_norm < 0.5) == (side == 0):
                    side_candidates.append((float(det[0]), idx))
            if side_candidates:
                _score, best_idx = max(side_candidates, key=lambda item: item[0])
                selected.append(dets[best_idx])
                used.add(best_idx)

        # If only one side is visible, return only that side.  Filling the
        # missing side with a second same-side detection is a common source of
        # ghost skeletons and cross-hand strikes.
        if selected:
            selected.sort(key=lambda det: (float(det[DET_XMIN_IDX]) + float(det[DET_XMAX_IDX])) * 0.5)
            return np.stack(selected, axis=0)

        order = np.argsort(-dets[:, 0])
        return dets[order[:1]]

    def _stabilize_palm_roi(
        self,
        roi: tuple[float, float, float, float],
        side_key: int,
    ) -> tuple[float, float, float, float]:
        """Temporally smooth palm-derived ROI center/size/rotation."""
        prev = self._stable_palm_rois.get(side_key)
        if prev is None:
            self._stable_palm_rois[side_key] = roi
            return roi

        alpha = 0.25
        cx = prev[0] + alpha * (roi[0] - prev[0])
        cy = prev[1] + alpha * (roi[1] - prev[1])
        sz = prev[2] + alpha * (roi[2] - prev[2])
        # Circular interpolation for angle; avoids jumps around +/-pi.
        dtheta = math.atan2(math.sin(roi[3] - prev[3]), math.cos(roi[3] - prev[3]))
        rot = prev[3] + alpha * dtheta
        out = (cx, cy, sz, rot)
        self._stable_palm_rois[side_key] = out
        return out

    def _schedule_async_palm(self, rgb: np.ndarray) -> None:
        """Start one background palm pass if async mode is enabled and idle."""
        if not self._async_palm or self._palm_executor is None:
            return
        # If a completed future is waiting, keep it for the next process() call
        # to consume; replacing it here would drop a fresh palm result.
        if self._palm_future is not None:
            return
        # Copy the frame because the caller owns the input buffer.
        self._palm_future = self._palm_executor.submit(self._run_palm_timed, rgb.copy())

    def _consume_async_palm(
        self,
        profile: dict[str, Any],
        *,
        wait: bool = False,
    ) -> np.ndarray | None:
        """Return finished async palm detections, optionally waiting for them."""
        fut = self._palm_future
        if fut is None:
            return None
        if not wait and not fut.done():
            return None
        t_wait0 = time.perf_counter()
        dets, palm_ms = fut.result()
        wait_ms = (time.perf_counter() - t_wait0) * 1000.0
        self._palm_future = None
        profile["async_palm_ms"] += palm_ms
        profile["palm_wait_ms"] += wait_ms
        profile["num_detections"] = int(dets.shape[0])
        return dets

    def _roi_from_landmarks(
        self, lm21: tuple, iw: int, ih: int,
    ) -> tuple[float, float, float, float]:
        """Derive a tracking ROI from 21 landmarks (like MediaPipe's hand_recrop).

        Uses wrist(0) → middle_finger_mcp(9) for rotation,
        bounding box of all points for size, with MediaPipe-style expansion.
        """
        xs = [lm21[i].x * iw for i in range(21)]
        ys = [lm21[i].y * ih for i in range(21)]
        # Rotation from wrist to middle finger MCP, matching MediaPipe's
        # target-angle convention for palm/hand ROIs.
        wx, wy = lm21[0].x * iw, lm21[0].y * ih
        mx, my = lm21[9].x * iw, lm21[9].y * ih
        rotation = math.pi / 2.0 - math.atan2(-(my - wy), mx - wx)

        # Bounding box of all landmarks
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        box_w = xmax - xmin
        box_h = ymax - ymin
        long_side = max(box_w, box_h)

        # Expand like MediaPipe (2.0x bounding box)
        roi_size = long_side * 2.0
        # Shift center slightly towards fingers (up along hand axis)
        shift_y = -0.1
        cx -= box_h * shift_y * math.sin(rotation)
        cy += box_h * shift_y * math.cos(rotation)
        return cx, cy, roi_size, rotation

    def _run_hand_from_roi(
        self, rgb: np.ndarray, roi: tuple[float, float, float, float],
        iw: int, ih: int,
    ) -> tuple[_HandLms | None, tuple[float, float, float, float] | None]:
        """Run hand landmark on a single ROI. Returns (landmarks, new_roi) or (None, None)."""
        cx, cy, sz, rot = roi
        patch = self._warp_roi(rgb, cx, cy, sz, rot, out_size=224)
        result = self._hand_tracker.process(patch)
        if not result.multi_hand_landmarks:
            return None, None
        hlm = result.multi_hand_landmarks[0]
        lm_flat = np.array(
            [[l.x, l.y, l.z] for l in hlm.landmark], dtype=np.float32,
        )
        lm_orig = self._inv_lm(lm_flat, cx, cy, sz, rot, iw, ih)
        lm21 = tuple(
            _Lm(float(lm_orig[j, 0]), float(lm_orig[j, 1]), float(lm_orig[j, 2]))
            for j in range(lm_orig.shape[0])
        )
        new_roi = self._roi_from_landmarks(lm21, iw, ih)
        return _HandLms(lm21), new_roi

    def _run_hands_from_detections(
        self,
        rgb: np.ndarray,
        dets: np.ndarray,
        iw: int,
        ih: int,
        profile: dict[str, Any],
    ) -> tuple[list[_HandLms], list[tuple[float, float, float, float]]]:
        """Run hand landmark on top scored palm detections."""
        if dets.shape[0] == 0:
            return [], []

        dets = self._select_palm_detections(dets)

        lms_list: list[_HandLms] = []
        new_rois: list[tuple[float, float, float, float]] = []
        for det in dets:
            t_hand0 = time.perf_counter()
            cx, cy, sz, rot = self._palm_det_to_roi(det, iw, ih, target_size=224)
            side_key = 0 if cx < iw * 0.5 else 1
            cx, cy, sz, rot = self._stabilize_palm_roi((cx, cy, sz, rot), side_key)
            patch = self._warp_roi(rgb, cx, cy, sz, rot, out_size=224)
            result = self._hand_tracker.process(patch)
            if not result.multi_hand_landmarks:
                profile["hand_ms"] += (time.perf_counter() - t_hand0) * 1000.0
                continue
            hlm = result.multi_hand_landmarks[0]
            lm_flat = np.array(
                [[l.x, l.y, l.z] for l in hlm.landmark], dtype=np.float32,
            )
            lm_orig = self._inv_lm(lm_flat, cx, cy, sz, rot, iw, ih)
            lm21 = tuple(
                _Lm(float(lm_orig[j, 0]), float(lm_orig[j, 1]), float(lm_orig[j, 2]))
                for j in range(lm_orig.shape[0])
            )
            lms_list.append(_HandLms(lm21))
            new_rois.append(self._roi_from_landmarks(lm21, iw, ih))
            profile["hand_ms"] += (time.perf_counter() - t_hand0) * 1000.0
        return lms_list, new_rois

    def _profiled_result(
        self,
        result: HandTrackingResult,
        profile: dict[str, Any],
        t0: float,
        rgb_for_async: np.ndarray | None = None,
    ) -> HandTrackingResult:
        if rgb_for_async is not None and result.multi_hand_landmarks:
            self._schedule_async_palm(rgb_for_async)
        profile["total_ms"] = (time.perf_counter() - t0) * 1000.0
        profile["num_hands"] = len(result.multi_hand_landmarks or [])
        profile["async_pending"] = (
            self._palm_future is not None and not self._palm_future.done()
        )
        self.last_profile = profile
        return result

    def process(self, rgb: np.ndarray) -> HandTrackingResult:
        self._ensure_imports()
        t_total0 = time.perf_counter()
        profile: dict[str, Any] = {
            "mode": "palm",
            "palm_ms": 0.0,
            "hand_ms": 0.0,
            "total_ms": 0.0,
            "async_palm_ms": 0.0,
            "palm_wait_ms": 0.0,
            "num_detections": 0,
            "num_hands": 0,
            "palm_redetect_every": self._PALM_REDETECT_EVERY,
            "async_palm": self._async_palm,
            "async_pending": False,
            "landmark_correction": self._landmark_corrector is not None,
        }
        ih, iw = rgb.shape[:2]

        # --- Consume a finished background palm pass and refresh ROIs on current frame. ---
        async_dets = self._consume_async_palm(profile, wait=False)
        if async_dets is not None:
            profile["mode"] = "async_palm"
            if async_dets.shape[0] > 0:
                lms_list, new_rois = self._run_hands_from_detections(
                    rgb, async_dets, iw, ih, profile
                )
                if lms_list:
                    self._prev_rois = new_rois
                    self._palm_skip_count = 0
                    return self._profiled_result(
                        self._finalize(lms_list),
                        profile,
                        t_total0,
                        rgb_for_async=rgb,
                    )
            elif not self._prev_rois:
                return self._profiled_result(HandTrackingResult([], []), profile, t_total0)

        # --- Try tracking from previous ROIs (skip palm) ---
        use_tracking = (
            len(self._prev_rois) > 0
            and (
                self._async_palm
                or self._palm_skip_count < self._PALM_REDETECT_EVERY
            )
        )

        if use_tracking:
            profile["mode"] = "tracking"
            lms_list: list[_HandLms] = []
            new_rois: list[tuple[float, float, float, float]] = []
            for roi in self._prev_rois:
                t_hand0 = time.perf_counter()
                hlm, nroi = self._run_hand_from_roi(rgb, roi, iw, ih)
                profile["hand_ms"] += (time.perf_counter() - t_hand0) * 1000.0
                if hlm is not None:
                    lms_list.append(hlm)
                    new_rois.append(nroi)

            if lms_list:
                self._prev_rois = new_rois
                self._palm_skip_count += 1
                return self._profiled_result(
                    self._finalize(lms_list),
                    profile,
                    t_total0,
                    rgb_for_async=rgb,
                )

            # Tracking lost — fall through to palm detection
            self._prev_rois.clear()

        # --- Full palm detection ---
        profile["mode"] = "async_wait" if self._async_palm and self._palm_future else "palm"
        self._palm_skip_count = 0
        if self._async_palm and self._palm_future is not None:
            dets = self._consume_async_palm(profile, wait=True)
            if dets is None:  # defensive; wait=True should always produce a result.
                dets = np.empty((0, 19), dtype=np.float32)
        else:
            t_palm0 = time.perf_counter()
            dets = self._run_palm(rgb)
            profile["palm_ms"] = (time.perf_counter() - t_palm0) * 1000.0
        profile["num_detections"] = int(dets.shape[0])
        if dets.shape[0] == 0:
            self._prev_rois.clear()
            return self._profiled_result(HandTrackingResult([], []), profile, t_total0)

        lms_list, new_rois = self._run_hands_from_detections(rgb, dets, iw, ih, profile)

        self._prev_rois = new_rois
        return self._profiled_result(
            self._finalize(lms_list),
            profile,
            t_total0,
            rgb_for_async=rgb,
        )

    def _finalize(self, lms_list: list[_HandLms]) -> HandTrackingResult:
        """Assign handedness, sort left-to-right, apply EMA smoothing."""
        handed: list[_Handedness] = []
        for hlm in lms_list:
            wx = hlm.landmark[0].x
            lab = "Right" if wx < 0.5 else "Left"
            handed.append(_Handedness(lab, 1.0))

        if len(lms_list) >= 2:
            pairs = list(zip(lms_list, handed))
            pairs.sort(key=lambda p: p[0].landmark[0].x)
            lms_list = [p[0] for p in pairs]
            handed = [p[1] for p in pairs]
            if len(pairs) >= 2:
                handed[0] = _Handedness("Right", handed[0].classification[0].score)
                handed[1] = _Handedness("Left", handed[1].classification[0].score)

        # Apply EMA smoothing to reduce NPU INT8 jitter.  Key the smoother by
        # screen-side instead of list index so a temporarily single detected
        # right hand does not inherit the previous left-hand state (or vice
        # versa), which can create large false velocities and ghost strikes.
        used_smoother_keys: set[int] = set()
        for i, hlm in enumerate(lms_list):
            side_key = 0 if hlm.landmark[0].x < 0.5 else 1
            if side_key in used_smoother_keys:
                side_key = 10 + i
            used_smoother_keys.add(side_key)
            smoothed = self._smoother.smooth(side_key, hlm.landmark)
            lms_list[i] = _HandLms(smoothed)

        if self._landmark_corrector is not None:
            lms_list = self._landmark_corrector.apply(lms_list, handed)

        return HandTrackingResult(lms_list, handed)

    def close(self) -> None:
        if self._palm_executor is not None:
            self._palm_executor.shutdown(wait=True, cancel_futures=False)
            self._palm_executor = None
            self._palm_future = None
        self._hand_tracker.close()
        if self._palm_ie is not None:
            self._palm_ie.dispose()
            self._palm_ie = None
        self._palm_intr = None


def create_tracker(
    backend: str,
    *,
    max_hands: int,
    model_complexity: int,
    dxnn_path: str,
    dxnn_layout: Optional[str],
    palm_tflite: Optional[str] = None,
    palm_dxnn: Optional[str] = None,
    hand_tflite: Optional[str] = None,
    palm_redetect_every: int = 0,
    async_palm: bool = False,
    landmark_correction: Optional[str] = None,
) -> HandTracker:
    b = backend.strip().lower()
    if b == "cpu":
        return MediapipeHandTracker(
            max_num_hands=max_hands,
            model_complexity=model_complexity,
        )
    if b == "npu":
        if not dxnn_path or not dxnn_path.strip():
            raise SystemExit("NPU 백엔드는 --dxnn 경로가 필요합니다.")
        return DxnnHandTracker(
            dxnn_path.strip(),
            layout_path=dxnn_layout,
            max_hands=max_hands,
        )

    if b in ("npu-full", "npu_full", "cpu-baseline", "cpu_baseline"):
        # npu-full: palm TFLite(CPU) + hand .dxnn(NPU)
        # cpu-baseline: palm TFLite(CPU) + hand TFLite(CPU)

        # --- Resolve palm model ---
        resolved_palm_dxnn: Optional[str] = None
        resolved_palm_tflite: Optional[str] = None

        if b in ("cpu-baseline", "cpu_baseline"):
            # cpu-baseline always uses TFLite palm
            if palm_tflite and palm_tflite.strip():
                resolved_palm_tflite = palm_tflite.strip()
            else:
                default_tflite = Path(__file__).resolve().parent / "models" / "vendor" / "palm_detection_lite.tflite"
                if default_tflite.is_file():
                    resolved_palm_tflite = str(default_tflite)
                else:
                    raise SystemExit(
                        "cpu-baseline 백엔드: --palm-tflite 을 지정하거나 "
                        f"{default_tflite} 을 배치하세요."
                    )
        elif palm_dxnn and palm_dxnn.strip():
            resolved_palm_dxnn = palm_dxnn.strip()
        elif palm_tflite and palm_tflite.strip():
            resolved_palm_tflite = palm_tflite.strip()
        else:
            # Auto-detect: prefer TFLite (CPU, float32).  Palm .dxnn is kept
            # only for explicit experiments because INT8 quantization breaks
            # the score head on the current model.
            base = Path(__file__).resolve().parent / "models" / "vendor"
            default_tflite = base / "palm_detection_lite.tflite"
            if default_tflite.is_file():
                resolved_palm_tflite = str(default_tflite)
            else:
                raise SystemExit(
                    "npu-full 백엔드: --palm-tflite 을 지정하거나 "
                    f"{default_tflite} 을 배치하세요. "
                    "Palm .dxnn은 양자화 품질 문제로 자동 선택하지 않습니다 "
                    "(실험 시 --palm-dxnn 명시)."
                )

        # --- Resolve hand landmark model ---
        resolved_hand_dxnn: Optional[str] = None
        resolved_hand_tflite: Optional[str] = None

        if b in ("cpu-baseline", "cpu_baseline"):
            # cpu-baseline always uses TFLite hand
            if hand_tflite and hand_tflite.strip():
                resolved_hand_tflite = hand_tflite.strip()
            else:
                default_hand = Path(__file__).resolve().parent / "models" / "vendor" / "hand_landmark_lite.tflite"
                if default_hand.is_file():
                    resolved_hand_tflite = str(default_hand)
                else:
                    raise SystemExit(
                        "cpu-baseline 백엔드: --hand-tflite 을 지정하거나 "
                        f"{default_hand} 을 배치하세요."
                    )
        else:
            if not dxnn_path or not dxnn_path.strip():
                raise SystemExit("npu-full 백엔드는 --dxnn (hand landmark) 경로가 필요합니다.")
            resolved_hand_dxnn = dxnn_path.strip()

        return FullNpuHandsTracker(
            palm_dxnn_path=resolved_palm_dxnn,
            palm_tflite_path=resolved_palm_tflite,
            hand_dxnn_path=resolved_hand_dxnn,
            hand_tflite_path=resolved_hand_tflite,
            hand_layout_path=dxnn_layout,
            max_hands=max_hands,
            palm_redetect_every=palm_redetect_every,
            async_palm=async_palm,
            landmark_correction_path=(
                landmark_correction
                if b not in ("cpu-baseline", "cpu_baseline")
                else None
            ),
        )
    raise SystemExit(f"알 수 없는 --backend: {backend} (cpu | cpu-baseline | npu | npu-full)")
