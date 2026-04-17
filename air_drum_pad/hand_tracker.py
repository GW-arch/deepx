"""손 랜드마크 추적: CPU(MediaPipe) / NPU(DX-RT + .dxnn) 공통 인터페이스."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np


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
        return HandTrackingResult(
            multi_hand_landmarks=list(res.multi_hand_landmarks or []),
            multi_handedness=list(res.multi_handedness or []),
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
            parts.append(("Left", self._infer_region(left, x_scale=0.5, x_bias=0.0)))
            parts.append(("Right", self._infer_region(right, x_scale=0.5, x_bias=0.5)))
            lms: list[_HandLms] = []
            hnd: list[_Handedness] = []
            for lab, sub in parts:
                if not sub.multi_hand_landmarks:
                    continue
                lms.append(sub.multi_hand_landmarks[0])
                sc = float(sub.multi_handedness[0].classification[0].score)
                hnd.append(_Handedness(lab, sc))
            return HandTrackingResult(lms, hnd)
        return self._infer_region(rgb, x_scale=1.0, x_bias=0.0)

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
                    lab = "Left" if x0 < 0.5 else "Right"
                else:
                    lab = "Left" if si == 0 else "Right"
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
                lab = "Left" if x0 < 0.5 else "Right"
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


def create_tracker(
    backend: str,
    *,
    max_hands: int,
    model_complexity: int,
    dxnn_path: str,
    dxnn_layout: Optional[str],
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
    raise SystemExit(f"알 수 없는 --backend: {backend} (cpu | npu)")
