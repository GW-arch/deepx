"""손가락 관절 + 손끝 속도로 타격 판정 (고정 영역 없음, 추적 기반)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# MediaPipe hand landmarks — fingertip indices
FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
FINGER_LABELS: dict[int, str] = {
    4: "thumb",
    8: "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}

# 타격 판정용: 각 손가락의 (MCP/IP, PIP, TIP) — 각도는 가운데 관절에서 계산
FINGER_ANGLE_CHAIN: dict[int, tuple[int, int, int]] = {
    4: (2, 3, 4),  # thumb: CMC→IP 구간은 생략, MCP(2)-IP(3)-TIP(4)
    8: (5, 6, 8),  # index: MCP-PIP-TIP (DIP 생략해 안정적으로)
    12: (9, 10, 12),
    16: (13, 14, 16),
    20: (17, 18, 20),
}

# Per-finger tuning from live use: middle-finger flexion is often smaller in
# camera coordinates, so it needs a lower angular threshold to avoid missed hits.
FINGER_VY_TRIGGER_SCALE: dict[int, float] = {
    12: 0.85,  # middle
}
FINGER_JOINT_DPS_TRIGGER_SCALE: dict[int, float] = {
    12: 0.65,  # middle
}

_DEFAULT_SOUND_BY_SLOT: tuple[str, ...] = (
    "kick",
    "snare",
    "hat",
    "ride",
    "tom_l",
    "tom_m",
    "hat_o",
    "crash",
    "clap",
    "rim",
)


def sound_key_for_finger(
    hand_id: int,
    landmark_id: int,
    *,
    max_hands: int = 2,
    sound_slots: Optional[tuple[str, ...]] = None,
) -> str:
    """sound_slots: 길이 10 권장 — [손0 엄지…소지, 손1 엄지…소지]. None이면 기본 킥/스네어… 매핑."""
    if landmark_id not in FINGER_LABELS:
        slots = sound_slots if sound_slots is not None else _DEFAULT_SOUND_BY_SLOT
        return slots[0]
    fi = FINGERTIP_INDICES.index(landmark_id)
    slot = int(hand_id) * len(FINGERTIP_INDICES) + fi
    slots = sound_slots if sound_slots is not None else _DEFAULT_SOUND_BY_SLOT
    nh = min(max(1, max_hands), 2)
    need = nh * len(FINGERTIP_INDICES)
    if len(slots) < need:
        return slots[slot % len(slots)]
    return slots[slot]


def load_instrument_slots_json(
    path: str,
    valid_keys: Optional[frozenset[str]] = None,
) -> tuple[str, ...]:
    """
    JSON 형식:
      { "slots": ["kick","snare", ...] }  또는  ["kick","snare", ...]
    길이 10(양손×5손가락). valid_keys가 None이면 드럼 kit_keys(), 피아노는 main에서 frozenset(kit) 넘김.
    """
    from drumkit_audio import kit_keys

    valid = valid_keys if valid_keys is not None else frozenset(kit_keys())
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "slots" in raw:
        slots = raw["slots"]
    elif isinstance(raw, list):
        slots = raw
    else:
        raise ValueError("JSON은 배열이거나 {\"slots\": [...] } 형식이어야 합니다.")
    if not isinstance(slots, list) or not all(isinstance(x, str) for x in slots):
        raise ValueError("slots는 문자열 배열이어야 합니다.")
    bad = [x for x in slots if x not in valid]
    if bad:
        raise ValueError(f"알 수 없는 sound key: {bad}. 사용 가능: {sorted(valid)}")
    need = 10  # 양손 × 5 (max-hands=1이면 앞 5개만 사용)
    if len(slots) < need:
        raise ValueError(f"slots는 최소 {need}개(손0 엄지→소지, 손1 엄지→소지 순) 필요합니다.")
    return tuple(slots)


def _angle_deg_at_b(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    """B에서의 각도(도), 벡터 BA·BC."""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(c))


class InstrumentStrikeDetector:
    """
    실제로 손가락으로 치는 것과 비슷하게:
    - 손끝의 **아래 방향 속도** (타격면을 향하는 내리침)
    - 같은 손가락 관절의 **각속도** (관절이 움직이며 스틱/손가락이 가속되는 느낌)

    둘을 동시에 만족할 때만 타격으로 인정해, 팔만 흔들 때 생기는 오탐을 줄입니다.
    """

    def __init__(
        self,
        *,
        vy_trigger: float = 0.025,
        joint_dps_trigger: float = 16.0,
        cooldown_s: float = 0.10,
        min_tip_disp: float = 0.008,
        relative_vy_scale: float = 0.30,
        min_conf: float = 0.5,
        max_hands: int = 2,
        sound_mapper: Optional[Callable[[int, int], str]] = None,
    ) -> None:
        self.vy_trigger = vy_trigger
        self.joint_dps_trigger = joint_dps_trigger
        self.cooldown_s = cooldown_s
        self.min_tip_disp = min_tip_disp
        self.relative_vy_scale = relative_vy_scale
        self.min_conf = min_conf
        self.max_hands = max_hands
        if sound_mapper is not None:
            self._sound_mapper = sound_mapper
        else:

            def _default_map(h: int, lm: int) -> str:
                return sound_key_for_finger(h, lm, max_hands=max_hands)

            self._sound_mapper = _default_map
        self._prev_y: dict[tuple[int, int], float] = {}
        self._prev_x: dict[tuple[int, int], float] = {}
        self._prev_rel_y: dict[tuple[int, int], float] = {}
        self._prev_t: dict[tuple[int, int], float] = {}
        self._prev_angle: dict[tuple[int, int], float] = {}
        self._last_hit: dict[tuple[int, int], float] = {}
        self._last_hit_y: dict[tuple[int, int], float] = {}

    def reset(self) -> None:
        self._prev_y.clear()
        self._prev_x.clear()
        self._prev_rel_y.clear()
        self._prev_t.clear()
        self._prev_angle.clear()
        self._last_hit.clear()
        self._last_hit_y.clear()

    def update_finger(
        self,
        hand_id: int,
        tip_id: int,
        t_s: float,
        hand_lms: Any,
        conf: float,
    ) -> Optional[tuple[str, str]]:
        """
        hand_lms: 한 손의 landmark 리스트.
        Returns (track_id, sound_key) 또는 None.
        """
        vk = (hand_id, tip_id)
        if conf < self.min_conf:
            self._prev_y.pop(vk, None)
            self._prev_rel_y.pop(vk, None)
            self._prev_t.pop(vk, None)
            self._prev_angle.pop(vk, None)
            return None

        tip = hand_lms.landmark[tip_id]
        nx, ny = tip.x, tip.y
        wrist = hand_lms.landmark[0]
        rel_y = ny - wrist.y

        dt = 1e-4
        if vk in self._prev_t:
            dt = max(t_s - self._prev_t[vk], 1e-4)
        vy = 0.0
        if vk in self._prev_y:
            vy = (ny - self._prev_y[vk]) / dt
        rel_vy = 0.0
        if vk in self._prev_rel_y:
            rel_vy = (rel_y - self._prev_rel_y[vk]) / dt

        joint_dps = 0.0
        if tip_id in FINGER_ANGLE_CHAIN:
            a, b, c = FINGER_ANGLE_CHAIN[tip_id]
            la, lb, lc = hand_lms.landmark[a], hand_lms.landmark[b], hand_lms.landmark[c]
            ang = _angle_deg_at_b(la.x, la.y, lb.x, lb.y, lc.x, lc.y)
            if vk in self._prev_angle and vk in self._prev_t:
                joint_dps = (ang - self._prev_angle[vk]) / dt
            self._prev_angle[vk] = ang

        self._prev_y[vk] = ny
        self._prev_rel_y[vk] = rel_y
        self._prev_t[vk] = t_s

        # 아래로 빠른 내리침 + 관절이 동시에 움직임 (펴짐/절곡 급변)
        tip_threshold = self.vy_trigger * FINGER_VY_TRIGGER_SCALE.get(tip_id, 1.0)
        joint_threshold = (
            self.joint_dps_trigger * FINGER_JOINT_DPS_TRIGGER_SCALE.get(tip_id, 1.0)
        )
        tip_ok = vy >= tip_threshold
        # Reject whole-hand/model-origin jumps: a true finger hit should move
        # the fingertip downward at least a little relative to the wrist.
        rel_tip_ok = rel_vy >= tip_threshold * self.relative_vy_scale
        joint_ok = abs(joint_dps) >= joint_threshold

        if not (tip_ok and rel_tip_ok and joint_ok):
            return None

        # Minimum displacement since last hit — reject jitter/noise
        last_hit_y = self._last_hit_y.get(vk, ny - 1.0)
        disp = ny - last_hit_y  # positive = moved down since last hit
        if abs(disp) < self.min_tip_disp:
            return None

        if t_s - self._last_hit.get(vk, 0.0) < self.cooldown_s:
            return None

        self._last_hit[vk] = t_s
        self._last_hit_y[vk] = ny
        label = FINGER_LABELS.get(tip_id, "tip")
        track_id = f"h{hand_id}_{label}"
        sound_key = self._sound_mapper(hand_id, tip_id)
        return (track_id, sound_key)


@dataclass
class PadZone:
    """Normalized rectangular drum pad zone."""

    label: str
    sound_key: str
    x1: float
    y1: float
    x2: float
    y2: float
    color: tuple[int, int, int] = field(default_factory=lambda: (100, 200, 100))

    def contains(self, nx: float, ny: float) -> bool:
        return self.x1 <= nx <= self.x2 and self.y1 <= ny <= self.y2


class PadStrikeDetector:
    """
    Pad based strike detector for drum mode.

    It keeps the same two-part strike condition as InstrumentStrikeDetector
    (downward fingertip velocity + joint angular velocity), but returns the
    PadZone that contains the striking fingertip. Cooldown is tracked per pad
    label so different pads can be played in rapid succession.
    """

    def __init__(
        self,
        pads: list[PadZone],
        vy_trigger: float = 0.025,
        joint_dps_trigger: float = 16.0,
        cooldown_s: float = 0.10,
        relative_vy_scale: float = 0.30,
        min_conf: float = 0.5,
    ) -> None:
        self.pads = list(pads)
        self.vy_trigger = vy_trigger
        self.joint_dps_trigger = joint_dps_trigger
        self.cooldown_s = cooldown_s
        self.relative_vy_scale = relative_vy_scale
        self.min_conf = min_conf
        self._prev_y: dict[tuple[int, int], float] = {}
        self._prev_rel_y: dict[tuple[int, int], float] = {}
        self._prev_t: dict[tuple[int, int], float] = {}
        self._prev_angle: dict[tuple[int, int], float] = {}
        self._pad_last_hit: dict[str, float] = {}

    def reset(self) -> None:
        self._prev_y.clear()
        self._prev_rel_y.clear()
        self._prev_t.clear()
        self._prev_angle.clear()
        self._pad_last_hit.clear()

    def update_finger(
        self,
        hand_id: int,
        tip_id: int,
        t_s: float,
        hand_lms: Any,
        conf: float,
    ) -> Optional[PadZone]:
        """Returns the hit PadZone, or None when no pad strike is detected."""
        vk = (hand_id, tip_id)
        if conf < self.min_conf:
            self._prev_y.pop(vk, None)
            self._prev_rel_y.pop(vk, None)
            self._prev_t.pop(vk, None)
            self._prev_angle.pop(vk, None)
            return None

        tip = hand_lms.landmark[tip_id]
        nx, ny = tip.x, tip.y
        wrist = hand_lms.landmark[0]
        rel_y = ny - wrist.y

        dt = 1e-4
        if vk in self._prev_t:
            dt = max(t_s - self._prev_t[vk], 1e-4)
        vy = 0.0
        if vk in self._prev_y:
            vy = (ny - self._prev_y[vk]) / dt
        rel_vy = 0.0
        if vk in self._prev_rel_y:
            rel_vy = (rel_y - self._prev_rel_y[vk]) / dt

        joint_dps = 0.0
        if tip_id in FINGER_ANGLE_CHAIN:
            a, b, c = FINGER_ANGLE_CHAIN[tip_id]
            la, lb, lc = hand_lms.landmark[a], hand_lms.landmark[b], hand_lms.landmark[c]
            ang = _angle_deg_at_b(la.x, la.y, lb.x, lb.y, lc.x, lc.y)
            if vk in self._prev_angle and vk in self._prev_t:
                joint_dps = (ang - self._prev_angle[vk]) / dt
            self._prev_angle[vk] = ang

        self._prev_y[vk] = ny
        self._prev_rel_y[vk] = rel_y
        self._prev_t[vk] = t_s

        tip_threshold = self.vy_trigger * FINGER_VY_TRIGGER_SCALE.get(tip_id, 1.0)
        joint_threshold = (
            self.joint_dps_trigger * FINGER_JOINT_DPS_TRIGGER_SCALE.get(tip_id, 1.0)
        )
        tip_ok = vy >= tip_threshold
        # Reject ghost hits caused by the whole stationary hand/skeleton
        # shifting when another hand moves: actual strikes should move the
        # fingertip downward relative to the wrist, not only in image space.
        rel_tip_ok = rel_vy >= tip_threshold * self.relative_vy_scale
        joint_ok = abs(joint_dps) >= joint_threshold
        if not (tip_ok and rel_tip_ok and joint_ok):
            return None

        for pad in self.pads:
            if not pad.contains(nx, ny):
                continue
            if t_s - self._pad_last_hit.get(pad.label, -1e9) < self.cooldown_s:
                return None
            self._pad_last_hit[pad.label] = t_s
            return pad
        return None


def default_pad_zones() -> list[PadZone]:
    """Return the built-in 4x2 drum pad layout in normalized coordinates."""
    sounds = ["kick", "snare", "hat", "ride", "tom_l", "tom_m", "crash", "clap"]
    colors = [
        (180, 80, 80),
        (80, 180, 80),
        (80, 80, 200),
        (180, 180, 60),
        (60, 180, 180),
        (180, 60, 180),
        (60, 120, 200),
        (180, 120, 60),
    ]
    pads: list[PadZone] = []
    cols, rows = 4, 2
    x_margin, y_top, y_bot = 0.05, 0.35, 0.85
    pad_w = (1.0 - 2 * x_margin) / cols
    pad_h = (y_bot - y_top) / rows
    for i, (sound, color) in enumerate(zip(sounds, colors)):
        col, row = i % cols, i // cols
        x1 = x_margin + col * pad_w
        y1 = y_top + row * pad_h
        pads.append(
            PadZone(
                sound,
                sound,
                x1,
                y1,
                x1 + pad_w - 0.01,
                y1 + pad_h - 0.01,
                color,
            )
        )
    return pads


def _coerce_pad_color(raw: Any, label: str) -> tuple[int, int, int]:
    if raw is None:
        return (100, 200, 100)
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 3
        or not all(isinstance(c, int) for c in raw)
    ):
        raise ValueError(f"pad '{label}' color는 [B,G,R] 정수 3개 배열이어야 합니다.")
    if any(c < 0 or c > 255 for c in raw):
        raise ValueError(f"pad '{label}' color 값은 0~255 범위여야 합니다.")
    return (int(raw[0]), int(raw[1]), int(raw[2]))


def load_pad_zones_json(path: str, valid_keys: frozenset[str]) -> list[PadZone]:
    """
    Load drum pad zones from JSON.

    Expected format:
      {
        "pads": [
          {"label":"kick", "sound":"kick",
           "x1":0.05, "y1":0.35, "x2":0.29, "y2":0.59,
           "color":[80,80,180]}
        ]
      }
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "pads" not in raw:
        raise ValueError('JSON은 {"pads": [...]} 형식이어야 합니다.')

    items = raw["pads"]
    if not isinstance(items, list) or not items:
        raise ValueError("pads는 하나 이상의 패드 객체 배열이어야 합니다.")

    pads: list[PadZone] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"pads[{i}]는 객체여야 합니다.")

        label = item.get("label")
        sound_key = item.get("sound", item.get("sound_key"))
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"pads[{i}].label은 비어 있지 않은 문자열이어야 합니다.")
        if not isinstance(sound_key, str) or not sound_key.strip():
            raise ValueError(f"pad '{label}' sound는 비어 있지 않은 문자열이어야 합니다.")
        label = label.strip()
        sound_key = sound_key.strip()
        if sound_key not in valid_keys:
            raise ValueError(
                f"pad '{label}'의 알 수 없는 sound key: {sound_key}. "
                f"사용 가능: {sorted(valid_keys)}"
            )

        coords: dict[str, float] = {}
        for key in ("x1", "y1", "x2", "y2"):
            value = item.get(key)
            if not isinstance(value, (int, float)):
                raise ValueError(f"pad '{label}' {key}는 숫자여야 합니다.")
            value_f = float(value)
            if not 0.0 <= value_f <= 1.0:
                raise ValueError(f"pad '{label}' {key}는 0~1 범위여야 합니다.")
            coords[key] = value_f
        if coords["x1"] >= coords["x2"] or coords["y1"] >= coords["y2"]:
            raise ValueError(f"pad '{label}' 좌표는 x1<x2, y1<y2 이어야 합니다.")

        pads.append(
            PadZone(
                label=label,
                sound_key=sound_key,
                x1=coords["x1"],
                y1=coords["y1"],
                x2=coords["x2"],
                y2=coords["y2"],
                color=_coerce_pad_color(item.get("color"), label),
            )
        )
    return pads
