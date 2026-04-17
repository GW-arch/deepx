"""손가락 관절 + 손끝 속도로 타격 판정 (고정 영역 없음, 추적 기반)."""
from __future__ import annotations

import math
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


def sound_key_for_finger(hand_id: int, landmark_id: int, max_hands: int = 2) -> str:
    if landmark_id not in FINGER_LABELS:
        return _DEFAULT_SOUND_BY_SLOT[0]
    fi = FINGERTIP_INDICES.index(landmark_id)
    slot = int(hand_id) * len(FINGERTIP_INDICES) + fi
    cap = min(max_hands, 2) * len(FINGERTIP_INDICES)
    return _DEFAULT_SOUND_BY_SLOT[slot % max(cap, 1)]


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
        vy_trigger: float = 0.01,
        joint_dps_trigger: float = 120.0,
        cooldown_s: float = 0.12,
        min_conf: float = 0.5,
        max_hands: int = 2,
        sound_mapper: Optional[Callable[[int, int], str]] = None,
    ) -> None:
        self.vy_trigger = vy_trigger
        self.joint_dps_trigger = joint_dps_trigger
        self.cooldown_s = cooldown_s
        self.min_conf = min_conf
        self.max_hands = max_hands
        self._sound_mapper = sound_mapper or (
            lambda h, lm: sound_key_for_finger(h, lm, max_hands=max_hands)
        )
        self._prev_y: dict[tuple[int, int], float] = {}
        self._prev_t: dict[tuple[int, int], float] = {}
        self._prev_angle: dict[tuple[int, int], float] = {}
        self._last_hit: dict[tuple[int, int], float] = {}

    def reset(self) -> None:
        self._prev_y.clear()
        self._prev_t.clear()
        self._prev_angle.clear()
        self._last_hit.clear()

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
            self._prev_t.pop(vk, None)
            self._prev_angle.pop(vk, None)
            return None

        tip = hand_lms.landmark[tip_id]
        nx, ny = tip.x, tip.y

        dt = 1e-4
        if vk in self._prev_t:
            dt = max(t_s - self._prev_t[vk], 1e-4)
        vy = 0.0
        if vk in self._prev_y:
            vy = (ny - self._prev_y[vk]) / dt

        joint_dps = 0.0
        if tip_id in FINGER_ANGLE_CHAIN:
            a, b, c = FINGER_ANGLE_CHAIN[tip_id]
            la, lb, lc = hand_lms.landmark[a], hand_lms.landmark[b], hand_lms.landmark[c]
            ang = _angle_deg_at_b(la.x, la.y, lb.x, lb.y, lc.x, lc.y)
            if vk in self._prev_angle and vk in self._prev_t:
                joint_dps = (ang - self._prev_angle[vk]) / dt
            self._prev_angle[vk] = ang

        self._prev_y[vk] = ny
        self._prev_t[vk] = t_s

        # 아래로 빠른 내리침 + 관절이 동시에 움직임 (펴짐/절곡 급변)
        tip_ok = vy >= self.vy_trigger
        joint_ok = abs(joint_dps) >= self.joint_dps_trigger

        if not (tip_ok and joint_ok):
            return None

        if t_s - self._last_hit.get(vk, 0.0) < self.cooldown_s:
            return None

        self._last_hit[vk] = t_s
        label = FINGER_LABELS.get(tip_id, "tip")
        track_id = f"h{hand_id}_{label}"
        sound_key = self._sound_mapper(hand_id, tip_id)
        return (track_id, sound_key)
