"""Hit-zone mapping and velocity-based strike detection — multi-hand / multi-finger."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# MediaPipe hand landmarks: fingertip indices
FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
FINGER_LABELS: dict[int, str] = {
    4: "thumb",
    8: "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}


@dataclass
class Zone:
    name: str
    sound_key: str
    x0: float
    y0: float
    x1: float
    y1: float


def default_zones() -> list[Zone]:
    """4x4 pad grid in normalized coords; 16 distinct instruments."""
    keys = (
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
        "cowbell",
        "shaker",
        "conga",
        "bongo",
        "perc",
        "fx",
    )
    zones: list[Zone] = []
    cols, rows = 4, 4
    margin_x, margin_y = 0.02, 0.04
    cell_w = (1.0 - 2 * margin_x) / cols
    cell_h = (1.0 - 2 * margin_y) / rows
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x0 = margin_x + c * cell_w + 0.004
            x1 = margin_x + (c + 1) * cell_w - 0.004
            y0 = margin_y + r * cell_h + 0.006
            y1 = margin_y + (r + 1) * cell_h - 0.006
            zones.append(Zone(f"pad{idx+1:02d}", keys[idx], x0, y0, x1, y1))
    return zones


class StrikeDetector:
    """Per (hand_id, landmark_id) velocity state; cooldown per (hand, finger, zone)."""

    def __init__(
        self,
        zones: list[Zone],
        *,
        vy_trigger: float = 0.012,
        cooldown_s: float = 0.1,
        min_conf: float = 0.5,
    ) -> None:
        self.zones = zones
        self.vy_trigger = vy_trigger
        self.cooldown_s = cooldown_s
        self.min_conf = min_conf
        self._prev_y: dict[tuple[int, int], float] = {}
        self._prev_t: dict[tuple[int, int], float] = {}
        self._last_hit: dict[tuple[int, int, str], float] = {}

    def reset(self) -> None:
        self._prev_y.clear()
        self._prev_t.clear()
        self._last_hit.clear()

    def update_finger(
        self,
        hand_id: int,
        landmark_id: int,
        t_s: float,
        nx: float,
        ny: float,
        conf: float,
    ) -> Optional[tuple[str, str]]:
        """
        Returns (zone_name, sound_key) on strike, else None.
        """
        vk = (hand_id, landmark_id)
        if conf < self.min_conf:
            self._prev_y.pop(vk, None)
            self._prev_t.pop(vk, None)
            return None

        vy = 0.0
        if vk in self._prev_y and vk in self._prev_t:
            dt = max(t_s - self._prev_t[vk], 1e-4)
            vy = (ny - self._prev_y[vk]) / dt

        self._prev_y[vk] = ny
        self._prev_t[vk] = t_s

        if vy < self.vy_trigger:
            return None

        for z in self.zones:
            if not (z.x0 <= nx <= z.x1 and z.y0 <= ny <= z.y1):
                continue
            hk = (hand_id, landmark_id, z.name)
            if t_s - self._last_hit.get(hk, 0.0) < self.cooldown_s:
                continue
            self._last_hit[hk] = t_s
            return (z.name, z.sound_key)

        return None
