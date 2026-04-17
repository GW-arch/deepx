"""Hit-zone mapping and velocity-based strike detection with cooldown."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Zone:
    name: str
    sound_key: str
    x0: float
    y0: float
    x1: float
    y1: float


def default_zones() -> list[Zone]:
    # Normalized coords [0,1]. Bottom row: four pads.
    return [
        Zone("kick", "kick", 0.05, 0.58, 0.23, 0.95),
        Zone("snare", "snare", 0.27, 0.58, 0.48, 0.95),
        Zone("hat", "hat", 0.52, 0.58, 0.73, 0.95),
        Zone("tom", "tom", 0.77, 0.58, 0.98, 0.95),
    ]


class StrikeDetector:
    def __init__(
        self,
        zones: list[Zone],
        *,
        vy_trigger: float = 0.012,
        cooldown_s: float = 0.12,
        min_conf: float = 0.5,
    ) -> None:
        self.zones = zones
        self.vy_trigger = vy_trigger
        self.cooldown_s = cooldown_s
        self.min_conf = min_conf
        self._last_hit: dict[str, float] = {z.name: 0.0 for z in zones}
        self._prev_y: Optional[float] = None
        self._prev_t: Optional[float] = None

    def reset(self) -> None:
        self._prev_y = None
        self._prev_t = None
        self._last_hit = {z.name: 0.0 for z in self.zones}

    def update(self, t_s: float, nx: float, ny: float, conf: float) -> Optional[tuple[str, str]]:
        """
        Returns (zone_name, sound_key) when a strike is detected, else None.
        nx, ny: normalized fingertip position (MediaPipe).
        """
        if conf < self.min_conf:
            self._prev_y = None
            self._prev_t = None
            return None

        vy = 0.0
        if self._prev_y is not None and self._prev_t is not None:
            dt = max(t_s - self._prev_t, 1e-4)
            vy = (ny - self._prev_y) / dt

        self._prev_y = ny
        self._prev_t = t_s

        if vy < self.vy_trigger:
            return None

        for z in self.zones:
            if not (z.x0 <= nx <= z.x1 and z.y0 <= ny <= z.y1):
                continue
            if t_s - self._last_hit[z.name] < self.cooldown_s:
                continue
            self._last_hit[z.name] = t_s
            return (z.name, z.sound_key)

        return None
