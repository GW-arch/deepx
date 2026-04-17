"""Pre-generated multi-instrument samples for low-latency pygame playback."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pygame


def _to_stereo_i16(mono: np.ndarray) -> np.ndarray:
    mono = np.clip(mono, -1.0, 1.0)
    i16 = (mono * 32767.0).astype(np.int16)
    return np.column_stack((i16, i16))


def _kick(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.18)
    t = np.arange(n, dtype=np.float64) / sr
    f0, f1 = 120.0, 45.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-6) * t * t)
    env = np.exp(-t * 18.0)
    return env * 0.9 * np.sin(phase)


def _snare(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.12)
    t = np.arange(n, dtype=np.float64) / sr
    noise = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
    tone = 0.35 * np.sin(2.0 * np.pi * 180.0 * t)
    env = np.exp(-t * 35.0)
    return env * (0.65 * noise + tone)


def _hat_closed(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.06)
    t = np.arange(n, dtype=np.float64) / sr
    noise = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
    env = np.exp(-t * 70.0)
    return env * 0.45 * noise


def _hat_open(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.18)
    t = np.arange(n, dtype=np.float64) / sr
    noise = np.random.uniform(-0.8, 0.8, n).astype(np.float64)
    env = np.exp(-t * 12.0)
    return env * 0.55 * noise


def _tom(sr: int, f0: float, f1: float) -> np.ndarray:
    n = int(sr * 0.14)
    t = np.arange(n, dtype=np.float64) / sr
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-6) * t * t)
    env = np.exp(-t * 22.0)
    return env * 0.75 * np.sin(phase)


def _crash(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.45)
    t = np.arange(n, dtype=np.float64) / sr
    noise = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
    tone = 0.2 * np.sin(2.0 * np.pi * 400.0 * t) * np.exp(-t * 8.0)
    env = np.exp(-t * 4.5)
    return env * (0.7 * noise + tone)


def _ride(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.25)
    t = np.arange(n, dtype=np.float64) / sr
    noise = np.random.uniform(-0.6, 0.6, n).astype(np.float64)
    metal = 0.35 * np.sin(2.0 * np.pi * (300.0 * t + 50.0 * np.sin(t * 30)))
    env = np.exp(-t * 18.0)
    return env * (0.5 * noise + metal)


def _clap(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.08)
    t = np.arange(n, dtype=np.float64) / sr
    noise = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
    env1 = np.exp(-((t - 0.01) ** 2) / 1e-5)
    env2 = np.exp(-((t - 0.035) ** 2) / 1e-5)
    env = env1 + 0.7 * env2
    return env * 0.55 * noise


def _rim(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.04)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t * 90.0)
    return env * 0.85 * np.sin(2.0 * np.pi * 800.0 * t)


def _cowbell(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.15)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t * 25.0)
    return env * 0.5 * (
        np.sin(2.0 * np.pi * 560.0 * t) + 0.4 * np.sin(2.0 * np.pi * 845.0 * t)
    )


def _shaker(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.1)
    noise = np.random.uniform(-0.5, 0.5, n).astype(np.float64)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t * 40.0)
    return env * noise * 0.6


def _conga(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.12)
    t = np.arange(n, dtype=np.float64) / sr
    f0, f1 = 130.0, 90.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-6) * t * t)
    env = np.exp(-t * 28.0)
    return env * 0.7 * np.sin(phase)


def _bongo(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.08)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t * 45.0)
    return env * 0.65 * np.sin(2.0 * np.pi * (420.0 * t + 80.0 * np.sin(2 * np.pi * 60 * t)))


def _perc_pop(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.05)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t * 55.0)
    return env * 0.7 * np.sin(2.0 * np.pi * 200.0 * t * (1.0 + 4.0 * t))


def _fx_whoosh(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.2)
    t = np.arange(n, dtype=np.float64) / sr
    noise = np.random.uniform(-0.4, 0.4, n).astype(np.float64)
    sweep = np.linspace(0.3, 1.0, n)
    env = np.sin(np.pi * t / max(t[-1], 1e-6))
    return env * sweep * noise * 0.5


# Order matches default_zones() row-major: 4x4 grid
_KIT_BUILDERS: dict[str, Callable[[int], np.ndarray]] = {
    "kick": lambda sr: _kick(sr),
    "snare": lambda sr: _snare(sr),
    "hat": lambda sr: _hat_closed(sr),
    "ride": lambda sr: _ride(sr),
    "tom_l": lambda sr: _tom(sr, 95.0, 55.0),
    "tom_m": lambda sr: _tom(sr, 120.0, 70.0),
    "hat_o": lambda sr: _hat_open(sr),
    "crash": lambda sr: _crash(sr),
    "clap": lambda sr: _clap(sr),
    "rim": lambda sr: _rim(sr),
    "cowbell": lambda sr: _cowbell(sr),
    "shaker": lambda sr: _shaker(sr),
    "conga": lambda sr: _conga(sr),
    "bongo": lambda sr: _bongo(sr),
    "perc": lambda sr: _perc_pop(sr),
    "fx": lambda sr: _fx_whoosh(sr),
}


def build_kit(sample_rate: int = 22050) -> dict[str, pygame.mixer.Sound]:
    pygame.mixer.pre_init(sample_rate, size=-16, channels=2, buffer=256)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(32)
    out: dict[str, pygame.mixer.Sound] = {}
    for key, builder in _KIT_BUILDERS.items():
        mono = builder(sample_rate)
        out[key] = pygame.sndarray.make_sound(_to_stereo_i16(mono))
    return out


def kit_keys() -> tuple[str, ...]:
    return tuple(_KIT_BUILDERS.keys())
