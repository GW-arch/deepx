"""Pre-generated drum-like samples for low-latency pygame playback."""
from __future__ import annotations

import numpy as np
import pygame


def _to_stereo_i16(mono: np.ndarray) -> np.ndarray:
    mono = np.clip(mono, -1.0, 1.0)
    i16 = (mono * 32767.0).astype(np.int16)
    return np.column_stack((i16, i16))


def _kick(sample_rate: int = 22050) -> np.ndarray:
    n = int(sample_rate * 0.18)
    t = np.arange(n, dtype=np.float64) / sample_rate
    f0, f1 = 120.0, 45.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-6) * t * t)
    env = np.exp(-t * 18.0)
    return env * 0.9 * np.sin(phase)


def _snare(sample_rate: int = 22050) -> np.ndarray:
    n = int(sample_rate * 0.12)
    t = np.arange(n, dtype=np.float64) / sample_rate
    noise = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
    tone = 0.35 * np.sin(2.0 * np.pi * 180.0 * t)
    env = np.exp(-t * 35.0)
    return env * (0.65 * noise + tone)


def _hat(sample_rate: int = 22050) -> np.ndarray:
    n = int(sample_rate * 0.06)
    t = np.arange(n, dtype=np.float64) / sample_rate
    noise = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
    env = np.exp(-t * 70.0)
    return env * 0.45 * noise


def _tom(sample_rate: int = 22050) -> np.ndarray:
    n = int(sample_rate * 0.14)
    t = np.arange(n, dtype=np.float64) / sample_rate
    f0, f1 = 95.0, 55.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-6) * t * t)
    env = np.exp(-t * 22.0)
    return env * 0.75 * np.sin(phase)


def build_kit(sample_rate: int = 22050) -> dict[str, pygame.mixer.Sound]:
    pygame.mixer.pre_init(sample_rate, size=-16, channels=2, buffer=256)
    pygame.mixer.init()
    return {
        "kick": pygame.sndarray.make_sound(_to_stereo_i16(_kick(sample_rate))),
        "snare": pygame.sndarray.make_sound(_to_stereo_i16(_snare(sample_rate))),
        "hat": pygame.sndarray.make_sound(_to_stereo_i16(_hat(sample_rate))),
        "tom": pygame.sndarray.make_sound(_to_stereo_i16(_tom(sample_rate))),
    }
