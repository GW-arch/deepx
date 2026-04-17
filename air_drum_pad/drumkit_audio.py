"""Pre-generated multi-instrument samples for low-latency pygame playback."""
from __future__ import annotations

import re
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


# --- 피아노 모드: 음명 키 (예: C4, D#5) → 짧은 사인파 + 감쇠 ---

_PITCH_CLASS: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

# 손0 엄지→소지, 손1 엄지→소지 — C 메이저 10음 (바꾸려면 --instruments + JSON)
PIANO_DEFAULT_SLOTS: tuple[str, ...] = (
    "C4",
    "D4",
    "E4",
    "F4",
    "G4",
    "A4",
    "B4",
    "C5",
    "D5",
    "E5",
)


def note_name_to_midi(name: str) -> int:
    """'C4', 'D#5', 'Bb3' 형식 → MIDI 노트 번호."""
    m = re.match(r"^([A-Ga-g])([#b]?)(\d+)\s*$", name.strip())
    if not m:
        raise ValueError(f"잘못된 음 이름: {name!r} (예: C4, D#5, Bb3)")
    letter = m.group(1).upper()
    acc = m.group(2)
    octave = int(m.group(3))
    pc = _PITCH_CLASS[letter]
    if acc == "#":
        pc += 1
    elif acc == "b":
        pc -= 1
    elif acc != "":
        raise ValueError(name)
    # C4 = MIDI 60
    return 12 * (octave + 1) + pc


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _piano_tone(sample_rate: int, freq_hz: float, duration: float = 0.32) -> np.ndarray:
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    env = np.exp(-t * 5.5) * (1.0 - np.exp(-t * 180.0))  # 짧은 어택
    h2 = 0.12 * np.sin(4 * np.pi * freq_hz * t)
    h3 = 0.06 * np.sin(6 * np.pi * freq_hz * t)
    wave = np.sin(2 * np.pi * freq_hz * t) + h2 + h3
    return 0.35 * env * wave


def build_piano_kit_for_slots(
    slots: tuple[str, ...],
    sample_rate: int = 22050,
) -> dict[str, pygame.mixer.Sound]:
    """slots에 등장하는 음명만 미리 합성 (커스텀 JSON용)."""
    uniq = tuple(dict.fromkeys(slots))
    return build_piano_kit(sample_rate, note_names=uniq)


def build_piano_kit(
    sample_rate: int = 22050,
    note_names: tuple[str, ...] = PIANO_DEFAULT_SLOTS,
) -> dict[str, pygame.mixer.Sound]:
    """음명 문자열 키로 pygame Sound 생성 (드럼 `build_kit`과 동일 API 형태)."""
    pygame.mixer.pre_init(sample_rate, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(32)
    out: dict[str, pygame.mixer.Sound] = {}
    for name in note_names:
        hz = _midi_to_hz(note_name_to_midi(name))
        mono = _piano_tone(sample_rate, hz)
        out[name] = pygame.sndarray.make_sound(_to_stereo_i16(mono))
    return out


def piano_kit_keys(note_names: tuple[str, ...] = PIANO_DEFAULT_SLOTS) -> tuple[str, ...]:
    return tuple(note_names)


def load_piano_slots_json(path: str) -> tuple[str, ...]:
    """피아노용 instruments JSON — 값은 C4, D#5 같은 음명 10개."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "slots" in raw:
        slots = raw["slots"]
    elif isinstance(raw, list):
        slots = raw
    else:
        raise ValueError('JSON은 {"slots":[...]} 또는 배열이어야 합니다.')
    if not isinstance(slots, list) or not all(isinstance(x, str) for x in slots):
        raise ValueError("slots는 문자열 배열")
    if len(slots) < 10:
        raise ValueError("slots는 10개(양손×5손가락) 필요")
    out: list[str] = []
    for s in slots[:10]:
        note_name_to_midi(s)  # 형식 검증
        out.append(s.strip())
    return tuple(out)
