# Plan: Drum Mode — 화면 패드 기반 타격 감지로 전환

## Context

현재 drum 모드는 piano 모드와 동일한 구조로, **손가락별(finger) strike 감지** → 각 손가락에 sound key 매핑 방식이다.  
목표: drum 모드에서는 화면에 **사각형 패드**를 여러 개 표시하고, 손(손가락 끝)이 그 패드 영역 안에서 아래로 내리치면 해당 패드의 소리가 나도록 변경한다. Piano 모드는 그대로 유지.

---

## 변경 범위

| 파일 | 변경 내용 |
|------|-----------|
| `strike_detector.py` | `PadZone` dataclass + `PadStrikeDetector` 클래스 + `default_pad_zones()` + `load_pad_zones_json()` 추가 |
| `main.py` | drum 모드에서 `PadStrikeDetector` 사용, `draw_pads()` 함수, `--drum-pads` arg 추가 |
| `pads.example.json` (신규) | 기본 8-패드 레이아웃 JSON 예시 |

---

## Step 1: `strike_detector.py` — PadZone + PadStrikeDetector 추가

파일 끝에 아래 두 클래스를 추가한다. 기존 코드는 수정하지 않는다.

### PadZone (dataclass)

```python
from dataclasses import dataclass, field

@dataclass
class PadZone:
    label: str               # 표시 이름 (예: "kick")
    sound_key: str           # drumkit_audio 키 (예: "kick")
    x1: float                # 정규화 좌표 [0,1], 좌상단
    y1: float
    x2: float                # 정규화 좌표 [0,1], 우하단
    y2: float
    color: tuple = field(default_factory=lambda: (100, 200, 100))  # BGR

    def contains(self, nx: float, ny: float) -> bool:
        return self.x1 <= nx <= self.x2 and self.y1 <= ny <= self.y2
```

### PadStrikeDetector

기존 `InstrumentStrikeDetector`와 동일한 vy + joint 이중 조건을 유지하되, strike 시 **어느 손가락이든** 패드 안에 있으면 해당 패드를 반환한다.  
패드별 cooldown을 별도로 관리해 같은 패드 연타를 제어한다.

```python
class PadStrikeDetector:
    def __init__(self, pads, vy_trigger=0.03, joint_dps_trigger=20.0,
                 cooldown_s=0.12, min_conf=0.5):
        self.pads = pads
        self.vy_trigger = vy_trigger
        self.joint_dps_trigger = joint_dps_trigger
        self.cooldown_s = cooldown_s
        self.min_conf = min_conf
        self._prev_y: dict = {}
        self._prev_t: dict = {}
        self._prev_angle: dict = {}
        self._pad_last_hit: dict = {}   # pad.label → last hit time

    def update_finger(self, hand_id, tip_id, t_s, hand_lms, conf) -> Optional[PadZone]:
        # 기존 InstrumentStrikeDetector.update_finger와 동일한 vy/joint 계산
        # hit 조건 만족 시 → 손끝이 들어있는 PadZone 반환 (없으면 None)
        # 패드별 cooldown 적용
        ...
```

### default_pad_zones() 함수

인자 없이 호출하면 8-패드 기본 레이아웃 반환. 화면 중앙 영역(y: 0.35~0.85)에 4×2 그리드.

```python
def default_pad_zones() -> list[PadZone]:
    # 4열 × 2행
    # 행1 (y 0.35~0.60): kick | snare | hat | ride
    # 행2 (y 0.60~0.85): tom_l | tom_m | crash | clap
    sounds = ["kick","snare","hat","ride","tom_l","tom_m","crash","clap"]
    colors = [(180,80,80),(80,180,80),(80,80,200),(180,180,60),
              (60,180,180),(180,60,180),(60,120,200),(180,120,60)]
    pads = []
    cols, rows = 4, 2
    x_margin, y_top, y_bot = 0.05, 0.35, 0.85
    pad_w = (1.0 - 2*x_margin) / cols
    pad_h = (y_bot - y_top) / rows
    for i, (s, c) in enumerate(zip(sounds, colors)):
        col, row = i % cols, i // cols
        x1 = x_margin + col * pad_w
        y1 = y_top + row * pad_h
        pads.append(PadZone(s, s, x1, y1, x1+pad_w-0.01, y1+pad_h-0.01, c))
    return pads
```

### load_pad_zones_json() 함수

```python
def load_pad_zones_json(path: str, valid_keys: frozenset) -> list[PadZone]:
    # JSON 형식: {"pads": [{"label":"kick","sound":"kick","x1":0.05,"y1":0.35,"x2":0.29,"y2":0.59}, ...]}
    ...
```

---

## Step 2: `main.py` — drum 모드 패드 감지로 교체

### 2-1. `--drum-pads` 인자 추가

```python
p.add_argument("--drum-pads", type=str, default="",
    help="드럼 패드 레이아웃 JSON (기본: 내장 8-패드 그리드)")
```

### 2-2. 초기화 분기

```python
if args.piano:
    # 기존 piano 로직 그대로
    ...
else:
    # drum: PadStrikeDetector 사용
    from strike_detector import PadStrikeDetector, default_pad_zones, load_pad_zones_json
    kit = build_kit()
    if args.drum_pads.strip():
        pad_zones = load_pad_zones_json(args.drum_pads, frozenset(kit))
    else:
        pad_zones = default_pad_zones()
    pad_det = PadStrikeDetector(
        pad_zones,
        vy_trigger=args.vy_trigger,
        joint_dps_trigger=args.joint_dps,
        cooldown_s=args.cooldown,
    )
```

### 2-3. 메인 루프 — drum 분기

기존 `det.update_finger(...)` 호출 블록을 `if args.piano` / `else` 로 분리:

```python
# drum 모드
for hand_idx, hand_lms in enumerate(landmarks_list):
    for fid in FINGERTIP_INDICES:
        hit_pad = pad_det.update_finger(hand_idx, fid, t, hand_lms, conf)
        if hit_pad:
            kit[hit_pad.sound_key].play()
            strike_events.append((t + STRIKE_DISPLAY_SEC,
                                   hit_pad.label,
                                   hit_pad.color))
            active_pads[hit_pad.label] = t + 0.12  # 히트 플래시 상태 기록
    # 손끝 궤적 그리기 (기존 trail 코드 유지)
    ...
```

### 2-4. 패드 그리기 함수 `draw_pads()`

`main.py`에 헬퍼 함수 추가:

```python
def draw_pads(frame: np.ndarray, pads: list, active_until: dict, t: float) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for pad in pads:
        x1, y1 = int(pad.x1*w), int(pad.y1*h)
        x2, y2 = int(pad.x2*w), int(pad.y2*h)
        is_active = t < active_until.get(pad.label, 0)
        fill_color = tuple(min(255, c+100) for c in pad.color) if is_active else pad.color
        cv2.rectangle(overlay, (x1,y1), (x2,y2), fill_color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x1,y1), (x2,y2),
                      fill_color if is_active else pad.color, 2, cv2.LINE_AA)
        cv2.putText(frame, pad.label, (x1+8, y1+32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
```

루프 안에서 `frame`에 패드를 그린 후 resize:

```python
# drum 모드일 때만
if not args.piano:
    draw_pads(frame, pad_zones, active_pads, t)
vid_scaled = cv2.resize(frame, (VIDEO_W, VIDEO_H), ...)
```

---

## Step 3: `pads.example.json` 신규 파일

```json
{
  "description": "커스텀 드럼 패드 레이아웃 예시 (정규화 좌표 0~1)",
  "pads": [
    {"label": "kick",  "sound": "kick",  "x1": 0.05, "y1": 0.35, "x2": 0.29, "y2": 0.59, "color": [80, 80, 180]},
    {"label": "snare", "sound": "snare", "x1": 0.30, "y1": 0.35, "x2": 0.54, "y2": 0.59, "color": [80, 180, 80]},
    {"label": "hat",   "sound": "hat",   "x1": 0.55, "y1": 0.35, "x2": 0.74, "y2": 0.59, "color": [200, 80, 80]},
    {"label": "ride",  "sound": "ride",  "x1": 0.75, "y1": 0.35, "x2": 0.94, "y2": 0.59, "color": [60, 180, 180]},
    {"label": "tom_l", "sound": "tom_l", "x1": 0.05, "y1": 0.60, "x2": 0.29, "y2": 0.84, "color": [180, 180, 60]},
    {"label": "tom_m", "sound": "tom_m", "x1": 0.30, "y1": 0.60, "x2": 0.54, "y2": 0.84, "color": [180, 60, 180]},
    {"label": "crash", "sound": "crash", "x1": 0.55, "y1": 0.60, "x2": 0.74, "y2": 0.84, "color": [60, 120, 200]},
    {"label": "clap",  "sound": "clap",  "x1": 0.75, "y1": 0.60, "x2": 0.94, "y2": 0.84, "color": [180, 120, 60]}
  ]
}
```

---

## 유지되는 것

- Piano 모드: 변경 없음 (`InstrumentStrikeDetector` + finger → sound_key 매핑 그대로)
- NPU / CPU 백엔드 선택: 변경 없음 (hand_tracker.py 무관)
- Trail 시각화, FPS 오버레이, 사이드바 레이아웃: 그대로 유지
- `--vy-trigger`, `--joint-dps`, `--cooldown` 파라미터: drum 패드 모드에서도 동일하게 적용

---

## 검증 방법

```bash
# 기본 8-패드 레이아웃
python main.py

# 커스텀 패드 JSON
python main.py --drum-pads pads.example.json

# 기존 piano 모드 변화 없음 확인
python main.py --piano
```

- 패드 사각형이 화면에 표시되는지 확인
- 손가락 끝이 패드 영역 안에서 아래로 내리칠 때 소리 발생 확인
- 패드 밖에서 내리쳐도 소리 미발생 확인
- 히트 시 패드가 밝아지는 시각 피드백 확인
- Piano 모드 (`--piano`)는 기존과 동일하게 동작하는지 확인

