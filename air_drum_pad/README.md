# AI Air-Drum Pad (prototype)

**관리 저장소 (SSH):** `git@github.com:GW-arch/deepx.git`  
웹: [github.com/GW-arch/deepx](https://github.com/GW-arch/deepx)

## 동작 개요 (실제로 치는 것처럼)

화면에 **고정 패드 영역은 없습니다.** 손가락 끝을 **추적**하고, 아래 두 조건을 **동시에** 만족할 때만 한 번 친 것으로 봅니다.

1. **손끝 하강 속도** — 막대기 끝이 아래로 빠르게 움직임 (`vy`, 정규화 좌표/초)
2. **관절 각속도** — MCP–PIP–TIP(엄지는 IP 포함)에서 잰 각도가 프레임마다 충분히 변함 → 손가락 관절이 실제로 휘둘러짐

악기 종류는 **어느 손 × 어느 손가락**인지로 매핑합니다 (기본값은 코드에 있음).

### 악기 바꾸기

1. 사용 가능한 키: `python3 main.py --list-instruments`
2. `instruments.example.json` 을 복사해 `slots` 배열 **10개**를 수정 (순서: **손0** 엄지→소지, **손1** 엄지→소지).
3. 실행: `python3 main.py --camera 0 --instruments my.json`

**음색**을 바꾸려면 `drumkit_audio.py`의 `_KIT_BUILDERS`에 키를 추가·수정한 뒤 JSON에서 그 키를 쓰면 됩니다.

### 피아노 모드

- 실행: `python3 main.py --piano --camera 0`  
  **`--instruments` 없이** 켜면 기본적으로 `instruments.piano.example.json`의 **고정 10키 매핑(C4–E5)** 을 사용합니다.
- 다른 고정 음 배열을 쓰려면: `instruments.piano.example.json` 참고 후
  `python3 main.py --piano --instruments 내피아노.json --camera 0`  
  (`slots` 값은 `C4`, `D#5`, `Bb3` 같은 **음명** 10개)
- 음색은 짧은 **합성** 사인파(실제 샘플 피아노는 아님).
- 사용 가능 음명(기본 10개 나열): `python3 main.py --piano --list-instruments`

## 실행

```bash
cd air_drum_pad
pip3 install -r requirements.txt
```

**sudo password: `deepx123!`**

### 빠른 실행 (모드별)

```bash
# ── 1) CPU 기본 (추가 모델 불필요) ──
python3 main.py --camera 0

# 피아노:
python3 main.py --piano --camera 0

# ── 2) CPU-baseline: Palm(CPU TFLite) + Hand(CPU TFLite) — NPU 없이 동일 파이프라인 ──
python3 main.py --backend cpu-baseline --piano --camera 0

# ── 3) NPU-full: Palm(CPU TFLite) + Hand(NPU .dxnn) — 정식 파이프라인 ──
python3 main.py --backend npu-full --piano --camera 0 \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite.json \
  --palm-tflite models/vendor/palm_detection_lite.tflite

# ── 4) NPU dual-halves: 화면 반분할 근사 (가장 빠름, palm 없음) ──
python3 main.py --backend npu --piano --camera 0 \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite_dual.json
```

> **스크립트로 실행:** `scripts/run_cpu_piano.sh` 또는 `scripts/run_npu_piano.sh` 를 사용하면 환경변수(`DISPLAY`, `XAUTHORITY`)를 자동으로 설정합니다.

### CPU vs CPU+NPU 비교

#### 백엔드별 모델 실행 위치

| `--backend` | Palm Detection | Hand Landmark | 비고 |
|-------------|----------------|---------------|------|
| `cpu` (기본) | CPU (MediaPipe 내장) | CPU (MediaPipe 내장) | 추가 파일 불필요, float32 |
| `cpu-baseline` | CPU (TFLite, float32) | CPU (TFLite, float32) | NPU 없이 npu-full과 동일 파이프라인 (비교 기준선) |
| `npu` | 없음 (dual-halves 근사) | **NPU** (.dxnn, int8) | palm 검출 없이 화면 좌우 반분할 |
| `npu-full` | CPU (TFLite, float32) | **NPU** (.dxnn, int8) | 정식 2-hand 파이프라인 |

> **왜 palm은 CPU인가?** Palm detection .dxnn을 INT8 양자화하면 score head가 파괴됩니다 (ONNX↔NPU 상관 -0.11). DeepX NPU는 INT8 전용 가속기이므로 float32 실행이 불가능합니다. 따라서 palm은 TFLite(CPU, float32)로, hand landmark만 NPU(int8)로 실행하는 하이브리드가 최선입니다. 자세한 분석: [`models/README.md`](models/README.md).

#### 성능 비교

| 구성 | Palm 추론 | Hand 추론 (per hand) | 전체 (2 hands) | 비고 |
|------|----------:|---------------------:|---------------:|------|
| `cpu` (MediaPipe) | ~15 ms | ~10 ms | ~35 ms | 모두 float32 |
| `cpu-baseline` (TFLite palm + TFLite hand) | ~95 ms | ~5 ms | ~105 ms | 모두 float32, 비교 기준선 |
| `npu-full` (TFLite palm + NPU hand) | ~95 ms | ~8 ms | ~111 ms | 매 프레임 palm 실행 |
| `npu` (dual-halves) | 0 ms | ~8 ms × 2 | ~16 ms | palm 검출 없음, 근사 |

> **npu-full은 매 프레임 palm detection을 실행합니다.** 이전에는 landmark 기반 ROI 트래킹으로 palm을 5프레임에 1번만 실행했으나, NPU INT8 양자화 편향이 프레임마다 누적되어 드리프트(최대 dy=0.26)를 일으켰습니다. 항상 palm을 실행하면 드리프트가 제거됩니다(mean |dy|=0.01).

#### cpu-baseline 사용 예시

```bash
# TFLite 모델 자동 탐색 (models/vendor/ 내)
python3 main.py --backend cpu-baseline --max-hands 2

# 명시적 경로 지정
python3 main.py --backend cpu-baseline \
  --palm-tflite models/vendor/palm_detection_lite.tflite \
  --hand-tflite models/vendor/hand_landmark_lite.tflite \
  --max-hands 2
```

> `cpu-baseline`은 `npu-full`과 동일한 파이프라인(palm detection → ROI crop → hand landmark)을 **모두 CPU TFLite**로 실행합니다. NPU 가속 효과를 정확히 비교할 수 있는 기준선입니다.

#### npu-full 사용 예시

```bash
python3 main.py --backend npu-full \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite.json \
  --palm-tflite models/vendor/palm_detection_lite.tflite \
  --max-hands 2
```

`--palm-tflite` 생략 시 `models/vendor/palm_detection_lite.tflite` 자동 탐색. `--palm-dxnn` 플래그도 존재하나 양자화 품질 문제로 **명시적으로 지정한 실험에서만 사용**하세요.

#### 데이터셋 기반 오프라인 벤치마크

라이브 카메라 없이 `dataset/frame_*.png`를 반복 재생해 백엔드별 지연과 landmark 오차를 비교합니다.

```bash
# 기본: cpu-baseline vs npu-full, 동일 palm+ROI 파이프라인 비교
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full

# palm skip/ROI tracking 실험: palm 1회 후 최대 5프레임은 landmark ROI로 추적
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --palm-redetect-every 5

# CSV/JSON 저장
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --csv /tmp/air_drum_bench.csv --json /tmp/air_drum_bench.json
```

`--palm-redetect-every 0`이 기본값이며 매 프레임 palm detection을 실행합니다(드리프트 최소). `N>0`은 지연을 줄이는 **실험 옵션**입니다.

NPU 예시는 `models/README.md` 와 `scripts/run_npu_piano.sh` 참고.  
요약: **MediaPipe TFLite → ONNX** (`tools/export_mediapipe_hand_onnx.py`) → **DX-COM** (SNU 서버 `tools/compile_server_snu.sh`) → 보드에서 `--backend npu-full --palm-tflite … --dxnn …`.

**`dx_engine`은 반드시 보드의 DX-RT와 맞는 빌드**를 쓰세요. 오래된 wheel만 설치하면 `InferenceEngine`이 멈추거나 힙 오류가 날 수 있습니다. 설치 절차는 `requirements-npu.txt` 주석을 따르면 됩니다.

- 종료: `q`
- 민감도: `--vy-trigger`, `--joint-dps` (관절 각속도 하한, deg/s), `--cooldown`

느리게만 움직이면 안 울리게 하려면 `--joint-dps`를 올리고, 너무 안 나오면 `--vy-trigger` / `--joint-dps`를 내립니다.

### 창(OpenCV)이 안 뜰 때

- **모니터에 연결된 보드에서**, 로그인한 **데스크톱 터미널**에서 실행하세요. (Cursor/SSH `tty` 세션만 쓰면 `DISPLAY`는 있어도 창이 로컬에 안 붙을 수 있습니다.)
- 같은 셸에서 한 번 설정 후 실행:

```bash
export DISPLAY=:0
export XAUTHORITY="$HOME/.Xauthority"   # 파일이 있을 때
```

- 카메라가 다른 프로그램에 잡혀 있으면 `Camera read failed` 후 바로 종료합니다. `fuser -v /dev/video0` 로 점유 프로세스를 확인하세요.

## 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [EXPERIMENTS.md](docs/EXPERIMENTS.md) — 벤치마크 결과 포함
- [Palm+Hand NPU 파이프라인 계획](docs/PLAN_NPU_FULL_HAND_PIPELINE.md) — Phase 0~6 로드맵 + 체크박스  
- [다음 세션 실험 가이드](docs/NEXT_SESSION_NPU_PALM.md) — 명령어·체크리스트·완료 기준

## 악기 매핑 다이어그램

`instruments/` 디렉터리에 각 프리셋별 손-소리 매핑 이미지가 있습니다.  
재생성: `python3 tools/gen_instrument_diagrams.py`

### Drum Kit — Default

![Drum Default](instruments/drum_default.png)

### Piano — Default C Major

![Piano Default](instruments/piano_default.png)

### Piano — Custom JSON

![Piano Custom](instruments/piano_custom.png)

### All Available Drum Sounds (16종)

![All Drums](instruments/drum_all_instruments.png)

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 카메라, 손 추적(`--backend`), 관절선·손끝 궤적 표시 |
| `hand_tracker.py` | CPU(MediaPipe) / CPU-baseline(TFLite) / NPU(DX-RT `.dxnn`) 백엔드 — Palm TFLite(CPU) + Hand .dxnn(NPU) 또는 TFLite(CPU) |
| `strike_detector.py` | `InstrumentStrikeDetector` — 손끝 속도 + 관절 각속도 |
| `drumkit_audio.py` | 16종 합성 샘플, 손가락 슬롯에 매핑 |
| `tools/export_mediapipe_hand_onnx.py` | 공개 TFLite → ONNX + 레이아웃 생성 |
| `tools/export_mediapipe_palm_onnx.py` | Palm TFLite 추출 + ONNX 변환 시도 |
| `tools/palm_decode.py` | Palm SSD 앵커 생성 · box 디코드 · weighted NMS · letterbox 제거 |
| `tools/palm_letterbox.py` | Palm 192×192 입력 전처리 (keep aspect, zero pad) |
| `tools/palm_roi.py` | Palm keypoints → 회전 ROI → affine warp 224×224 (hand landmark 입력) |
| `tools/palm_mp_spec.py` | MediaPipe palm detection 그래프 상수 |
| `tools/smoke_palm_interpreter.py` | Palm TFLite I/O 스모크 테스트 |
| `tools/compile_dxnn.sh` | DX-COM 호출 래퍼 (`DX_COM` 환경변수 지원) |
| `tools/benchmark_dataset.py` | 저장된 `dataset/frame_*.png`로 백엔드 지연·landmark 오차 비교 |
| `tools/capture_dataset.py` | SPACE → delay → burst 방식의 데이터셋 캡처 |
| `tools/gen_instrument_diagrams.py` | 악기 매핑 다이어그램 PNG 생성 (matplotlib) |
| `instruments/` | 생성된 매핑 다이어그램 이미지 (drum, piano 등) |
