# AI Air-Drum Pad (prototype)

**관리 저장소 (SSH):** `git@github.com:GW-arch/deepx.git`
웹: [github.com/GW-arch/deepx](https://github.com/GW-arch/deepx)

## 동작 개요 (실제로 치는 것처럼)

손가락 끝을 **추적**하고, 아래 두 조건을 **동시에** 만족할 때만 한 번 친 것으로 봅니다.

1. **손끝 하강 속도** — 막대기 끝이 아래로 빠르게 움직임 (`vy`, 정규화 좌표/초)
2. **관절 각속도** — MCP–PIP–TIP(엄지는 IP 포함)에서 잰 각도가 프레임마다 충분히 변함 → 손가락 관절이 실제로 휘둘러짐

- **드럼 모드(기본)**: 화면에 그려진 **사각형 패드 영역** 안에서 어떤 손가락이든 내리치면 해당 패드의 드럼 소리가 납니다.
- **피아노 모드(`--piano`)**: **어느 손 × 어느 손가락**인지로 음을 매핑합니다.

### 드럼 패드 바꾸기

1. 사용 가능한 키: `python3 main.py --list-instruments`
2. `pads.example.json` 을 복사해 각 패드의 `label`, `sound`, `x1/y1/x2/y2`, `color`를 수정합니다. 좌표는 카메라 프레임 기준 정규화 좌표(0~1)입니다.
3. 실행: `python3 main.py --camera 0 --drum-pads my-pads.json`

**음색**을 바꾸려면 `drumkit_audio.py`의 `_KIT_BUILDERS`에 키를 추가·수정한 뒤 JSON에서 그 키를 쓰면 됩니다.

### 피아노 모드

- 실행: `python3 main.py --piano --camera 0`
  **`--instruments` 없이** 켜면 기본적으로 `instruments.piano.example.json`의 **고정 10키 매핑**을 사용합니다.
  기본값은 왼손 엄지→소지 **G4–C4**, 오른손 엄지→소지 **C5–G5**입니다.
- 다른 고정 음 배열을 쓰려면: `instruments.piano.example.json` 참고 후
  `python3 main.py --piano --instruments 내피아노.json --camera 0`
  (`slots` 값은 `C4`, `D#5`, `Bb3` 같은 **음명** 10개)
- 음색은 약 0.5초 길이의 **합성** 사인파(실제 샘플 피아노는 아님).
- 사용 가능 음명(기본 10개 나열): `python3 main.py --piano --list-instruments`

## 실행

```bash
cd air_drum_pad
pip3 install -r requirements.txt
```

**sudo password: `deepx123!`**

### 빠른 실행 (모드별)

```bash
# ── 1) 기본: NPU-full(Palm CPU TFLite + Hand NPU .dxnn) + guided-style windowed UI ──
python3 main.py --camera 0

# 피아노:
python3 main.py --piano --camera 0

# ── 2) CPU-baseline: Palm(CPU TFLite) + Hand(CPU TFLite) — NPU 없이 동일 커스텀 ROI 파이프라인 ──
python3 main.py --backend cpu-baseline --piano --camera 0

# ── 3) CPU MediaPipe: 데모 영상용 정확도 우선 fallback ──
python3 main.py --backend cpu --piano --camera 0

# ── 4) NPU dual-halves: 화면 반분할 근사 (가장 빠름, palm 없음) ──
python3 main.py --backend npu --piano --camera 0 \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite_dual.json

# ── 5) PINTO NPU 실험: Palm(CPU TFLite) + PINTO hand(.dxnn NPU) ──
python3 main.py --backend pinto-npu --piano --camera 0
```

### 데모 비디오용 쉬운 드럼 런처

`demo_run.py`는 드럼 모드를 켜고 배경 트랙을 함께 재생합니다. 데모에서는 복잡한 8패드 대신 큰 4패드(`kick`, `clap`, `snare`, `crash`) 레이아웃을 자동으로 사용합니다. 저작권 음악 파일은 포함하지 않으므로, 실제 Queen 곡을 쓰려면 본인이 가진 합법적인 오디오 파일을 지정하세요.

```bash
# dataset/ 안에 MP3/WAV/OGG가 하나 있으면 자동으로 사용
python3 demo_run.py --camera 0

# 또는 실제 배경음악 파일을 명시해서 실행
python3 demo_run.py --camera 0 --backing-track ~/Music/we_will_rock_you.mp3

# 오디오 파일이 없으면 royalty-free "stomp stomp clap" 가이드 루프를 자동 생성
# 가이드 생성도 끄려면:
python3 demo_run.py --camera 0 --no-guide

# 패턴만 확인
python3 demo_run.py --print-pattern
```

기본 데모 패턴은 80 BPM 기준:

```text
Count:  1   &   2   &   3   &   4   &
Main:   kick kick clap  -   kick kick clap  -

Tiny variation every 4th bar:
        kick kick clap  -   kick kick crash -
```

> **스크립트로 실행:** `scripts/run_cpu_piano.sh`, `scripts/run_npu_piano.sh`(dual-halves), `scripts/run_npu_full_piano.sh`(Palm+Hand 정식 파이프라인, correction 자동 적용 가능)를 사용하면 환경변수(`DISPLAY`, `XAUTHORITY`)를 자동으로 설정합니다.

### CPU vs CPU+NPU 비교

#### 백엔드별 모델 실행 위치

| `--backend` | Palm Detection | Hand Landmark | 비고 |
|-------------|----------------|---------------|------|
| `cpu` | CPU (MediaPipe 내장) | CPU (MediaPipe 내장) | 추가 파일 불필요, 데모 영상용 손끝 정확도 우선 |
| `cpu-baseline` | CPU (TFLite, float32) | CPU (TFLite, float32) | NPU 없이 npu-full과 동일 파이프라인 (비교 기준선) |
| `npu` | 없음 (dual-halves 근사) | **NPU** (.dxnn, int8) | palm 검출 없이 화면 좌우 반분할 |
| `npu-full` (기본) | CPU (TFLite, float32) | **NPU** (.dxnn, int8) | 안정적인 2-hand 파이프라인, guided-style UI 기본 |
| `npu-full --palm-dxnn ...local.dxnn` | **NPU** (.dxnn, int8) | **NPU** (.dxnn, int8) | DX-COM 2.3.0 재컴파일 palm 후보; offline replay 통과, live 재검증 필요 |
| `pinto-cpu` | CPU (TFLite, float32) | CPU (PINTO ONNX, float32) | PINTO sparse hand model 비교 기준 |
| `pinto-npu` | CPU (TFLite, float32) | **NPU** (PINTO .dxnn, int8) | PINTO sparse hand model NPU adoption 실험 |
| `pinto-npu --palm-dxnn ...local.dxnn` | **NPU** (.dxnn, int8) | **NPU** (PINTO .dxnn, int8) | 가장 빠른 full-NPU PINTO 후보; right-hand agreement 열세 |

> **Palm NPU 상태:** DX-COM 2.3.0으로 컴파일한 `palm_detection_lite_minmax_local.dxnn` / `palm_detection_lite_ema_local.dxnn`은 보드 offline replay에서 accepted palm을 생성하고 hand landmark stage까지 연결됩니다. 기본 live path는 안정성 검증 전까지 CPU palm을 유지하고, full-NPU palm은 `--palm-dxnn`으로 명시 실험합니다. 상세 기록: [`docs/SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md`](docs/SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md).

#### 성능 비교

| 구성 | Palm 단계 | Hand 단계 | 전체 | 비고 |
|------|----------:|----------:|-----:|------|
| `cpu` (MediaPipe) | MediaPipe internal | MediaPipe internal | 64.95 ms | 90-frame replay, high variance |
| `cpu-baseline` (TFLite palm + TFLite hand) | 40.29 ms | 45.72 ms | 86.42 ms | 90-frame replay, 비교 기준선 |
| `pinto-cpu` (TFLite palm + PINTO ONNX hand) | 38.59 ms | 50.59 ms | 89.58 ms | 90-frame replay |
| `npu-full` (TFLite palm + NPU hand) | 39.84 ms | 8.48 ms | 48.61 ms | 90-frame replay, live 기본 경로 |
| `pinto-npu` (TFLite palm + PINTO NPU hand) | 40.23 ms | 8.62 ms | 49.14 ms | 90-frame replay, 실험용 |
| `npu-full --palm-dxnn palm_detection_lite_minmax_local.dxnn` | 10.00 ms | 13.85 ms | 24.32 ms | 90-frame replay, full-NPU 후보 |
| `pinto-npu --palm-dxnn palm_detection_lite_minmax_local.dxnn` | 9.92 ms | 12.91 ms | 23.29 ms | 90-frame replay, 더 빠르지만 right-hand agreement 열세 |
| `npu` (dual-halves) | 0 ms | NPU-only hand pass | 8.42 ms | 90-frame replay, palm 검출 없음 |

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

`npu-full`은 `main.py` 기본 백엔드입니다. 기본 hand `.dxnn`, layout JSON, palm TFLite는 `models/vendor/`와 `models/`에서 자동으로 선택됩니다. Landmark correction은 camera/pose-specific이라 기본 비활성화되어 있으며, 필요할 때만 명시적으로 켭니다. 기본 화면은 selfie mirror이며, live UI는 guided evaluator와 같은 windowed PANDA title/yellow skeleton 스타일을 사용합니다. 데모 영상 촬영에서는 endpoint 정확도를 위해 `demo_run.py`가 별도로 `cpu` backend를 기본 사용합니다.

```bash
python3 main.py --backend npu-full --max-hands 2

# 명시적 실행
python3 main.py --backend npu-full \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite.json \
  --palm-tflite models/vendor/palm_detection_lite.tflite \
  --max-hands 2

# 보정이 실제 camera/pose에서 좋아지는지 확인한 뒤 opt-in 합니다.
python3 main.py --backend npu-full \
  --landmark-correction models/npu_landmark_correction.bias.json \
  --max-hands 2
```

`--palm-tflite` 생략 시 `models/vendor/palm_detection_lite.tflite` 자동 탐색. local 재컴파일 palm 후보를 검증할 때는 `--palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn`을 명시합니다.

#### 데이터셋 기반 오프라인 벤치마크

라이브 카메라 없이 `dataset/frame_*.png`를 반복 재생해 백엔드별 지연과 landmark 오차를 비교합니다.

```bash
# 기본: cpu-baseline vs npu-full, 동일 palm+ROI 파이프라인 비교
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full

# PINTO adoption 비교
python3 tools/benchmark_dataset.py --backends cpu-baseline,pinto-cpu,pinto-npu,npu-full --limit 90 --warmup 0

# local DX-COM 2.3.0 palm 재컴파일 후보 검증
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn \
  --limit 90 --warmup 0

# palm skip/ROI tracking 실험: palm 1회 후 최대 5프레임은 landmark ROI로 추적
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --palm-redetect-every 5

# palm skip 값을 한 번에 sweep
python3 tools/sweep_palm_redetect.py --values 0,1,2,3,5,10 \
  --backends cpu-baseline,npu-full --csv /tmp/palm_sweep.csv

# 오차가 큰 프레임 overlay 저장 (green=cpu-baseline, red=npu-full)
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --debug-dir /tmp/air_drum_debug --debug-top-k 10

# 실험용 async palm: palm은 백그라운드, hand는 이전 ROI로 계속 추적
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --async-palm --frame-interval-ms 16.7

# NPU landmark 보정 생성(CPU baseline 기준) 및 보정 적용 benchmark
python3 tools/calibrate_npu_landmarks.py \
  --kind bias \
  --output models/npu_landmark_correction.bias.json
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --landmark-correction models/npu_landmark_correction.bias.json

# 더 공격적인 affine 보정은 offline 분석용입니다. Live UI에서는 skeleton shape가
# 과도하게 왜곡될 수 있으므로 기본값으로 사용하지 않습니다.
python3 tools/calibrate_npu_landmarks.py \
  --output models/npu_landmark_correction.dataset.json
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --landmark-correction models/npu_landmark_correction.dataset.json

# CSV/JSON 저장
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --csv /tmp/air_drum_bench.csv --json /tmp/air_drum_bench.json
```

`--palm-redetect-every 0`이 기본값이며 매 프레임 palm detection을 실행합니다(드리프트 최소). `N>0`과 `--async-palm`은 지연을 줄이는 **실험 옵션**입니다.

`--landmark-correction`은 현재 dataset에서 학습한 NPU→CPU xy 보정입니다. 두 보정 파일 모두 dataset-specific이므로 다른 조명/카메라/손 자세에서는 별도 hold-out 검증이 필요합니다. 이 보드의 현재 replay 기준으로는 무보정이 전체 평균 오차가 가장 낮았고, 저장된 보정 파일은 특히 Right hand 후반 프레임에서 skeleton을 더 왜곡했습니다.

#### 품질 체크 / 회귀 테스트

```bash
# 문법 체크 + unittest + palm decode test + 짧은 dataset benchmark smoke
./scripts/check_quality.sh

# benchmark smoke가 오래 걸리거나 모델 파일이 없는 환경에서는 생략
RUN_BENCH_SMOKE=0 ./scripts/check_quality.sh

# 순수 단위 테스트만
python3 -m unittest discover -s tests -v
```

NPU 예시는 `models/README.md` 와 `scripts/run_npu_piano.sh` 참고.
요약: **MediaPipe TFLite → ONNX** (`tools/export_mediapipe_hand_onnx.py`) → **DX-COM** (SNU 서버 `tools/compile_server_snu.sh`) → 보드에서 `--backend npu-full --palm-tflite … --dxnn …`.

**`dx_engine`은 반드시 보드의 DX-RT와 맞는 빌드**를 쓰세요. 오래된 wheel만 설치하면 `InferenceEngine`이 멈추거나 힙 오류가 날 수 있습니다. 설치 절차는 `requirements-npu.txt` 주석을 따르면 됩니다.

- 종료: `q`
- 민감도: `--vy-trigger`(기본 0.025), `--joint-dps`(기본 16deg/s), `--cooldown`(기본 0.10초)

느리게만 움직이면 안 울리게 하려면 `--joint-dps`를 올리고, 너무 안 나오면 `--vy-trigger` / `--joint-dps`를 내립니다. 중지는 카메라상 관절 변화가 작게 잡히는 경우가 많아서 내부적으로 조금 더 민감하게 보정됩니다.

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
- [MIDTERM_PRESENTATION_2026.html](docs/MIDTERM_PRESENTATION_2026.html) / [PDF](docs/MIDTERM_PRESENTATION_2026.pdf) — 중간발표용 슬라이드
- [REPO_CAPABILITIES_SLIDES.html](docs/REPO_CAPABILITIES_SLIDES.html) — 레포 기능·실험 가능성 HTML 슬라이드
- [Palm+Hand NPU 파이프라인 계획](docs/PLAN_NPU_FULL_HAND_PIPELINE.md) — Phase 0~6 로드맵 + 체크박스
- [다음 세션 실험 가이드](docs/NEXT_SESSION_NPU_PALM.md) — 명령어·체크리스트·완료 기준

## 악기 매핑 다이어그램

`instruments/` 디렉터리에 각 프리셋별 UI 안내 이미지가 있습니다. 드럼은 사각형 패드 레이아웃, 피아노는 손가락별 음 매핑입니다.
재생성: `python3 tools/gen_instrument_diagrams.py`

### Drum Pads — Default

![Drum Default](instruments/drum_default.png)

### Piano — Default C Major

![Piano Default](instruments/piano_default.png)

### Piano — Custom JSON

![Piano Custom](instruments/piano_custom.png)

### All Available Drum Sounds (16종, pad keys)

![All Drums](instruments/drum_all_instruments.png)

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 카메라, 손 추적(`--backend`, 기본 `npu-full`), guided-style hand skeleton/live overlay 표시 |
| `hand_tracker.py` | CPU(MediaPipe) / CPU-baseline(TFLite) / NPU(DX-RT `.dxnn`) 백엔드 — Palm TFLite(CPU) + Hand .dxnn(NPU) 또는 TFLite(CPU) |
| `strike_detector.py` | `InstrumentStrikeDetector` / `PadStrikeDetector` — 손끝 속도 + 관절 각속도 |
| `drumkit_audio.py` | 16종 합성 드럼 샘플과 피아노 합성음 |
| `tools/export_mediapipe_hand_onnx.py` | 공개 TFLite → ONNX + 레이아웃 생성 |
| `tools/export_mediapipe_palm_onnx.py` | Palm TFLite 추출 + ONNX 변환 시도 |
| `tools/palm_decode.py` | Palm SSD 앵커 생성 · box 디코드 · weighted NMS · letterbox 제거 |
| `tools/palm_letterbox.py` | Palm 192×192 입력 전처리 (keep aspect, zero pad) |
| `tools/palm_roi.py` | Palm keypoints → 회전 ROI → affine warp 224×224 (hand landmark 입력) |
| `tools/palm_mp_spec.py` | MediaPipe palm detection 그래프 상수 |
| `tools/smoke_palm_interpreter.py` | Palm TFLite I/O 스모크 테스트 |
| `tools/compile_dxnn.sh` | DX-COM 호출 래퍼 (`DX_COM` 환경변수 지원) |
| `tools/benchmark_dataset.py` | 저장된 `dataset/frame_*.png`로 백엔드 지연·landmark 오차 비교, 오차 overlay 저장 |
| `tools/calibrate_npu_landmarks.py` | CPU baseline 기준 NPU landmark affine 보정 JSON 생성 |
| `tools/sweep_palm_redetect.py` | `--palm-redetect-every` 값을 sweep해 지연-오차 곡선 CSV/JSON 생성 |
| `tools/capture_dataset.py` | SPACE → delay → burst 방식의 데이터셋 캡처, manifest 기록 |
| `tools/gen_instrument_diagrams.py` | 악기 매핑 다이어그램 PNG 생성 (matplotlib) |
| `instruments/` | 생성된 매핑 다이어그램 이미지 (drum, piano 등) |
| `tests/` | strike detector, ROI transform, benchmark/sweep helper 단위 테스트 |
| `scripts/check_quality.sh` | 문법·단위·palm decode·benchmark smoke 통합 검사 |
