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
  **`--instruments` 없이** 켜면: **양손 손목 사이 거리**로 음역을 잡습니다.  
  - 손을 **가깝게** 두면 → 왼손·오른손 모두 **중음역**에 가깝게(겹치는 느낌).  
  - 손을 **멀리** 벌리면 → **왼손(Left)은 더 낮은 펜타토닉**, **오른손(Right)은 더 높은 펜타토닉**으로 벌어집니다.  
  - MediaPipe **Left / Right** 라벨로 좌우를 구분합니다(셀카 미러면 체감이 반대일 수 있음).  
  - 화면 아래에 `d=… L:… R:…` 힌트가 뜹니다.
- **고정 음 배열**을 쓰려면: `instruments.piano.example.json` 참고 후  
  `python3 main.py --piano --instruments 내피아노.json --camera 0`  
  (`slots` 값은 `C4`, `D#5`, `Bb3` 같은 **음명** 10개 — 이 경우 거리 자동 음역은 끔)
- 음색은 짧은 **합성** 사인파(실제 샘플 피아노는 아님).
- 사용 가능 음명(기본 10개 나열): `python3 main.py --piano --list-instruments`

## 실행

```bash
cd air_drum_pad
pip3 install -r requirements.txt
python3 main.py --camera 0
# 피아노:
# python3 main.py --piano --camera 0
```

### CPU vs NPU (손 추론)

| `--backend` | 설명 |
|-------------|------|
| `cpu` (기본) | MediaPipe Hands |
| `npu` | DX-RT `.dxnn` + `dx_engine` (레이아웃 JSON) — hand landmark만 NPU, palm 검출 없음(dual-halves 근사) |
| `npu-full` | Palm detection + Hand landmark — 전부 NPU `.dxnn` (또는 palm만 TFLite CPU 폴백) — 정식 2-hand 파이프라인 |

**npu-full 예시 (전부 NPU):**

```bash
python3 main.py --backend npu-full \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite.json \
  --palm-dxnn models/vendor/palm_detection_lite.dxnn \
  --max-hands 2
```

`--palm-dxnn` / `--palm-tflite` 생략 시 자동 탐색 순서: `.dxnn` → `.tflite` (둘 다 `models/vendor/`).

**CPU 폴백 (palm만 TFLite):**

```bash
python3 main.py --backend npu-full \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --palm-tflite models/vendor/palm_detection_lite.tflite \
  --max-hands 2
```

| Palm 모델 | 경로 플래그 | 추론 장치 | Palm 전용 레이턴시 |
|-----------|------------|-----------|-------------------|
| `.dxnn` | `--palm-dxnn` | NPU | ~12 ms |
| `.tflite` | `--palm-tflite` | CPU (XNNPACK) | ~95 ms |

NPU 예시는 `models/README.md` 와 `scripts/run_npu_piano.sh` 참고.  
요약: **MediaPipe TFLite → ONNX** (`tools/export_mediapipe_hand_onnx.py`) → **DX-COM** (로컬 `tools/compile_dxnn.sh` 또는 SNU 서버 `tools/compile_server_snu.sh`) → 보드에서 `--backend npu --dxnn …`.

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
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)
- [Palm+Hand NPU 파이프라인 계획](docs/PLAN_NPU_FULL_HAND_PIPELINE.md) — Phase 0~5 로드맵 + 체크박스  
- [다음 세션 실험 가이드](docs/NEXT_SESSION_NPU_PALM.md) — 명령어·체크리스트·완료 기준

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 카메라, 손 추적(`--backend`), 관절선·손끝 궤적 표시 |
| `hand_tracker.py` | CPU(MediaPipe) / NPU(DX-RT `.dxnn`) 랜드마크 |
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
