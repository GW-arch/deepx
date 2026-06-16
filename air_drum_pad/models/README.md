# 모델 파이프라인: Palm Detection + Hand Landmark → NPU

이 디렉터리는 **레이아웃 JSON**과 문서를 두고, 실제 **TFLite / ONNX / .dxnn** 대용량 파일은 기본적으로 `vendor/`에 두며(`.gitignore`로 onnx 등 생략 가능) 필요 시 스크립트로 생성합니다.

## 한 줄 요약

1. **Google MediaPipe 공개 TFLite** (`hand_landmark_lite.tflite`, `palm_detection_lite.tflite`) 다운로드  
2. **`tflite2onnx`** 로 ONNX 변환 (`tools/export_mediapipe_hand_onnx.py`, `tools/export_mediapipe_palm_onnx.py`)  
3. **DX-COM**(DEEPX SDK)으로 `.dxnn` 컴파일 — `tools/compile_server_snu.sh all` 또는 `tools/compile_dxnn.sh`  
4. 보드에서 **`python3 main.py --backend npu-full --palm-tflite … --dxnn … --dxnn-layout …`** 또는 **`python3 main.py --backend cpu-baseline`** (모델 자동 탐색)

MediaPipe **앱 전체**가 아니라, 그 안의 **신경망 파일**을 ONNX로 옮긴 뒤 DXNN으로 빌드하는 흐름입니다.

## 파이프라인 구조 (`npu-full` / `cpu-baseline` 백엔드)

```
카메라 프레임 (RGB)
  ├─ Palm Detection                         ← 192×192 NHWC
  │    ├─ default npu-full: TFLite, CPU float32
  │    └─ experimental full-NPU: local DX-COM .dxnn, NPU int8
  │    └─ 2016 SSD anchors → NMS → palm bounding box
  ├─ ROI crop + letterbox → 224×224 패치
  └─ Hand Landmark                          ← 224×224 NHWC
       ├─ npu-full: .dxnn (NPU int8, uint8 입력)
       ├─ cpu-baseline: TFLite (CPU float32, float32 입력)
       ├─ pinto-npu: PINTO .dxnn (NPU int8, uint8 입력)
       ├─ pinto-cpu: PINTO ONNX (CPU float32 입력)
       └─ 21 keypoints + handedness + score
```

기본 live path는 Palm detection을 **TFLite (CPU)** 로 실행합니다. DX-COM 2.3.0으로 컴파일한 palm 후보(`palm_detection_lite_minmax_local.dxnn`, `palm_detection_lite_ema_local.dxnn`)는 board offline replay에서 accepted detections를 만들고 hand landmark stage까지 연결됩니다. Hand landmark는 **NPU .dxnn** 으로 실행합니다. 상세 기록은 [`docs/SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md`](../docs/SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md)에 둡니다.

> **왜 양자화가 필요한가?** DeepX NPU 하드웨어는 **INT8 연산만 지원**하는 고정 기능 가속기입니다. NPU로 실행하려면 반드시 float32→INT8 양자화가 필요합니다. 양자화 없이 float32로 돌리려면 CPU(TFLite)를 써야 합니다. NPU의 장점은 속도(INT8 conv를 ARM CPU 대비 ~10-50x 빠르게 처리)이므로, **정확도가 살아있는 모델은 NPU로, 양자화에 취약한 모델은 CPU로** 나누는 하이브리드가 최선입니다.

Palm detection → ONNX export·전처리 상수·SSD anchor 디코딩은 `tools/export_mediapipe_palm_onnx.py`, `tools/palm_mp_spec.py`, `tools/palm_letterbox.py`, `tools/palm_decode.py` 를 참고하세요.  
계획 문서: [`docs/PLAN_NPU_FULL_HAND_PIPELINE.md`](../docs/PLAN_NPU_FULL_HAND_PIPELINE.md)

## 1) ONNX 만들기 (호스트)

```bash
cd air_drum_pad
pip3 install -r requirements-export.txt
python3 tools/export_mediapipe_hand_onnx.py --variant lite --dual-halves-layout
# 결과: models/vendor/hand_landmark_lite.onnx
#       models/vendor/dxnn_layout.mediapipe_hand_lite*.json
# 레이아웃만 저장소에 두려면 models/dxnn_layout.mediapipe_hand_lite*.json 을 참고(이미 동기화됨)
```

- **`--variant full`**: 더 큰 모델(지연↑, 정확도↑).  
- **`--dual-halves-layout`**: 화면 좌·우 반으로 나눠 **각각 한 번씩** 추론하는 설정 JSON(`*_dual.json`) — `--max-hands 2` 용 **근사 양손**(손바닥 검출 없음).  
- ONNX I/O는 `export` 스크립트가 stdout에 찍습니다. 기본: 입력 `input_1` `[1,3,224,224]` NCHW float, 출력 `Identity` `[1,63]` + `Identity_1` 손 존재 점수 등.

## 2) DX-COM으로 `.dxnn` (호스트 또는 크로스 컴파일 환경)

### SNU 실습 컴파일 서버 (강의 환경)

[실습5 문서 (DX-AS NPU 컴파일)](https://sites.google.com/view/dxs-2603-snu/home/%EC%8B%A4%EC%8A%B55-dx-as-npu-%EC%BB%B4%ED%8C%8C%EC%9D%BC-%EB%B0%8F-%EC%B6%94%EB%A1%A0-%EC%8B%A4%EC%8A%B5) 기준:

| 항목 | 값 |
|------|-----|
| SSH | `ssh -p 443 user12@43.203.143.33` |
| SCP | `scp -P 443 …` (대문자 **P**) |
| PWD | `snu*npu&&` |
| 샘플 ONNX/JSON | 서버 `~/sample/` |
| 컴파일 결과 | 서버 `~/output/*.dxnn` |
| 컴파일 명령 | `taskset -c N,N+1 dx_com -m <onnx> -c <json> -o ~/output` — **userN 이면 N,N+1** (예: user12 → `taskset -c 12,13`) |

교육키트(Orange Pi)에서 한 번에 올리고 받기:

```bash
cd air_drum_pad
export DX_COMPILE_USER=user12
# 권장: ssh-copy-id 로 공개키 등록 후 비밀번호 없이 접속
# 또는: echo '비밀번호한줄' > ~/.snupass && chmod 600 ~/.snupass && export DX_COMPILE_PASSFILE=~/.snupass

# Hand landmark 컴파일 (기본)
./tools/compile_server_snu.sh all

# Palm detection 컴파일 (참고용 — 양자화 문제로 현재 TFLite 사용)
DX_COMPILE_PASSFILE=/tmp/.snupass \
  COMPILE_ONNX=models/vendor/palm_detection_lite.onnx \
  COMPILE_JSON=models/dxcom/palm_detection_lite.json \
  COMPILE_OUT_NAME=palm_detection_lite.dxnn \
  bash tools/compile_server_snu.sh all
```

레이아웃 JSON(`models/dxnn_layout.mediapipe_hand_lite*.json`)은 **`parse_model -m …`로 확인한 출력 순서**에 맞춥니다.  
MediaPipe 손 ONNX→DX-COM 결과가 흔히 **`Identity` [1,63] 랜드마크 + `Identity_1`/`Identity_2` 스칼라** 형태이므로, 기본 레이아웃은 `landmarks_tensor_index: 0` 과 `outputs.coordinate_space: "letterbox_patch_pixels"`(224 패치 픽셀→원본 RGB 정규화 역변환)을 씁니다.  
이미 정규화된 다른 텐서만 쓰는 컴파일본이면 `coordinate_space` 를 `"normalized"` 로 두고 `landmarks_tensor_index` 만 바꾸면 됩니다.

DX-COM용 보정 설정:
- `models/dxcom/hand_landmark_lite.json` — hand landmark 모델 (서버 `~/sample` 과 동일)
- `models/dxcom/palm_detection_lite.json` — palm detection 모델 (ema calibration)
- `models/dxcom/palm_detection_lite_minmax.json` — palm detection 모델 (minmax calibration)

### aarch64 보드(Orange Pi 등)

`~/dx-all-suite/dx-compiler/install.sh` 기준 **DX-COM(dxcom) 공식 설치는 amd64 / x86_64만 지원**합니다.  
이 보드에서는 **ONNX → `.dxnn` 컴파일을 할 수 없고**, x86_64 PC(또는 해당 아키텍처 환경)에서 컴파일한 `hand_landmark_lite.dxnn` 파일만 복사해 오면 됩니다.  
**DX-RT**(`dx_engine`, `parse_model`, `run_model`)는 aarch64에 설치된 상태로 `--backend npu` 실행에 사용합니다.

DX-COM CLI 이름은 설치본마다 다릅니다. 스크립트는 `dxcom`(dx-all-suite venv) / `dx_com` 등을 찾고, 없으면 **`DX_COM` 환경변수**로 전체 명령을 넘기면 됩니다.

```bash
export DX_COM='dx_com --your-flags-here'   # 예시 — 실제 플래그는 DEEPX 매뉴얼 따름
./tools/compile_dxnn.sh models/vendor/hand_landmark_lite.onnx models/vendor/hand_landmark_lite.dxnn
```

컴파일 시 **INT8 양자화·입출력 이름 고정** 등은 DX-COM 설정 JSON에서 지정합니다(제품 매뉴얼).

## 3) 보드에서 실행

1. [dx_rt](https://github.com/DEEPX-AI/dx_rt) **`python_package`를 보드에서 빌드·설치** → `dx_engine` 버전이 설치된 **DX-RT(`libdxrt.so`)와 일치**해야 합니다. 구 pip에서는 `pip install .` 이 실패할 수 있으므로 `requirements-npu.txt` 안내를 따르세요.  
2. `pip3 install -r requirements-npu.txt` (MediaPipe 없이 실행 가능)  
3. 예:

```bash
export DXNN=/path/to/hand_landmark_lite.dxnn
./scripts/run_npu_piano.sh
# 또는 정식 Palm+Hand 파이프라인(+보정 JSON 자동 적용)
./scripts/run_npu_full_piano.sh
# 또는 (npu-full: palm TFLite + hand NPU)
python3 main.py --backend npu-full \
  --palm-tflite models/vendor/palm_detection_lite.tflite \
  --dxnn "$DXNN" \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite.json \
  --max-hands 2 --piano --camera 0
# cpu-baseline: palm TFLite + hand TFLite (모두 CPU, 비교 기준선)
python3 main.py --backend cpu-baseline --max-hands 2 --piano --camera 0
# pinto-npu: palm TFLite + PINTO hand landmark DXNN (실험용)
python3 main.py --backend pinto-npu --max-hands 2 --piano --camera 0
# pinto-cpu: palm TFLite + PINTO hand landmark ONNX (비교 기준)
python3 main.py --backend pinto-cpu --max-hands 2 --piano --camera 0
# npu 백엔드 (dual-halves, palm detection 없음)
python3 main.py --backend npu \
  --dxnn "$DXNN" \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite_dual.json \
  --max-hands 2 --piano --camera 0
```

모델 정보 확인: `parse_model -m ./hand_landmark_lite.dxnn` (DX-RT 도구).

### NPU landmark 보정(JSON)

INT8 NPU hand landmark가 CPU TFLite(float32) 대비 일정한 xy 편향을 보일 때, 저장된 dataset으로 affine 보정을 fit할 수 있습니다.

```bash
python3 tools/calibrate_npu_landmarks.py \
  --output models/npu_landmark_correction.dataset.json

python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --landmark-correction models/npu_landmark_correction.dataset.json

python3 main.py --backend npu-full \
  --palm-tflite models/vendor/palm_detection_lite.tflite \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite.json \
  --landmark-correction models/npu_landmark_correction.dataset.json
```

현재 저장소의 `models/npu_landmark_correction.dataset.json`은 `dataset/` 90프레임으로 fit한 예시입니다. Dataset-specific calibration이므로 새 조명/카메라/손 자세에서는 별도 hold-out 검증이 필요합니다. Live 기본 실행은 보정 없이 시작하고, 위 벤치마크로 실제 환경에서 오차가 줄어드는 것을 확인한 뒤 `--landmark-correction`을 opt-in 하세요.

## 레이아웃 JSON 키 요약

| 키 | 의미 |
|----|------|
| `input.tensor_layout` | `nchw` / `nhwc` / `auto` |
| `input.square_pad` | ROI를 정사각으로 패딩 후 224 리사이즈(반쪽 화면 모드에서 권장) |
| `inference.dual_horizontal_halves` | `true`이면 좌·우 반 각각 추론 후 x 좌표를 전체 화면으로 합침 |
| `outputs.landmarks_tensor_index` | 63 floats 랜드마크 텐서 인덱스 |
| `confidence.tensor_index` | 손 존재 점수(예: MediaPipe ONNX의 두 번째 출력). `null`이면 게이트 없음 |

## Palm NPU 컴파일 결과와 한계

Palm detection ONNX export는 `tools/export_mediapipe_palm_onnx.py` 로 정상 동작합니다. DX-COM 2.3.0에서 dataset calibration path를 `dataset/`으로 맞춰 컴파일한 두 후보는 보드 검증을 통과했습니다. checksum, compile commands, tensor correlation, and benchmark details are recorded in [`docs/SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md`](../docs/SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md).

| 후보 | `frame_000` DXNN detections | score corr vs TFLite | `frame_060` DXNN detections | score corr vs TFLite |
|------|----------------------------:|---------------------:|----------------------------:|---------------------:|
| `palm_detection_lite_minmax_local.dxnn` | 2 | 0.9809 | 2 | 0.9780 |
| `palm_detection_lite_ema_local.dxnn` | 2 | 0.9809 | 2 | 0.9780 |

90-frame replay with `palm_detection_lite_minmax_local.dxnn`:

| Backend | Mean | P95 | Profile |
|---------|-----:|----:|---------|
| `cpu-baseline` | 86.42 ms | 87.31 ms | palm 40.29 ms + hand 45.72 ms |
| `npu-full --palm-dxnn palm_detection_lite_minmax_local.dxnn` | 24.32 ms | 29.71 ms | palm 10.00 ms + hand 13.85 ms |

**현재 해법**: 안정적인 live 기본값은 CPU palm + NPU hand입니다. 가장 빠른 meaningful-accuracy 후보는 재컴파일한 palm `.dxnn`을 명시하는 full-NPU path입니다:

```bash
python3 main.py --backend npu-full \
  --palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn \
  --dxnn models/vendor/hand_landmark_lite.dxnn \
  --dxnn-layout models/dxnn_layout.mediapipe_hand_lite.json \
  --max-hands 2 --piano --camera 0
```
