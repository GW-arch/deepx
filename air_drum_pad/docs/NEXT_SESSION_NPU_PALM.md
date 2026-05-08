# 다음 세션 실험 가이드 — Palm + Hand NPU 파이프라인

## 2026-05-08 추가 구현 결과

- `tools/benchmark_dataset.py` 추가: `dataset/frame_*.png`를 재생해 `cpu-baseline`/`npu-full` 지연, palm/hand 세부 시간, landmark 오차를 반복 측정.
- `FullNpuHandsTracker.last_profile` 및 `--palm-redetect-every N` 추가: 기본은 `0`(매 프레임 palm), `N>0`은 palm skip/ROI tracking 실험.
- `npu-full`의 palm 자동 탐색을 **TFLite 우선**으로 변경. Palm .dxnn은 score head 양자화 실패가 알려져 있으므로 `--palm-dxnn` 명시 시에만 실험적으로 사용.
- `--async-palm` 실험 옵션 추가: background palm + foreground ROI tracking.
- `tools/sweep_palm_redetect.py` 추가: `--palm-redetect-every` sweep CSV/JSON 생성.
- `tools/benchmark_dataset.py --debug-dir` 추가: landmark 오차 큰 프레임 overlay 자동 저장.
- `tools/capture_dataset.py`에 session/label/notes manifest 기록 추가.

빠른 재현:

```bash
cd ~/deepx/air_drum_pad
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --palm-redetect-every 5
python3 tools/sweep_palm_redetect.py --values 0,1,2,3,5,10 --backends cpu-baseline,npu-full
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --debug-dir /tmp/air_drum_debug
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --async-palm --frame-interval-ms 16.7
```

## 2026-04-17 세션 #4 결과

- **커밋:** 82aab86 (`main`) — `FullNpuHandsTracker` 가 `palm_dxnn_path` / `palm_tflite_path` 둘 다 지원
  - `--palm-dxnn` CLI 플래그 추가 (`main.py`)
  - 당시 `create_tracker()` 자동 탐색은 `.dxnn` → `.tflite` 우선순위였으나, 2026-05-08 이후 기본은 TFLite 우선으로 변경
  - Palm .dxnn 레이턴시: **12 ms** (vs TFLite CPU 95 ms) — 속도는 ~8× 빠름
  - **그러나 Palm .dxnn INT8 양자화로 score head 파괴** — ONNX↔NPU score 상관 -0.11, max sigmoid 0.01 vs ONNX 0.90
  - 시도한 조합: ema/minmax calibration, `--aggressive_partitioning` (0 CPU groups), `--opt_level 0`/`1` — 모두 실패
  - **결론: Palm detection은 TFLite (CPU, float32) 로 고정**, Hand landmark만 NPU
  - `_run_palm()` .dxnn 경로: NHWC uint8 입력 → `dx_engine.InferenceEngine.run()`
  - `close()` 에서 `palm_ie.dispose()` 호출
  - 디스플레이 `cv2.flip(frame, 1)` 추가 (셀카 미러)
  - `os.chdir(_SCRIPT_DIR)` 추가 (상대 경로 안정화)

## 2026-04-17 세션 #3 결과

- **커밋:** TBD (`main`)
- **완료:**
  - DX-COM 컴파일 성공 — SNU 서버(user12, taskset -c 12,13, opt_level 1)
  - `models/vendor/palm_detection_lite.dxnn` — 4.7 MB, .dxnn v8, DX-COM v2.1.0-rc.4
  - `parse_model` 확인: input `input_1` NCHW, outputs `Identity`(boxes) + `Identity_1`(scores)
  - 1 NPU task + 2 CPU tasks (concat), 48.5 MB memory, Div(x=255) NPU 내장
  - `models/dxnn_layout.mediapipe_palm_lite.json` 최종 확정 (nchw, div255 baked)
  - Phase 3 DX-COM 체크박스 완료 — PLAN 전 항목 ✓

## 2026-04-17 세션 #2 결과

- **커밋:** 948cfdf (`main`)
- **완료:**
  - Phase 1 단위 테스트 — `tools/test_palm_decode.py` 20/20 통과 (MediaPipe Hands wrist 비교 포함)
  - Phase 3 ONNX 변환 — `tools/dequant_palm_fp32.py`: flatc JSON 라운드트립으로 FP16→FP32 디퀀트 + DEQUANTIZE op 제거 → `tflite2onnx` 성공
  - `models/vendor/palm_detection_lite.onnx` — NCHW [1,3,192,192], 3.9 MB, TFLite 대비 max_diff=0.000122
- **보류:** DX-COM 미설치(install.sh 자격증명 필요) + SNU 컴파일 서버(비밀번호 필요) → palm `.dxnn` 빌드 불가

## 2026-04-17 세션 #1 결과

- **커밋:** `5403e83` (`main`)
- **완료:** Phase 1(palm_decode), Phase 2(palm_roi), Phase 3 layout JSON, Phase 4(FullNpuHandsTracker + CLI)
- **보류:** Palm ONNX 변환(tflite2onnx 실패, tf2onnx 필요) → palm `.dxnn` 빌드 불가

---

이 문서는 **다음 작업 세션**에서 바로 이어서 할 **실험·구현 순서**만 적습니다. 배경·아키텍처는 [`PLAN_NPU_FULL_HAND_PIPELINE.md`](PLAN_NPU_FULL_HAND_PIPELINE.md)를 보세요.

---

## 0. 세션 시작 시 확인

| 항목 | 확인 방법 |
|------|-----------|
| 작업 디렉터리 | `cd ~/deepx/air_drum_pad` (본인 경로에 맞게) |
| `dx_engine` | `python3 -c "import dx_engine; print(dx_engine.__version__)"` — DX-RT와 맞는 빌드(소스 `python_package` 권장) |
| NPU | `dxrt-cli -s` 정상, 필요 시 `dxrt-cli -r 0` |
| Palm TFLite | `ls models/vendor/palm_detection_lite.tflite` 없으면 아래 Step A |
| Hand `.dxnn` | `models/vendor/hand_landmark_lite.dxnn` (기존과 동일) |

---

## Step A — Palm 자산 준비 (5분)

```bash
cd ~/deepx/air_drum_pad
python3 tools/export_mediapipe_palm_onnx.py --variant lite
```

- 기대: `models/vendor/palm_detection_lite.tflite` 생성. ONNX는 `tflite2onnx` 실패가 **정상**(알려진 제약).
- 기록: 터미널에 찍힌 에러 한 줄을 실험 로그에 남기기.

---

## Step B — Palm TFLite I/O 스모크 (선택, TF/tflite_runtime 필요)

```bash
# 둘 중 하나 설치 후
python3 tools/smoke_palm_interpreter.py --variant lite
```

- 기대: `input` shape / `output` 텐서 개수·shape 출력, `invoke OK`.
- 기록: 입력 dtype이 `float32`인지 `uint8`인지 — `palm_letterbox.py` 출력 dtype 맞출 때 사용.

---

## Step C — Letterbox 전처리 스모크 (의존성 없음)

```bash
python3 tools/palm_letterbox.py
python3 tools/palm_letterbox.py --camera 0   # 카메라 있을 때
```

- 기대: `tensor (1, 192, 192, 3) float32`, `LetterboxPadding(...)` 출력.

---

## Step D — Phase 1 구현 ✅ 완료

`tools/test_palm_decode.py` 20/20 통과. `test_vs_mediapipe()` — 카메라 프레임에서 palm det score=0.865, MP wrist가 palm 박스 내부에 위치, kp0↔wrist 거리=0.0707.

---

## Step E — ONNX / `.dxnn`

- ✅ Palm ONNX: `tools/dequant_palm_fp32.py`로 변환 성공. 경로: `models/vendor/palm_detection_lite.onnx` (NCHW [1,3,192,192]).
- ✅ `dx_com`으로 `.dxnn` 빌드 완료 — SNU 서버 user12, `tools/compile_server_snu.sh`.
- ✅ `parse_model -m palm_detection_lite.dxnn` 확인 → 레이아웃 JSON 최종 확정 (`models/dxnn_layout.mediapipe_palm_lite.json`).

---

## Step F — Phase 2 (ROI → landmark)

- `hand_landmark_tracking` 그래프의 **손 ROI 워프** 규칙 문서/코드 조사 후, `DxnnHandTracker` 입력을 **전체 프레임이 아닌 ROI**로 제한하는 브랜치 설계.
- CPU MediaPipe vs NPU-only landmark **동일 프레임** 오차(손목 픽셀 거리) 측정 — [`EXPERIMENTS.md`](EXPERIMENTS.md) 지표 표에 한 줄 추가 가능.

---

## 세션 끝날 때 남길 것

- [x] `docs/PLAN_NPU_FULL_HAND_PIPELINE.md` 안 Phase 체크박스 갱신  
- [x] 본 파일(`NEXT_SESSION_NPU_PALM.md`) 상단에 **날짜·커밋 해시·실험 요약 3줄** 적기  
- [ ] 막힌 경우: 재현 커맨드 + 로그 스니펫을 이슈/메모에 붙여넣기

---

## 한 줄 요약

**현재 상태:** Phase 0~7 완료. Palm detection은 INT8 양자화 실패로 **TFLite (CPU)** 로 고정, Hand landmark는 **NPU (.dxnn)** 로 실행. `cpu-baseline` 백엔드 추가 — NPU 없이 동일 파이프라인을 CPU TFLite로 실행하는 비교 기준선. `_PALM_REDETECT_EVERY = 0` (매 프레임 palm 실행, ROI 드리프트 제거).

`--backend npu-full --palm-tflite …` 가 실사용 구성입니다. `--backend cpu-baseline`은 NPU 없이 비교용입니다.
