# 계획: Palm(NPU) + Hand landmark(NPU) — MediaPipe Hands 동급 파이프라인

**다음 세션에서 바로 실험할 순서:** [`NEXT_SESSION_NPU_PALM.md`](NEXT_SESSION_NPU_PALM.md) (Step A~F, 명령어·완료 기준).

## 목표

| 단계 | 내용 |
|------|------|
| 최종 | 손 **검출(박스/키포인트)** 과 **21 랜드마크**를 모두 `.dxnn` + DX-RT로 돌리고, MediaPipe `hands` 그래프와 동일한 **전처리·앵커·NMS·letterbox 제거** 규칙을 Python에서 재현한다. |
| 중간 | Palm은 TFLite(CPU)만 검증 → ONNX/DX-COM 성공 시 NPU로 이전. |

## 아키텍처 (MediaPipe `palm_detection_cpu.pbtxt` 기준)

```mermaid
flowchart LR
  subgraph cpu_glue["CPU 글루 (반드시 필요)"]
    A[RGB 프레임]
    B[ImageToTensor 192x192 letterbox float 0..1]
    C[Detection letterbox 제거]
    D[ROI → affine warp]
    E[Hand landmark 입력 224x224]
  end
  subgraph npu["NPU (.dxnn)"]
    P[Palm SSD]
    L[Hand landmark]
  end
  A --> B --> P
  P -->|raw tensors| Decode[NMS·박스·키포인트 디코드]
  Decode --> C --> D --> E --> L
```

- **Palm 모델**: `192×192` 입력, SSD식 출력 → **앵커 2016개**, `TensorsToDetections` 옵션은 저장소 `tools/palm_mp_spec.py`에 고정값으로 둠 (MediaPipe 상수와 동일).
- **Hand landmark 모델**: 기존 `hand_landmark_lite.dxnn` — Palm이 만든 **손 ROI**에 맞춰 크롭·워프한 뒤 넣어야 CPU MediaPipe와 유사한 품질이 난다.

## 단계별 실행 계획

### Phase 0 — 완료 (이번 커밋)

- [x] `tools/palm_mp_spec.py`: MediaPipe `palm_detection_cpu.pbtxt`와 동일한 **SSD / 디코딩 / NMS / letterbox 입력** 상수.
- [x] `tools/palm_letterbox.py`: `ImageToTensorCalculator`에 대응하는 **192×192, keep aspect, [0,1], zero pad**.
- [x] `tools/export_mediapipe_palm_onnx.py`: pip `mediapipe` 번들에서 `palm_detection_{lite,full}.tflite` 복사 + `tflite2onnx` 시도 + ONNX I/O 출력(성공 시).
- [x] `tools/smoke_palm_interpreter.py`: `tensorflow` 또는 `tflite_runtime` 있으면 Palm TFLite **입출력 shape** 스모크.
- [x] 본 문서.

### Phase 1 — Palm 출력 디코드 (CPU, NumPy)

- [x] `tools/palm_decode.py`: `SsdAnchorsCalculator`와 동등한 **앵커 생성** (C++ `ssd_anchors_calculator.cc` 이식).
- [x] `TensorsToDetectionsCalculator` 규격 반영: `num_boxes=2016`, `num_coords=18`, `keypoint_coord_offset=4`, `num_keypoints=7`, `reverse_output_order=true`, `sigmoid_score=true`, 스케일 192 등 (`palm_mp_spec.py` 참조).
- [x] `NonMaxSuppressionCalculator` 규화: IoU, `min_suppression_threshold=0.3`, WEIGHTED.
- [x] `DetectionLetterboxRemovalCalculator`: `palm_letterbox`가 돌려준 padding 메타로 박스를 **원본 이미지 정규 좌표**로 되돌림.
- [x] 단위 테스트: `tools/test_palm_decode.py` — 20/20 통과. MediaPipe `Hands` wrist 위치가 우리 palm 박스 안에 포함, 키포인트 거리 0.07 이내.

### Phase 2 — Hand ROI → landmark 입력

- [x] MediaPipe `hand_landmark_tracking` 그래프의 **RectTransformation** / **warp** 규칙 이식 (회전·스케일·종횡비). `tools/palm_roi.py` — Palm의 7개 키포인트로 손 사각형을 정의, affine warp 224×224, 역변환.
- [x] 기존 `DxnnHandTracker`가 받는 크롭을 "전체 프레임"이 아니라 **위 ROI**로 제한 — `FullNpuHandsTracker`에서 사용.

### Phase 3 — Palm `.dxnn`

- [x] ONNX 변환: `tools/dequant_palm_fp32.py` — flatc JSON 라운드트립으로 FP16→FP32 디퀀트 + DEQUANTIZE op 제거 → `tflite2onnx` 성공. `models/vendor/palm_detection_lite.onnx` (NCHW, 3.9 MB). TFLite 대비 max_diff=0.000122.
- [x] `dx_com`으로 `palm_detection_lite.dxnn` 빌드 완료 (SNU 서버 user12, opt_level 1). `parse_model` 확인: input `input_1` NCHW, outputs `Identity`(boxes) + `Identity_1`(scores), 1 NPU + 2 CPU tasks, 48.5 MB. Div(x=255) NPU 내장.
- [x] `models/dxnn_layout.mediapipe_palm_lite.json` 초안.

### Phase 4 — 통합 트래커

- [x] `hand_tracker.py`에 `FullNpuHandsTracker`: Palm TFLite(CPU) + landmark `.dxnn`(NPU), Palm `.dxnn` 확보 시 NPU 교체 가능.
- [x] `main.py --backend npu-full` + `--palm-tflite` 플래그.

### Phase 5 — 정리

- [x] README 업데이트: `npu-full` 사용법, 도구 목록, 구현 현황 → PLAN 체크박스로 이전.

## 리스크·메모

- **`tflite2onnx`로 palm Tflite 변환**: 현재 보드에서 **IndexError 등으로 실패**할 수 있음 → Phase 3에서 다른 변환기 필수일 수 있음.
- **Palm Tflite**: `pip show mediapipe` 설치 경로의 `mediapipe/modules/palm_detection/palm_detection_lite.tflite` (스크립트가 자동 복사).
- **DX-COM**: Palm 그래프에 **커스텀 op / FP16** 등이 있으면 컴파일 제약이 있을 수 있음 — 컴파일 로그로 op 단위 확인.

## 참고 링크

- [palm_detection_cpu.pbtxt](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt)
- [ssd_anchors_calculator.cc](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc) (앵커 생성)
