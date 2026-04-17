# 다음 세션 실험 가이드 — Palm + Hand NPU 파이프라인

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

## Step D — Phase 1 구현 (본 세션의 핵심 작업)

**목표:** `tools/palm_decode.py`에서 `NotImplementedError` 제거.

1. **앵커 생성**  
   - 참고: MediaPipe [`ssd_anchors_calculator.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc)  
   - 입력: `tools/palm_mp_spec.py`의 `SSD_*` 상수와 동일 옵션.

2. **TensorsToDetections**  
   - `DET_*` 상수 그대로 사용 (`palm_mp_spec.py` 옆 `palm_decode.py`).

3. **Weighted NMS**  
   - `NMS_MIN_SUPPRESSION_THRESHOLD = 0.3`, IoU.

4. **Letterbox 제거**  
   - `palm_letterbox.rgb_uint8_to_palm_input_tensor`가 반환하는 `LetterboxPadding`으로 detection 좌표를 **원본 이미지 정규화 [0,1]** 로 변환.

**검증 아이디어 (택1~복수):**

- 동일 RGB에 대해 **MediaPipe `Hands`** 한 번 돌리고, 내부 palm과 직접 비교는 어렵우므로  
  **눈으로**: 디코드된 박스를 `cv2.rectangle`으로 그려 저장 PNG 3장.  
- 또는 TFLite 출력 텐서를 **raw로 덤프**한 뒤, C++ MediaPipe 단위 테스트 수치가 있으면 대조.

**완료 기준:** 한 프레임에서 **≥1개** detection이 합리적인 위치(이미지 안, 손이 있는 영상 사용)에 나오고, 빈 배경에서는 점수 낮게 나오면 1차 성공.

---

## Step E — ONNX / `.dxnn` (호스트 또는 DX-COM 환경)

- Palm ONNX: `tflite2onnx` 실패 시 **tf2onnx / onnx 변환 경로** 조사 후 한 경로로 고정.
- `parse_model -m palm_detection_lite.dxnn`으로 입출력 기록.
- 레이아웃 초안: `models/dxnn_layout.mediapipe_palm_lite.json` (이름은 프로젝트 규칙에 맞게).

---

## Step F — Phase 2 (ROI → landmark)

- `hand_landmark_tracking` 그래프의 **손 ROI 워프** 규칙 문서/코드 조사 후, `DxnnHandTracker` 입력을 **전체 프레임이 아닌 ROI**로 제한하는 브랜치 설계.
- CPU MediaPipe vs NPU-only landmark **동일 프레임** 오차(손목 픽셀 거리) 측정 — [`EXPERIMENTS.md`](EXPERIMENTS.md) 지표 표에 한 줄 추가 가능.

---

## 세션 끝날 때 남길 것

- [ ] `docs/PLAN_NPU_FULL_HAND_PIPELINE.md` 안 Phase 체크박스 갱신  
- [ ] 본 파일(`NEXT_SESSION_NPU_PALM.md`) 상단에 **날짜·커밋 해시·실험 요약 3줄** 적기 (선택)  
- [ ] 막힌 경우: 재현 커맨드 + 로그 스니펫을 이슈/메모에 붙여넣기

---

## 한 줄 요약

**다음 세션:** Step D(`palm_decode.py` 완성) → 검증(시각화 또는 수치) → Step E(ONNX/`.dxnn`) 순으로 진행하면 됩니다.
