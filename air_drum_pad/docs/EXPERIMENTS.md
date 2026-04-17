# AI Air-Drum Pad — 실험 방법서

**문서 버전:** 1.0  
**저장소 (Git SSH):** `git@github.com:GW-arch/deepx.git` · [웹](https://github.com/GW-arch/deepx)

**Palm + Hand NPU 다음 세션 실험 절차:** [`NEXT_SESSION_NPU_PALM.md`](NEXT_SESSION_NPU_PALM.md) · 로드맵: [`PLAN_NPU_FULL_HAND_PIPELINE.md`](PLAN_NPU_FULL_HAND_PIPELINE.md)

---

## 1. 목표 지표

| 지표 | 단위 | 비고 |
|------|------|------|
| Inference Latency | ms | NPU/CPU 추론만 |
| E2E Latency | ms | 영상상 동작 ~ 스피커 파형 |
| Throughput | FPS | 비전 루프 |
| Power / FPS·W | W | 동일 부하 비교 |
| Hit Accuracy | % | 메트로놈 동기 실험 |

목표 참고: E2E **&lt; 15 ms**, Inference **&lt; ~2.5 ms/frame**(환경별), FPS **&gt; 60**, Accuracy **≥ 98%**(기획서 기준).

---

## 2. 환경 기록 (매 세션)

- 하드웨어: SBC, M1 펌웨어, 카메라 모델·해상도·FPS  
- 소프트웨어: OS, 커널, DX-RT/모델 버전, ALSA period  
- 실행 인자: `vy_trigger`, `cooldown`, Zone 좌표  
- 조도·배경·카메라 거리

---

## 3. E2E Latency

1. 고속 카메라(또는 ≥120 FPS)로 **손 + 스피커(또는 화면+오디오)** 동시 기록.  
2. 영상에서 타격 프레임, 오디오에서 첫 파형 샘플 식별.  
3. \( \Delta t = t_{\mathrm{audio}} - t_{\mathrm{video}} \), N≥30회, 평균·표준편차·P95.

---

## 4. Inference Latency·FPS

- 워밍업 후 N≥1000회 단일 프레임 추론 시간.  
- 앱 루프에서 `frames / time` — **동일 녹화 영상 입력**으로 CPU vs NPU 비교 권장.

---

## 5. 전력

- 동일 워크로드 T분, 평균 전력 \(P\), FPS \(F\) → **FPS/W**.

---

## 6. 정확도

- 메트로놈 템포 고정, 의도 타격 100회, 허용 시간창 ±W ms 내 TP.  
- FP(고스트 노트), FN(누락) 기록.

---

## 7. 재현성

- Git 커밋, 모델 체크섬, 원시 CSV 로그(`t`, `frame_id`, `infer_ms`, `trigger`) 보관.

---

## 8. 벤치마크 결과

### 8.1 환경

| 항목 | 값 |
|------|-----|
| Board | Orange Pi 5 Plus (RK3588, aarch64) |
| OS | Linux, Python 3.10.12 |
| DX-RT / dx_engine | 1.1.4 |
| DX-COM | v2.1.0-rc.4 (SNU 서버) |
| Camera | USB, 640×480 |
| Commit | `99f838d` |

### 8.2 Palm Detection — NPU (.dxnn) vs CPU (TFLite)

**재현 방법:**

```bash
cd ~/deepx/air_drum_pad
python3 -c "
import sys, time, cv2, numpy as np
sys.path.insert(0, 'tools')
from hand_tracker import FullNpuHandsTracker

hand_dxnn = 'models/vendor/hand_landmark_lite.dxnn'
hand_layout = 'models/dxnn_layout.mediapipe_hand_lite.json'

# NPU palm
t_npu = FullNpuHandsTracker(
    palm_dxnn_path='models/vendor/palm_detection_lite.dxnn',
    hand_dxnn_path=hand_dxnn, hand_layout_path=hand_layout, max_hands=2)

# TFLite palm
t_tfl = FullNpuHandsTracker(
    palm_tflite_path='models/vendor/palm_detection_lite.tflite',
    hand_dxnn_path=hand_dxnn, hand_layout_path=hand_layout, max_hands=2)
t_npu._ensure_imports(); t_tfl._ensure_imports()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(5): cap.read()
_, frame = cap.read(); cap.release()
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Warm up
for _ in range(3): t_npu.process(rgb); t_tfl.process(rgb)

# Timed runs
for label, tracker in [('NPU', t_npu), ('TFL', t_tfl)]:
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        tracker.process(rgb)
        times.append((time.perf_counter() - t0) * 1000)
    print(f'{label}: avg={np.mean(times):.1f}ms min={np.min(times):.1f}ms max={np.max(times):.1f}ms')
t_npu.close(); t_tfl.close()
"
```

**결과 (2026-04-17, commit `99f838d`):**

| 측정 대상 | Palm NPU (.dxnn) | Palm CPU (TFLite XNNPACK) | 배수 |
|-----------|----------------:|------------------------:|-----:|
| Palm detection only | **12.2 ms** | 95.1 ms | ~8× |
| Full pipeline (palm+hand, 10-frame avg) | **7.3 ms** | 43.1 ms | ~6× |
| Full pipeline min | 6.8 ms | 40.5 ms | — |
| Full pipeline max | 8.0 ms | 45.0 ms | — |

> **Note:** "Full pipeline" 시간이 "palm only" 보다 짧은 이유: 손이 감지되지 않은 프레임에서는
> hand landmark 추론이 스킵되므로 palm detection 이 거의 전부.
> 캐시 워밍업 후 palm NPU 추론 자체는 ~7 ms 수준.

### 8.3 npu-full End-to-End (손 감지 시)

이전 세션 (#1, TFLite palm + NPU hand):

| 조건 | 시간 | 손 수 |
|------|-----:|------:|
| Palm TFLite + Hand .dxnn | 99.4 ms | 2 |

현재 세션 (#4, NPU palm + NPU hand):

| 조건 | 시간 (추정) | 비고 |
|------|----------:|------|
| Palm .dxnn + Hand .dxnn | ~20–30 ms | palm 12ms + hand ~8ms/hand + CPU glue |

> 실제 2-hand 감지 시 벤치마크는 다음 세션에서 카메라 앞 손 촬영으로 측정 예정.

---

## 9. 참조

- 아키텍처: `docs/ARCHITECTURE.md`
