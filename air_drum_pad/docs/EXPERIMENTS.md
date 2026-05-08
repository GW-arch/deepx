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

> **결론:** Palm .dxnn은 속도는 빠르나 INT8 양자화로 **score head가 파괴**되어 (ONNX↔NPU 상관 -0.11) 실제 손 감지가 불가합니다. 아래 레이턴시 수치는 참고용이며, **실사용은 TFLite (CPU)** 입니다.

**Palm Detection 레이턴시 (참고):**

| Palm 백엔드 | 추론 시간 | 비고 |
|-------------|----------:|------|
| NPU (.dxnn) | ~12 ms | score 파괴로 **사용 불가** |
| CPU (TFLite XNNPACK) | ~95 ms | **실사용** (float32, 정확) |

### 8.3 End-to-End 레이턴시 비교

| 백엔드 | Palm | Hand (per hand) | 전체 (2 hands) | 비고 |
|---------|-----:|----------------:|---------------:|------|
| `cpu` (MediaPipe) | ~15 ms | ~10 ms | ~35 ms | 모두 float32, 추가 파일 불필요 |
| `cpu-baseline` (TFLite) | ~95 ms | ~5 ms | **~105 ms** | float32, npu-full 비교 기준선 |
| `npu-full` (TFLite+NPU) | ~95 ms | ~8 ms | **~111 ms** | 매 프레임 palm 실행 |
| `npu` (dual-halves) | 0 ms | ~8 ms × 2 | **~16 ms** | palm 검출 없음, 근사 |

> **npu-full / cpu-baseline**: 매 프레임 palm detection을 실행합니다 (`_PALM_REDETECT_EVERY = 0`). 이전에는 landmark 기반 ROI 트래킹으로 palm skip(5프레임에 1번)을 사용했으나, NPU INT8 양자화 편향 누적으로 ROI 드리프트(최대 dy=0.26)가 발생하여 비활성화했습니다.
> 
> **cpu-baseline vs npu-full**: 동일 파이프라인이지만 hand landmark를 CPU TFLite(float32) vs NPU .dxnn(INT8)로 실행. palm이 전체 시간의 ~90%를 차지하므로 NPU 가속 효과는 hand landmark 단독으로 비교해야 합니다 (~5ms TFLite vs ~8ms NPU).

> **Palm skip 최적화는 실험 옵션입니다.** 기본값은 정확도 우선(`--palm-redetect-every 0`, 매 프레임 palm)입니다. `--palm-redetect-every 5`처럼 지정하면 이전 프레임 랜드마크에서 다음 ROI를 예측해 palm detection을 건너뛰며, 지연은 줄지만 NPU INT8 편향이 누적될 수 있어 dataset benchmark로 오차를 반드시 확인합니다.

### 8.4 Offline dataset benchmark (2026-05-08)

도구:

```bash
cd air_drum_pad
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --palm-redetect-every 5
```

현재 `dataset/` 90프레임에서 확인한 예시 결과(Orange Pi 5 Plus, DX-RT/dx_engine 1.1.4):

| 구성 | 평균 | P95 | 세부 프로파일 | 비고 |
|------|-----:|----:|----------------|------|
| `cpu-baseline`, palm every frame | 84.40 ms | 86.78 ms | palm 39.26 ms + hand 44.80 ms | CPU TFLite 기준 |
| `npu-full`, palm every frame | 50.32 ms | 54.52 ms | palm 40.93 ms + hand 9.13 ms | 같은 palm+ROI, hand만 NPU |
| `npu-full`, `--palm-redetect-every 5` (20프레임 smoke) | 15.29 ms | 50.96 ms | palm frames 3/20, tracking frames 17/20 | 지연 개선, 정확도 회귀 확인 필요 |

`npu-full` vs `cpu-baseline` landmark 오차(90프레임, normalized xy):

| 손 | 매칭 프레임 | 전체 21점 평균 | 손끝 5점 평균 | 평균 max | 최대 max |
|----|-----------:|---------------:|--------------:|---------:|---------:|
| Right | 90 | 0.0270 | 0.0336 | 0.0532 | 0.1593 |
| Left | 83 | 0.0256 | 0.0353 | 0.0457 | 0.0734 |

CSV/JSON 저장:

```bash
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --csv /tmp/air_drum_bench.csv --json /tmp/air_drum_bench.json
```

---

## 9. 참조

- 아키텍처: `docs/ARCHITECTURE.md`
