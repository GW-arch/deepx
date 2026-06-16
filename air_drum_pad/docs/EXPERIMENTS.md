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
| DX-RT / dx_engine | DXRT v3.2.0 observed through `dxrt-cli`; NPU driver v2.1.0; firmware v2.5.0 |
| DX-COM | MediaPipe `.dxnn`: v2.1.0-rc.4 metadata; PINTO `.dxnn`: v2.3.0-rc.5 metadata |
| Camera | USB, 640×480 |
| Commit | See current `git log` for the exact report revision |

### 8.2 Palm Detection — NPU (.dxnn) vs CPU (TFLite)

> **결론:** Palm .dxnn은 속도는 빠르나 INT8 양자화로 **score head가 파괴**되어 (ONNX↔NPU 상관 -0.11) 실제 손 감지가 불가합니다. 아래 레이턴시 수치는 참고용이며, **실사용은 TFLite (CPU)** 입니다.

**Palm Detection 레이턴시 (참고):**

| Palm 백엔드 | 추론 시간 | 비고 |
|-------------|----------:|------|
| NPU (.dxnn) | ~10.4-11.2 ms | score 파괴로 **사용 불가**; no accepted palms, hand stage `0.00 ms` |
| CPU (TFLite XNNPACK) | ~39-42 ms | **실사용** (float32, 정확) |

### 8.3 End-to-End 레이턴시 비교

| 백엔드 | Palm | Hand (per hand) | 전체 (2 hands) | 비고 |
|---------|-----:|----------------:|---------------:|------|
| `cpu` (MediaPipe) | ~15 ms | ~10 ms | ~35 ms | 모두 float32, 추가 파일 불필요 |
| `cpu-baseline` (TFLite) | 41.70 ms | 45.55 ms | **87.73 ms** | 10-frame replay smoke, 비교 기준선 |
| `pinto-cpu` (TFLite+PINTO ONNX) | 38.62 ms | 49.85 ms | **88.85 ms** | 10-frame replay smoke |
| `npu-full` (TFLite+NPU) | 40.91 ms | 8.57 ms | **49.76 ms** | 10-frame replay smoke, 기본 경로 |
| `pinto-npu` (TFLite+PINTO DXNN) | 41.21 ms | 9.00 ms | **50.48 ms** | 10-frame replay smoke, 실험용 |
| `npu` (dual-halves) | 0 ms | NPU-only hand pass | **7.16 ms** | 10-frame replay smoke; palm 검출 없음, 근사 |

> **npu-full / cpu-baseline**: 매 프레임 palm detection을 실행합니다 (`_PALM_REDETECT_EVERY = 0`). 이전에는 landmark 기반 ROI 트래킹으로 palm skip(5프레임에 1번)을 사용했으나, NPU INT8 양자화 편향 누적으로 ROI 드리프트(최대 dy=0.26)가 발생하여 비활성화했습니다.
> 
> **cpu-baseline vs npu-full / pinto-npu**: 동일 palm+ROI 구조에서 hand landmark만 CPU float32 또는 NPU INT8로 바꾸어 비교합니다. 현재 replay에서는 CPU palm이 약 40 ms를 차지하므로, NPU 가속 효과는 hand landmark stage의 45-50 ms → 약 9 ms 감소로 해석해야 합니다.

> **Palm skip 최적화는 실험 옵션입니다.** 기본값은 정확도 우선(`--palm-redetect-every 0`, 매 프레임 palm)입니다. `--palm-redetect-every 5`처럼 지정하면 이전 프레임 랜드마크에서 다음 ROI를 예측해 palm detection을 건너뛰며, 지연은 줄지만 NPU INT8 편향이 누적될 수 있어 dataset benchmark로 오차를 반드시 확인합니다.

### 8.4 Offline dataset benchmark (2026-05-08)

도구:

```bash
cd air_drum_pad
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full
python3 tools/benchmark_dataset.py --backends cpu-baseline,pinto-cpu,pinto-npu,npu-full --limit 10 --warmup 0
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --palm-redetect-every 5
python3 tools/sweep_palm_redetect.py --values 0,1,2,3,5,10 \
  --backends cpu-baseline,npu-full --csv /tmp/palm_sweep.csv
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --debug-dir /tmp/air_drum_debug --debug-top-k 10
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --async-palm --frame-interval-ms 16.7
python3 tools/calibrate_npu_landmarks.py \
  --kind bias --output models/npu_landmark_correction.bias.json
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --landmark-correction models/npu_landmark_correction.bias.json
python3 tools/calibrate_npu_landmarks.py \
  --output models/npu_landmark_correction.dataset.json
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --landmark-correction models/npu_landmark_correction.dataset.json
```

현재 `dataset/` 에서 확인한 예시 결과(Orange Pi 5 Plus). `cpu-baseline`/`npu-full`의 주 비교는 90프레임 기준이며, PINTO rows는 컴파일된 PINTO `.dxnn` 추가 후 수행한 10-frame smoke입니다.

| 구성 | 평균 | P95 | 세부 프로파일 | 비고 |
|------|-----:|----:|----------------|------|
| `cpu-baseline`, palm every frame | 84.40 ms | 86.78 ms | palm 39.26 ms + hand 44.80 ms | CPU TFLite 기준 |
| `npu-full`, palm every frame | 50.32 ms | 54.52 ms | palm 40.93 ms + hand 9.13 ms | 같은 palm+ROI, hand만 NPU |
| `npu-full`, repeated 5-run measurement | 50.10 ms | 51.71 ms | palm 40.69 ms + hand 9.12 ms | 기본 CPU-palm + NPU-hand profile 재현 |
| `npu-full --palm-dxnn`, repeated run 1 | 10.39 ms | 12.84 ms | palm 10.34 ms + hand 0.00 ms | Palm NPU path는 빠르지만 accepted palm 없음 |
| `npu-full --palm-dxnn`, repeated run 2 | 10.88 ms | 13.13 ms | palm 10.82 ms + hand 0.00 ms | 같은 실패 모드 재현 |
| `pinto-cpu` (10-frame smoke) | 88.91 ms | 97.13 ms | palm 40.19 ms + hand 48.27 ms | PINTO ONNX CPU path는 동작하지만 latency win 아님 |
| `pinto-npu` (10-frame smoke) | 50.48 ms | 54.48 ms | palm 41.21 ms + hand 9.00 ms | PINTO DXNN path는 동작하며 hand stage를 NPU 속도로 실행 |
| `npu-full`, `--palm-redetect-every 5` (20프레임 smoke) | 15.29 ms | 50.96 ms | palm frames 3/20, tracking frames 17/20 | 지연 개선, 정확도 회귀 확인 필요 |
| `npu-full`, `--async-palm` smoke | 10–20 ms대 | 입력 pacing 의존 | tracking + async palm refresh | 실험용 파이프라인 |

`npu-full` vs `cpu-baseline` landmark 오차(90프레임, normalized xy):

| 손 | 매칭 프레임 | 전체 21점 평균 | 손끝 5점 평균 | 평균 max | 최대 max |
|----|-----------:|---------------:|--------------:|---------:|---------:|
| Right | 90 | 0.0220 | 0.0267 | 0.0439 | 0.1316 |
| Left | 83 | 0.0151 | 0.0225 | 0.0304 | 0.0547 |

저장된 NPU landmark 보정 파일도 테스트했지만 현재 replay 기준으로는 기본값으로 쓰지 않습니다. Dataset affine 보정은 right-hand mean error를 0.0314로 악화시켰고, bias 보정도 right-hand mean error를 0.0290으로 악화시켰습니다. 따라서 live demo default는 무보정이며, 보정은 controlled calibration set이 있을 때만 opt-in 합니다.

PINTO 10-frame smoke landmark 오차(`cpu-baseline` 기준, normalized xy):

| Backend | 손 | 전체 21점 평균 | 손끝 5점 평균 | 평균 max | 최대 max |
|---------|----|---------------:|--------------:|---------:|---------:|
| `pinto-cpu` | Right | 0.0176 | 0.0145 | 0.0321 | 0.0330 |
| `pinto-cpu` | Left | 0.0103 | 0.0064 | 0.0198 | 0.0208 |
| `pinto-npu` | Right | 0.0205 | 0.0313 | 0.0429 | 0.0441 |
| `pinto-npu` | Left | 0.0115 | 0.0140 | 0.0278 | 0.0285 |
| `npu-full` | Right | 0.0068 | 0.0100 | 0.0133 | 0.0154 |
| `npu-full` | Left | 0.0129 | 0.0178 | 0.0264 | 0.0308 |

`pinto-npu`는 `pinto-cpu` 대비 latency는 크게 개선되지만, 10-frame smoke 기준으로는 default `npu-full`을 대체할 정도의 landmark agreement 우위가 없습니다. PINTO를 palm 없이 `npu` dual-halves 방식에 넣은 임시 비교도 threshold `0.5`에서는 accepted hand가 없었고, threshold `0.0`으로 강제해도 Right `0.1514`, Left `0.1464` 수준이라 정확한 backend로 보기 어렵습니다.

세 축 비교 요약(backend 간 같은 window 비교를 위해 10-frame replay smoke 기준):

| Backend | 평균 latency | FPS 환산 | NPU-active share proxy | `cpu-baseline` 대비 평균 landmark error | 해석 |
|---------|-------------:|---------:|-----------------------:|----------------------------------------:|------|
| `cpu-baseline` | 87.73 ms | 11.4 FPS | 0.0% | 0.0000 reference | 정확도 기준선, NPU 없음 |
| `pinto-cpu` | 88.85 ms | 11.3 FPS | 0.0% | 0.0140 | 동작하지만 속도 이점 없음 |
| `npu-full` | 49.76 ms | 20.1 FPS | 17.2% | 0.0099 | 현재 NPU 사용 backend 중 정확도 최선 |
| `pinto-npu` | 50.48 ms | 19.8 FPS | 17.8% | 0.0160 | NPU 속도는 확보했지만 `npu-full` 정확도는 못 이김 |
| `npu` dual-halves | 7.16 ms | 139.7 FPS | 약 100% | 0.1456 | 가장 빠르고 NPU-bound지만 정확도 탈락 |
| `npu-full --palm-dxnn` | 10.88 ms | 91.9 FPS | 99.4% | invalid | palm NPU는 실행되나 accepted palm 없음 |

`NPU-active share proxy`는 raw `dxtop` 순간 util이 아니라, benchmark profile에서 NPU-backed stage 시간이 전체 vision-loop latency에서 차지하는 비율입니다. `npu-full`과 `pinto-npu`는 손 landmark stage만 NPU라 share가 약 17-18%로 낮고, CPU palm detection이 전체 latency를 지배합니다. 반면 `npu` dual-halves와 palm `.dxnn` path는 NPU 비중은 높지만 정확도 또는 accepted-palm 조건을 만족하지 못합니다.

CSV/JSON 저장:

```bash
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full \
  --csv /tmp/air_drum_bench.csv --json /tmp/air_drum_bench.json
```

오차 overlay는 `--debug-dir`에 PNG와 `manifest.json`을 저장합니다. reference는 기본적으로 `cpu-baseline`이며, overlay 색상은 green=reference, red=test입니다.

### 8.5 품질 체크

반복 구현 후 최소 검증:

```bash
./scripts/check_quality.sh
RUN_BENCH_SMOKE=0 ./scripts/check_quality.sh  # 모델/데이터셋 없는 환경
python3 -m unittest discover -s tests -v
```

현재 단위 테스트는 strike detector, ROI transform, dataset benchmark helper, palm redetect sweep helper, capture dataset indexing을 검사합니다.

---

## 9. 참조

- 아키텍처: `docs/ARCHITECTURE.md`
