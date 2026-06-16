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
| DX-COM | MediaPipe hand `.dxnn`: v2.1.0-rc.4 metadata; palm/PINTO `.dxnn`: DX-COM 2.3.0-class tooling |
| Camera | USB, 640×480 |
| Commit | See current `git log` for the exact report revision |

### 8.2 Palm Detection — NPU (.dxnn) vs CPU (TFLite)

> **결론:** DX-COM 2.3.0으로 컴파일한 palm `.dxnn` 후보는 보드 offline replay에서 accepted palm을 만들고 hand landmark stage까지 연결됩니다. CPU palm은 안정적인 live 기본값으로 유지하고, full-NPU palm은 `--palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn`으로 명시 실험합니다.

**Palm Detection 레이턴시 (90-frame replay):**

| Palm 백엔드 | Palm stage | Hand stage | 전체 | 비고 |
|-------------|-----------:|-----------:|-----:|------|
| CPU TFLite + NPU hand | 39.84 ms | 8.48 ms | 48.61 ms | conservative live default |
| NPU palm + NPU hand | 10.00 ms | 13.85 ms | 24.32 ms | best current offline speed/accuracy candidate |

**Palm .dxnn tensor validation (2026-06-16):**

```bash
python3 tools/debug_palm_outputs.py --backend all \
  --image dataset/frame_000.png \
  --dxnn models/vendor/palm_detection_lite_minmax_local.dxnn \
  --score-thresh 0.5 --dxnn-input-variant nhwc_u8
python3 tools/debug_palm_outputs.py --backend all \
  --image dataset/frame_060.png \
  --dxnn models/vendor/palm_detection_lite_minmax_local.dxnn \
  --score-thresh 0.5 --dxnn-input-variant nhwc_u8
```

| Candidate | Frame | TFLite detections | DXNN detections | DXNN score corr vs TFLite | DXNN box corr vs TFLite |
|-----------|-------|------------------:|----------------:|--------------------------:|------------------------:|
| `minmax_local` | `frame_000` | 2 | 2 | 0.9809 | 0.9937 |
| `minmax_local` | `frame_060` | 3 | 2 | 0.9780 | 0.9926 |
| `ema_local` | `frame_000` | 2 | 2 | 0.9809 | 0.9937 |
| `ema_local` | `frame_060` | 3 | 2 | 0.9780 | 0.9926 |

상세 checksum, compiler log, command는 [`SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md`](SESSION_2026_06_16_PALM_DXNN_LOCAL_RECOMPILE.md)에 기록했습니다.

### 8.3 End-to-End 레이턴시 비교

| 백엔드 | Palm | Hand (per hand) | 전체 (2 hands) | 비고 |
|---------|-----:|----------------:|---------------:|------|
| `cpu` (MediaPipe) | MediaPipe internal | MediaPipe internal | **64.95 ms** | 90-frame replay, high variance |
| `cpu-baseline` (TFLite) | 40.29 ms | 45.72 ms | **86.42 ms** | 90-frame replay, 비교 기준선 |
| `pinto-cpu` (TFLite+PINTO ONNX) | 38.59 ms | 50.59 ms | **89.58 ms** | 90-frame replay |
| `npu-full` (TFLite+NPU) | 39.84 ms | 8.48 ms | **48.61 ms** | 90-frame replay, live 기본 경로 |
| `pinto-npu` (TFLite+PINTO DXNN) | 40.23 ms | 8.62 ms | **49.14 ms** | 90-frame replay, 실험용 |
| `npu-full --palm-dxnn` | 10.00 ms | 13.85 ms | **24.32 ms** | 90-frame replay, full-NPU 후보 |
| `pinto-npu --palm-dxnn` | 9.92 ms | 12.91 ms | **23.29 ms** | 90-frame replay, faster but less accurate |
| `npu` (dual-halves) | 0 ms | NPU-only hand pass | **8.42 ms** | 90-frame replay; palm 검출 없음, 근사 |

> **npu-full / cpu-baseline**: 매 프레임 palm detection을 실행합니다 (`_PALM_REDETECT_EVERY = 0`). 이전에는 landmark 기반 ROI 트래킹으로 palm skip(5프레임에 1번)을 사용했으나, NPU INT8 양자화 편향 누적으로 ROI 드리프트(최대 dy=0.26)가 발생하여 비활성화했습니다.
> 
> **cpu-baseline vs npu-full / pinto-npu**: 동일 palm+ROI 구조에서 hand landmark만 CPU float32 또는 NPU INT8로 바꾸어 비교합니다. CPU palm path에서는 palm이 약 40 ms를 차지합니다. `--palm-dxnn` path에서는 palm도 NPU로 이동해 전체 replay latency가 약 24 ms까지 내려갑니다.

> **Palm skip 최적화는 실험 옵션입니다.** 기본값은 정확도 우선(`--palm-redetect-every 0`, 매 프레임 palm)입니다. `--palm-redetect-every 5`처럼 지정하면 이전 프레임 랜드마크에서 다음 ROI를 예측해 palm detection을 건너뛰며, 지연은 줄지만 NPU INT8 편향이 누적될 수 있어 dataset benchmark로 오차를 반드시 확인합니다.

### 8.4 Offline dataset benchmark (2026-05-08)

도구:

```bash
cd air_drum_pad
python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full
python3 tools/benchmark_dataset.py --backends cpu-baseline,pinto-cpu,pinto-npu,npu-full --limit 90 --warmup 0
python3 tools/benchmark_dataset.py --backends cpu-baseline,pinto-npu,npu-full \
  --palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn --limit 90 --warmup 0
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

현재 `dataset/` 에서 확인한 예시 결과(Orange Pi 5 Plus). 모든 주요 backend rows는 90-frame replay 기준입니다.

| 구성 | 평균 | P95 | 세부 프로파일 | 비고 |
|------|-----:|----:|----------------|------|
| `cpu` (MediaPipe) | 64.95 ms | 127.61 ms | MediaPipe internal | functional CPU baseline, high variance |
| `cpu-baseline`, palm every frame | 86.42 ms | 87.31 ms | palm 40.29 ms + hand 45.72 ms | CPU TFLite 기준 |
| `npu-full`, CPU palm | 48.61 ms | 49.30 ms | palm 39.84 ms + hand 8.48 ms | 같은 palm+ROI, hand만 NPU |
| `npu-full --palm-dxnn` | 24.32 ms | 29.71 ms | palm 10.00 ms + hand 13.85 ms | full-NPU 후보, accuracy-speed 균형 최선 |
| `pinto-cpu` | 89.58 ms | 106.57 ms | palm 38.59 ms + hand 50.59 ms | PINTO ONNX CPU path는 동작하지만 latency win 아님 |
| `pinto-npu`, CPU palm | 49.14 ms | 50.59 ms | palm 40.23 ms + hand 8.62 ms | PINTO DXNN hand stage 실행 |
| `pinto-npu --palm-dxnn` | 23.29 ms | 28.79 ms | palm 9.92 ms + hand 12.91 ms | 더 빠르지만 right-hand agreement 약함 |
| `npu` dual-halves | 8.42 ms | 18.92 ms | no palm detector | 가장 빠르지만 정확도 탈락 |
| `npu-full`, `--palm-redetect-every 5` (20프레임 smoke) | 15.29 ms | 50.96 ms | palm frames 3/20, tracking frames 17/20 | 지연 개선, 정확도 회귀 확인 필요 |
| `npu-full`, `--async-palm` smoke | 10–20 ms대 | 입력 pacing 의존 | tracking + async palm refresh | 실험용 파이프라인 |

`npu-full` vs `cpu-baseline` landmark 오차(90프레임, normalized xy):

| Backend | 손 | 매칭 프레임 | 전체 21점 평균 | 손끝 5점 평균 | 평균 max | 최대 max |
|---------|----|-----------:|---------------:|--------------:|---------:|---------:|
| `npu-full`, CPU palm | Right | 89 | 0.0220 | 0.0266 | 0.0438 | 0.1316 |
| `npu-full`, CPU palm | Left | 90 | 0.0152 | 0.0226 | 0.0305 | 0.0547 |
| `npu-full --palm-dxnn` | Right | 89 | 0.0225 | 0.0262 | 0.0518 | 0.1045 |
| `npu-full --palm-dxnn` | Left | 90 | 0.0166 | 0.0244 | 0.0347 | 0.0559 |

저장된 NPU landmark 보정 파일도 테스트했지만 현재 replay 기준으로는 기본값으로 쓰지 않습니다. Dataset affine 보정은 right-hand mean error를 0.0314로 악화시켰고, bias 보정도 right-hand mean error를 0.0290으로 악화시켰습니다. 따라서 live demo default는 무보정이며, 보정은 controlled calibration set이 있을 때만 opt-in 합니다.

PINTO 90-frame landmark 오차(`cpu-baseline` 기준, normalized xy):

| Backend | 손 | 전체 21점 평균 | 손끝 5점 평균 | 평균 max | 최대 max |
|---------|----|---------------:|--------------:|---------:|---------:|
| `pinto-cpu` | Right | 0.0456 | 0.0449 | 0.0866 | 0.2272 |
| `pinto-cpu` | Left | 0.0126 | 0.0093 | 0.0270 | 0.0642 |
| `pinto-npu`, CPU palm | Right | 0.0454 | 0.0572 | 0.0873 | 0.2262 |
| `pinto-npu`, CPU palm | Left | 0.0141 | 0.0196 | 0.0302 | 0.0585 |
| `pinto-npu --palm-dxnn` | Right | 0.0468 | 0.0607 | 0.0907 | 0.2211 |
| `pinto-npu --palm-dxnn` | Left | 0.0152 | 0.0225 | 0.0307 | 0.0615 |

`pinto-npu`는 `pinto-cpu` 대비 latency는 크게 개선되지만, default MediaPipe-derived `npu-full`을 대체할 정도의 landmark agreement 우위는 없습니다. Palm NPU를 붙이면 `pinto-npu --palm-dxnn`은 23.29 ms까지 내려가지만 right-hand error가 `npu-full --palm-dxnn`보다 큽니다.

세 축 비교 요약(90-frame replay 기준):

| Backend | 평균 latency | FPS 환산 | NPU-active share proxy | `cpu-baseline` 대비 평균 landmark error | 해석 |
|---------|-------------:|---------:|-----------------------:|----------------------------------------:|------|
| `cpu-baseline` | 86.42 ms | 11.6 FPS | 0.0% | 0.0000 reference | 정확도 기준선, NPU 없음 |
| `pinto-cpu` | 89.58 ms | 11.2 FPS | 0.0% | 0.0290 | 동작하지만 속도 이점 없음 |
| `npu-full`, CPU palm | 48.61 ms | 20.6 FPS | 17.5% | 0.0186 | live 기본 경로, CPU palm 병목 |
| `pinto-npu`, CPU palm | 49.14 ms | 20.4 FPS | 17.5% | 0.0290 | NPU hand 속도는 확보, 정확도 열세 |
| `npu-full --palm-dxnn` | 24.32 ms | 41.1 FPS | 98.0% | 0.0195 | 현재 NPU-backed speed/accuracy 균형 최선 |
| `pinto-npu --palm-dxnn` | 23.29 ms | 42.9 FPS | 98.0% | 0.0305 | 빠르지만 right-hand error 큼 |
| `npu` dual-halves | 8.42 ms | 118.7 FPS | 약 100% | 0.1610 | 가장 빠르고 NPU-bound지만 정확도 탈락 |

`NPU-active share proxy`는 raw `dxtop` 순간 util이 아니라, benchmark profile에서 NPU-backed stage 시간이 전체 vision-loop latency에서 차지하는 비율입니다. CPU-palm `npu-full`과 `pinto-npu`는 손 landmark stage만 NPU라 share가 약 17-18%로 낮고, CPU palm detection이 전체 latency를 지배합니다. Palm-NPU rows는 palm과 hand가 모두 NPU-backed라 share가 약 98%입니다.

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
