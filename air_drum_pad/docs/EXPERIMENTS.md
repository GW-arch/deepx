# AI Air-Drum Pad — 실험 방법서

**문서 버전:** 1.0  
**저장소 (Git SSH):** `git@github.com:GW-arch/deepx.git` · [웹](https://github.com/GW-arch/deepx)

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

## 8. 참조

- 아키텍처: `docs/ARCHITECTURE.md`
