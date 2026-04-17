# AI Air-Drum Pad — 시스템 아키텍처 문서

**프로젝트명:** NPU 기반 초저지연 Hand Landmark Tracking을 활용한 AI Air-Drum Pad  
**문서 버전:** 1.0  
**대상 플랫폼:** DeepX M1 키트(Ubuntu 22.04, SBC + NPU + USB 카메라 + 디스플레이)  
**참고 구현:** 이 모노레포 `git@github.com:GW-arch/deepx.git` ([웹](https://github.com/GW-arch/deepx)) 의 `air_drum_pad/` Python 프로토타입(CPU/MediaPipe)

---

## 1. 목적 및 범위

카메라 입력부터 오디오 출력까지의 **E2E 파이프라인**을 계층적으로 정의하고, **CPU·NPU 이기종 연산 분담**, 스레드·버퍼 구조, 외부 인터페이스(V4L2, ALSA/JACK, DX-RT)를 명시한다.

---

## 2. 요구사항 요약

| 구분 | 요구 |
|------|------|
| 기능 | 허공 타격을 Hit Zone에 매핑하여 샘플 재생 |
| 실시간성 | E2E 지연 목표 **약 15~20 ms 이하**(보고서에서 정의 고정) |
| 처리량 | **≥ 60 FPS** 비전 루프 |
| NPU | Hand Landmark **CNN 추론**, INT8 양자화 |
| 오프라인 | **클라우드 불필요** |

---

## 3. 논리 아키텍처 (4계층)

1. **Input:** V4L2 / OpenCV — 프레임 + 타임스탬프  
2. **Vision:** 전처리 → **NPU 추론(DX-RT + .dxnn)** — 21관절 또는 검지 끝 + 신뢰도  
3. **Logic:** Hit Zone, 속도·가속도 휴리스틱, 쿨다운  
4. **Output:** ALSA/JACK, 사전 로드 PCM/WAV  

---

## 4. 소프트웨어 스택 (목표)

| 영역 | 기술 |
|------|------|
| OS | Ubuntu 22.04 |
| 비전 | OpenCV, (선택) GStreamer |
| 추론 | ONNX → **DX-COM** → **DX-RT**(.dxnn), INT8 PTQ |
| 로직 | C++ 권장 |
| 오디오 | ALSA 저버퍼 또는 JACK |

**프로토타입:** `main.py`(MediaPipe), `strike_detector.py`, `drumkit_audio.py`(pygame)

---

## 5. NPU 모델

- **주 모델:** MediaPipe 계열 **Hand Landmark**(21 keypoints), ONNX → INT8 → .dxnn  
- **보조(선택):** Palm/Hand **Detector** — 추적 실패 시에만 재실행(캐스케이드)

---

## 6. 타격 로직

- 검지 끝 정규화 좌표, \(v_y = \Delta y / \Delta t\), Zone 내부 + 임계 + 쿨다운.

---

## 7. 문서 참조

- 실험 절차: `docs/EXPERIMENTS.md`
