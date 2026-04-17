# deepx

DeepX M1 및 엣지 AI 관련 프로젝트를 이 저장소에서 관리합니다.

## 구성

| 경로 | 설명 |
|------|------|
| [`air_drum_pad/`](air_drum_pad/) | NPU 기반 AI Air-Drum Pad — Hand Landmark · Hit Zone · 저지연 오디오 프로토타입 및 문서 |

## 클론 후 실행 (Air-Drum 프로토타입)

```bash
cd air_drum_pad
pip3 install -r requirements.txt
python3 main.py --camera 0
```

종료: 창 포커스 상태에서 `q`.

## 문서

- [아키텍처](air_drum_pad/docs/ARCHITECTURE.md)
- [실험 방법](air_drum_pad/docs/EXPERIMENTS.md)

## 원격 저장소

- https://github.com/GW-arch/deepx
