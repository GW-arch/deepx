# deepx

DeepX M1 및 엣지 AI 관련 프로젝트를 이 저장소에서 관리합니다.

**원격(Git):** SSH만 사용합니다.

```bash
git clone git@github.com:GW-arch/deepx.git
cd deepx
```

웹: [github.com/GW-arch/deepx](https://github.com/GW-arch/deepx)

## 구성

| 경로 | 설명 |
|------|------|
| [`air_drum_pad/`](air_drum_pad/) | NPU 기반 AI Air-Drum Pad — Hand Landmark · Hit Zone · 저지연 오디오 프로토타입 및 문서 |

## 클론 후 실행 (Air-Drum 프로토타입)

```bash
git clone git@github.com:GW-arch/deepx.git
cd deepx/air_drum_pad
pip3 install -r requirements.txt
python3 main.py --camera 0
```

종료: 창 포커스 상태에서 `q`.

## 문서

- [아키텍처](air_drum_pad/docs/ARCHITECTURE.md)
- [실험 방법](air_drum_pad/docs/EXPERIMENTS.md)

## 원격 저장소

- `git@github.com:GW-arch/deepx.git` (SSH)
- `sites.google.com/view/dxs-2603-snu` (DEEPX)

## Compile Server
- `ssh user12@43.203.143.33 -p 443`
- `pw: snu*npu&&`
