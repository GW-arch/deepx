# AI Air-Drum Pad (prototype)

**관리 저장소:** [github.com/GW-arch/deepx](https://github.com/GW-arch/deepx) — 이 프로젝트는 항상 해당 레포 안에서 버전 관리합니다.

## 실행

```bash
pip3 install -r requirements.txt
python3 main.py --camera 0
```

- 종료: `q`
- 민감도: `--vy-trigger`, `--cooldown` (도움말: `python3 main.py -h`)

## 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 카메라, MediaPipe Hands, UI |
| `strike_detector.py` | Hit Zone, 하강 속도, 쿨다운 |
| `drumkit_audio.py` | pygame 드럼 샘플 |

DeepX M1에서는 MediaPipe 대신 DX-RT + Hand ONNX(.dxnn)로 Vision 레이어만 교체하면 됩니다.
