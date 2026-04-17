# AI Air-Drum Pad (prototype)

**관리 저장소 (SSH):** `git@github.com:GW-arch/deepx.git` — 이 디렉터리는 위 레포의 `air_drum_pad/` 로만 관리합니다.  
웹: [github.com/GW-arch/deepx](https://github.com/GW-arch/deepx)

## 실행

```bash
# 레포 루트에서
cd air_drum_pad
pip3 install -r requirements.txt
python3 main.py --camera 0
```

- 종료: `q`
- 민감도: `--vy-trigger`, `--cooldown` (도움말: `python3 main.py -h`)

## 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)

## 기능 (요약)

- **손 최대 2개** (`--max-hands 1|2`): 양손 동시 인식
- **손가락 끝 5개** (엄지·검지·중지·약지·소지): 각각 독립 속도·쿨다운으로 타격 판정
- **16패드(4×4)** 그리드: 킥/스네어/하이햇/라이드/톰/크래시/클랩 등 서로 다른 `sound_key`
- **pygame 채널 32** — 동시 발음(폴리) 대응

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 카메라, MediaPipe Hands, 다중 손·손가락 루프, UI |
| `strike_detector.py` | 16 Hit Zone, `(손 ID, 랜드마크)` 단위 속도·쿨다운 |
| `drumkit_audio.py` | 16종 합성 샘플 + `build_kit()` |

DeepX M1에서는 MediaPipe 대신 DX-RT + Hand ONNX(.dxnn)로 Vision 레이어만 교체하면 됩니다.
