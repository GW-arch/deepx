# AI Air-Drum Pad (prototype)

**관리 저장소 (SSH):** `git@github.com:GW-arch/deepx.git`  
웹: [github.com/GW-arch/deepx](https://github.com/GW-arch/deepx)

## 동작 개요 (실제로 치는 것처럼)

화면에 **고정 패드 영역은 없습니다.** 손가락 끝을 **추적**하고, 아래 두 조건을 **동시에** 만족할 때만 한 번 친 것으로 봅니다.

1. **손끝 하강 속도** — 막대기 끝이 아래로 빠르게 움직임 (`vy`, 정규화 좌표/초)
2. **관절 각속도** — MCP–PIP–TIP(엄지는 IP 포함)에서 잰 각도가 프레임마다 충분히 변함 → 손가락 관절이 실제로 휘둘러짐

악기 종류는 **어느 손 × 어느 손가락**인지로 고정 매핑 (예: 왼손 검지=스네어 …).

## 실행

```bash
cd air_drum_pad
pip3 install -r requirements.txt
python3 main.py --camera 0
```

- 종료: `q`
- 민감도: `--vy-trigger`, `--joint-dps` (관절 각속도 하한, deg/s), `--cooldown`

느리게만 움직이면 안 울리게 하려면 `--joint-dps`를 올리고, 너무 안 나오면 `--vy-trigger` / `--joint-dps`를 내립니다.

## 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 카메라, MediaPipe Hands, 관절선·손끝 궤적 표시 |
| `strike_detector.py` | `InstrumentStrikeDetector` — 손끝 속도 + 관절 각속도 |
| `drumkit_audio.py` | 16종 합성 샘플, 손가락 슬롯에 매핑 |

DeepX M1에서는 MediaPipe 대신 DX-RT + Hand ONNX(.dxnn)로 랜드마크만 넣어 주면 동일 로직을 쓸 수 있습니다.
