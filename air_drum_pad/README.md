# AI Air-Drum Pad (prototype)

**관리 저장소 (SSH):** `git@github.com:GW-arch/deepx.git`  
웹: [github.com/GW-arch/deepx](https://github.com/GW-arch/deepx)

## 동작 개요 (실제로 치는 것처럼)

화면에 **고정 패드 영역은 없습니다.** 손가락 끝을 **추적**하고, 아래 두 조건을 **동시에** 만족할 때만 한 번 친 것으로 봅니다.

1. **손끝 하강 속도** — 막대기 끝이 아래로 빠르게 움직임 (`vy`, 정규화 좌표/초)
2. **관절 각속도** — MCP–PIP–TIP(엄지는 IP 포함)에서 잰 각도가 프레임마다 충분히 변함 → 손가락 관절이 실제로 휘둘러짐

악기 종류는 **어느 손 × 어느 손가락**인지로 매핑합니다 (기본값은 코드에 있음).

### 악기 바꾸기

1. 사용 가능한 키: `python3 main.py --list-instruments`
2. `instruments.example.json` 을 복사해 `slots` 배열 **10개**를 수정 (순서: **손0** 엄지→소지, **손1** 엄지→소지).
3. 실행: `python3 main.py --camera 0 --instruments my.json`

**음색**을 바꾸려면 `drumkit_audio.py`의 `_KIT_BUILDERS`에 키를 추가·수정한 뒤 JSON에서 그 키를 쓰면 됩니다.

### 피아노 모드

- 실행: `python3 main.py --piano --camera 0`  
  **`--instruments` 없이** 켜면: **양손 손목 사이 거리**로 음역을 잡습니다.  
  - 손을 **가깝게** 두면 → 왼손·오른손 모두 **중음역**에 가깝게(겹치는 느낌).  
  - 손을 **멀리** 벌리면 → **왼손(Left)은 더 낮은 펜타토닉**, **오른손(Right)은 더 높은 펜타토닉**으로 벌어집니다.  
  - MediaPipe **Left / Right** 라벨로 좌우를 구분합니다(셀카 미러면 체감이 반대일 수 있음).  
  - 화면 아래에 `d=… L:… R:…` 힌트가 뜹니다.
- **고정 음 배열**을 쓰려면: `instruments.piano.example.json` 참고 후  
  `python3 main.py --piano --instruments 내피아노.json --camera 0`  
  (`slots` 값은 `C4`, `D#5`, `Bb3` 같은 **음명** 10개 — 이 경우 거리 자동 음역은 끔)
- 음색은 짧은 **합성** 사인파(실제 샘플 피아노는 아님).
- 사용 가능 음명(기본 10개 나열): `python3 main.py --piano --list-instruments`

## 실행

```bash
cd air_drum_pad
pip3 install -r requirements.txt
python3 main.py --camera 0
# 피아노:
# python3 main.py --piano --camera 0
```

### CPU vs NPU (손 추론)

| `--backend` | 설명 |
|-------------|------|
| `cpu` (기본) | MediaPipe Hands |
| `npu` | DX-RT `.dxnn` + `dx_engine` (레이아웃 JSON) |

NPU 예시는 `models/README.md` 와 `scripts/run_npu_piano.sh` 참고.  
요약: **MediaPipe TFLite → ONNX** (`tools/export_mediapipe_hand_onnx.py`) → **DX-COM** (로컬 `tools/compile_dxnn.sh` 또는 SNU 서버 `tools/compile_server_snu.sh`) → 보드에서 `--backend npu --dxnn …`.

- 종료: `q`
- 민감도: `--vy-trigger`, `--joint-dps` (관절 각속도 하한, deg/s), `--cooldown`

느리게만 움직이면 안 울리게 하려면 `--joint-dps`를 올리고, 너무 안 나오면 `--vy-trigger` / `--joint-dps`를 내립니다.

## 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 카메라, 손 추적(`--backend`), 관절선·손끝 궤적 표시 |
| `hand_tracker.py` | CPU(MediaPipe) / NPU(DX-RT `.dxnn`) 랜드마크 |
| `strike_detector.py` | `InstrumentStrikeDetector` — 손끝 속도 + 관절 각속도 |
| `drumkit_audio.py` | 16종 합성 샘플, 손가락 슬롯에 매핑 |
| `tools/export_mediapipe_hand_onnx.py` | 공개 TFLite → ONNX + 레이아웃 생성 |
| `tools/compile_dxnn.sh` | DX-COM 호출 래퍼 (`DX_COM` 환경변수 지원) |
