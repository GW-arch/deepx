# Session Note: NPU Hand Pipeline, PINTO Adoption, and Utilization

Date: 2026-06-16

## Executive Summary

- The usable `npu-full` path currently means CPU palm detection plus NPU MediaPipe hand landmark.
- The experimental full-NPU path exists via `--palm-dxnn models/vendor/palm_detection_lite.dxnn`, but it is not a valid tracker yet: palm inference runs on the NPU, but it produces no accepted palm detections in the benchmark path, so the hand landmark stage never runs.
- The low-accuracy concern is mainly in the hand landmark `.dxnn` INT8 behavior versus CPU float TFLite. The palm stage is a separate issue.
- Landmark correction is now opt-in because measured correction made the default right-hand error worse.
- PINTO hand landmark can run through both the `pinto-cpu` ONNX backend and the new `pinto-npu` DXNN backend.
- The SNU compile server was unreachable from this board during the session (`43.203.143.33:443` timed out). Local DX-COM install is not practical on this Orange Pi because the local board is `aarch64` while the compiler documentation/package path targets `x86_64/amd64`.

## Backend Modes

Current app-facing backend modes:

- `cpu`: MediaPipe-style CPU palm and CPU hand landmark path.
- `cpu-baseline`: CPU baseline used for benchmark comparison.
- `npu`: NPU hand landmark where supported by the current tracker path.
- `npu-full`: Full tracker wrapper. By default this is CPU palm plus NPU hand landmark.
- `pinto-cpu`: Experimental PINTO ONNX hand landmark on CPU, using the existing palm/ROI flow.
- `pinto-npu`: Experimental PINTO DXNN hand landmark on NPU, using the existing CPU palm/ROI flow.

Important distinction:

- `npu-full` without `--palm-dxnn`: valid hand pipeline, but palm remains CPU.
- `npu-full --palm-dxnn models/vendor/palm_detection_lite.dxnn`: palm inference is on NPU, but current detection acceptance fails and `hand=0.00 ms`.

## Current MediaPipe-Derived Models

Current production-like NPU hand landmark model:

- Source family: MediaPipe hand landmark lite.
- NPU file: `models/vendor/hand_landmark_lite.dxnn`.
- Input observed through `dx_engine`: `[1, 224, 224, 3]`, `uint8`.
- Output shapes observed through `dx_engine`: `(1, 63)`, `(1, 1)`, `(1, 1)`, `(1, 63)`.
- `parse_model` reported `.dxnn v8`, DX-COM `v2.1.0-rc.4`, outputs `Identity`, `Identity_1`, `Identity_2`, `Identity_3`; task output order was `Identity_3`, `Identity`, `Identity_1`, `Identity_2`.

Interpretation:

- The model is callable and the IO layout is not completely broken.
- The observed accuracy gap is consistent with quantization/output drift in the hand landmark model, not a total runtime failure.

## Landmark Accuracy Measurements

Dataset benchmark on 90 frames comparing CPU baseline and `npu-full`.

Best current default: no landmark correction.

- Right hand: mean `0.0220`, tips `0.0267`, max(avg) `0.0439`, max `0.1316`.
- Left hand: mean `0.0151`, tips `0.0225`, max(avg) `0.0304`, max `0.0547`.
- Latency: about `51 ms`.

Correction experiments:

- Dataset correction worsened right-hand mean to `0.0314`.
- Bias correction worsened right-hand mean to `0.0290`, while slightly improving some left-hand numbers.

Decision:

- Landmark correction is disabled by default.
- `scripts/run_npu_full_piano.sh` supports opt-in correction with `USE_LANDMARK_CORRECTION=1` or explicit `LANDMARK_CORRECTION=...`.

## NPU Utilization Measurements

### Valid `npu-full` default path

Command:

```bash
python3 tools/benchmark_dataset.py --backends npu-full --no-compare --runs 5 --warmup 1
```

Result:

- Frames: `450`.
- Latency mean: `50.10 ms`.
- p50: `49.79 ms`.
- p95: `51.71 ms`.
- FPS(mean): `19.96`.
- Profile: `palm=40.69 ms`, `hand=9.12 ms`, `modes={'palm': 450}`.

`dxtop` screen observation during the run:

- Core0 util: about `1.9%` to `2.1%`.
- Core1/Core2 util: `0.0%`.
- NPU memory: about `6.75 MiB / 3.92 GiB (0.17%)`.
- Temperature: about `50 C`.

Reason:

- This valid path spends most time in CPU palm detection. The NPU hand landmark model is short and invoked after CPU palm/ROI work, so average NPU utilization stays low.

`dxbenchmark` model-level reference for `hand_landmark_lite.dxnn`:

- NPU memory: `7,078,848 bytes`.
- NPU Core average: `582.27 us`.
- NPU Task average: `1379.53 us`.

This is a lower-level model benchmark, not the full app hand-stage time.

### Experimental palm-NPU path

Command:

```bash
python3 tools/benchmark_dataset.py --backends npu-full --palm-dxnn models/vendor/palm_detection_lite.dxnn --no-compare --runs 5 --warmup 1
```

Initial measurement:

- Frames: `450`.
- Latency mean: `11.22 ms`.
- p50: `11.00 ms`.
- p95: `13.12 ms`.
- FPS(mean): `89.09`.
- Profile: `palm=11.17 ms`, `hand=0.00 ms`, `modes={'palm': 450}`.

Repeated measurements on 2026-06-16:

- Run 1: mean `10.39 ms`, p50 `9.86 ms`, p95 `12.84 ms`, min `8.10 ms`, max `18.79 ms`, FPS(mean) `96.24`; profile `palm=10.34 ms`, `hand=0.00 ms`.
- Run 2: mean `10.88 ms`, p50 `10.41 ms`, p95 `13.13 ms`, min `7.90 ms`, max `20.19 ms`, FPS(mean) `91.95`; profile `palm=10.82 ms`, `hand=0.00 ms`.

`dxtop` screen observation during the earlier palm-NPU load:

- Core0 util: about `37.7%` to `46.4%`.
- Core1/Core2 util: `0.0%`.
- NPU memory: about `56.2 MiB / 3.92 GiB (1.40%)`.
- Temperature: about `50 C`.

Important caveat:

- This mode proves that the palm `.dxnn` executes and consumes NPU, but it does not prove a working full-NPU tracker.
- `hand=0.00 ms` means no palm passed the current postprocess/score threshold, so hand landmark did not run.
- `dxtop` does not provide a batch/log option in this installed version (`DX-TOP 1.0.1`); utilization values above are screen observations, while latency/profile values are CLI-captured benchmark output.

## Root Cause Notes

### Why hand landmark looked inaccurate

Likely root cause:

- The current hand landmark `.dxnn` is an INT8-compiled derivative of a MediaPipe float TFLite model.
- The model runs, but output distribution differs enough from CPU float32 baseline to produce visible landmark error.
- Dataset/bias correction was tested and is not a good default because it improves only narrow cases and worsens the right-hand aggregate.

Confidence: medium-high.

### Why palm-NPU full mode is not valid yet

Likely root cause:

- The palm `.dxnn` stage executes quickly on NPU.
- Its score/head/postprocess behavior does not match the CPU TFLite palm detector path closely enough to yield accepted detections.
- Because no palm detections are accepted, the hand landmark stage is skipped.

Confidence: medium. The benchmark evidence is strong (`hand=0.00 ms`), but exact cause still needs output-level debugging against CPU palm tensors.

## PINTO Model Adoption

External references:

- PINTO model zoo hand detection/tracking: `https://github.com/PINTO0309/PINTO_model_zoo/tree/main/033_Hand_Detection_and_Tracking`
- Connected ONNX source used: `https://github.com/PINTO0309/hand-gesture-recognition-using-onnx`

Downloaded ONNX:

- `models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx`

Observed ONNX IO:

- Input: `input ['N', 3, 224, 224]`.
- Outputs:
  - `xyz_x21 ['N', 63]`
  - `hand_score ['N', 1]`
  - `lefthand_0_or_righthand_1 ['N', 1]`

Preprocess:

- RGB.
- Resize/crop using existing ROI.
- Normalize with `/255`.
- Transpose HWC to CHW.

Repo additions for NPU adoption:

- `models/dxcom/pinto_hand_landmark_sparse.json`
- `models/dxnn_layout.pinto_hand_landmark_sparse.json`

Transferred compiled artifacts:

- `models/vendor/pinto_hand_landmark_sparse.dxnn`
- `models/vendor/pinto_hand_landmark_sparse_1x3x224x224_no_round.onnx`
- `build/dxcom/pinto_hand_landmark_sparse_no_round/compiler.log`

Artifact hashes:

- `.dxnn`: `8f35a014f9908f210a71edbba3808bb4940a2ca60ee3b45166bafac5aabdeccb`
- no-round ONNX: `eba37c1b40b2ce6be52b0fa30cbc7b234996950f038fec158229d1022bc9d403`
- `compiler.log`: `e1321908084905b3f159aa15f9bfb386624a716a0ca42550fbc9ed5d902f13ee`

Compile note:

- Dynamic ONNX and fixed-shape ONNX both failed DX-COM lowering on unsupported `onnx.Round`.
- `Round_0` only affects final handedness, so the compileable no-round ONNX replaces it with `Identity`.
- Runtime treats `lefthand_0_or_righthand_1` as a sigmoid probability and thresholds it at `>=0.5`.

Board-side DXRT smoke:

```text
inputs [{'name': 'input', 'shape': [1, 224, 224, 3], 'dtype': uint8}]
outputs:
  0 xyz_x21 [1, 63] float32
  1 hand_score [1, 1] float32
  2 lefthand_0_or_righthand_1 [1, 1] float32
```

CPU benchmark with the existing palm/ROI flow:

- Right hand: mean about `0.0454`, tips about `0.0449`, max(avg) about `0.0864`, max about `0.2272`.
- Left hand: mean about `0.0126`, tips about `0.0093`, max(avg) about `0.0270`, max about `0.0642`.

`pinto-cpu` latency smoke:

```bash
python3 tools/benchmark_dataset.py --backends pinto-cpu --no-compare --limit 10 --warmup 0
```

Result:

- Latency mean: `88.91 ms`.
- p50: `86.57 ms`.
- p95: `97.13 ms`.
- FPS(mean): `11.25`.
- Profile: `palm=40.19 ms`, `hand=48.27 ms`.

`pinto-npu` latency smoke:

```bash
python3 tools/benchmark_dataset.py --backends pinto-npu --no-compare --limit 10 --warmup 0
```

Result:

- Latency mean: `53.54 ms` in the first smoke run.
- p50: `50.53 ms`.
- p95: `64.15 ms`.
- FPS(mean): `18.68`.
- Profile: `palm=42.84 ms`, `hand=10.32 ms`.

Matched backend comparison:

```bash
python3 tools/benchmark_dataset.py --backends cpu-baseline,pinto-cpu,pinto-npu,npu-full --limit 10 --warmup 0
```

Result:

| Backend | Mean | P95 | Profile |
|---------|-----:|----:|---------|
| `cpu-baseline` | `87.73 ms` | `108.52 ms` | palm `41.70 ms` + hand `45.55 ms` |
| `pinto-cpu` | `88.85 ms` | `99.99 ms` | palm `38.62 ms` + hand `49.85 ms` |
| `pinto-npu` | `50.48 ms` | `54.48 ms` | palm `41.21 ms` + hand `9.00 ms` |
| `npu-full` | `49.76 ms` | `50.94 ms` | palm `40.91 ms` + hand `8.57 ms` |

Landmark error vs `cpu-baseline`:

| Backend | Hand | n | Mean | Tips | Max(avg) | Max |
|---------|------|--:|-----:|-----:|---------:|----:|
| `pinto-cpu` | Right | 10 | `0.0176` | `0.0145` | `0.0321` | `0.0330` |
| `pinto-cpu` | Left | 10 | `0.0103` | `0.0064` | `0.0198` | `0.0208` |
| `pinto-npu` | Right | 10 | `0.0205` | `0.0313` | `0.0429` | `0.0441` |
| `pinto-npu` | Left | 10 | `0.0115` | `0.0140` | `0.0278` | `0.0285` |
| `npu-full` | Right | 10 | `0.0068` | `0.0100` | `0.0133` | `0.0154` |
| `npu-full` | Left | 10 | `0.0129` | `0.0178` | `0.0264` | `0.0308` |

Conclusion:

- PINTO NPU adoption is no longer blocked at the runtime level.
- It is a latency win over `pinto-cpu`.
- It is not yet a default-path replacement because `npu-full` remains better on right-hand agreement in this smoke comparison.

Live smoke command:

```bash
python3 main.py --backend pinto-cpu --piano --camera 0
```

Observed app banner:

```text
Air-Drum [piano] backend=PINTO-CPU:pinto_hand_landmark_sparse_Nx3x224x224.onnx
```

## STMicro Hand Landmarks

Reference:

- `https://huggingface.co/STMicroelectronics/hand_landmarks`

Assessment:

- It is interesting because it is native INT8 at the TFLite level.
- It still cannot run directly on DEEPX NPU; it needs a DeepX-compatible compile path to `.dxnn`.
- It remains a candidate if the compile server or an x86_64 local DX-COM environment becomes available.

## Compile Path and Blockers

SNU compile instructions reference:

- `https://sites.google.com/view/dxs-2603-snu/home/%EC%8B%A4%EC%8A%B55-dx-as-npu-%EC%BB%B4%ED%8C%8C%EC%9D%BC-%EB%B0%8F-%EC%B6%94%EB%A1%A0-%EC%8B%A4%EC%8A%B5`

Expected remote flow:

- Compile server: `43.203.143.33`.
- SSH/SCP port: `443`.
- User accounts: course-provided `user1` through `user17`.
- Copy ONNX and JSON to `~/sample`.
- Compile with `dx_com`.
- Retrieve output from `~/output`.

Sanitized command shape used by the repo helper:

```bash
env DX_COMPILE_USER=user12 \
  SSHPASS=<course password> \
  COMPILE_ONNX=models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx \
  COMPILE_JSON=models/dxcom/pinto_hand_landmark_sparse.json \
  COMPILE_OUT_NAME=pinto_hand_landmark_sparse.dxnn \
  ./tools/compile_server_snu.sh all
```

Observed SNU blocker:

- `ssh -p 443 ... user12@43.203.143.33 true` timed out.
- The compile helper also timed out.

Local compiler blocker:

- `/home/orangepi/dx-all-suite/dx-compiler` exists.
- Documentation/package path targets `x86_64/amd64`.
- Current board is `aarch64`, so local DX-COM compilation is not practical here.

Resolved compile route:

- An x86_64 Ubuntu 22.04.4 PC compiled the no-round fixed-shape ONNX locally.
- DX-COM CLI reported `2.3.0`; generated `.dxnn` metadata reports `v2.3.0-rc.5`.
- The board can load and run the resulting `.dxnn` through DXRT v3.2.0, but this version mismatch should remain visible in future compatibility checks.

## Files Touched This Session

Code/config changes:

- `hand_tracker.py`: added `PintoOnnxHandLandmark`, wired `pinto-cpu`, added `pinto-npu`, added tensor-binary handedness support for DXNN layouts, and added optional hand ONNX path in full tracker construction.
- `main.py`: added `pinto-cpu`, `pinto-npu`, and `--hand-onnx`; landmark correction default now empty.
- `tools/benchmark_dataset.py`: added `pinto-cpu`, `pinto-npu`, and `--hand-onnx`.
- `tools/guided_eval.py`: added `pinto-cpu`, `pinto-npu`, `--hand-onnx`, and empty landmark-correction default.
- `scripts/run_npu_full_piano.sh`: correction opt-in through `USE_LANDMARK_CORRECTION=1` or explicit `LANDMARK_CORRECTION`.
- `models/dxcom/pinto_hand_landmark_sparse.json`: compile config for PINTO ONNX.
- `models/dxnn_layout.pinto_hand_landmark_sparse.json`: runtime layout metadata for the compiled PINTO `.dxnn`.

Docs/tests:

- `README.md`
- `models/README.md`
- `tests/test_cli_defaults.py`
- `tests/test_benchmark_dataset.py`

Generated/side-effect files:

- `profiler.json` was generated by `dxbenchmark`.

Pre-existing unrelated deletions seen in git status and not touched:

- `docs/MIDTERM_PRESENTATION_2026.html`
- `docs/MIDTERM_PRESENTATION_2026.pdf`
- `docs/MIDTERM_PRESENTATION_2026_FEEDBACK_2026_05_18.md`

## Verification Commands

Unit tests:

```bash
python3 -m unittest tests.test_cli_defaults tests.test_benchmark_dataset
```

Observed result:

- `10` tests passed.

Default NPU piano launcher smoke:

```bash
./scripts/run_npu_full_piano.sh
```

Observed banner:

```text
Air-Drum [piano] backend=NPU-FULL:hand_landmark_lite.dxnn
```

No `+CALIB` suffix was present, confirming correction is off by default.

## Next Actions

1. Debug palm `.dxnn` outputs against CPU TFLite palm outputs before treating `--palm-dxnn` as valid.
2. Expand `pinto-npu` beyond the 10-frame smoke comparison before making it a default candidate.
3. Re-test STMicro native INT8 TFLite only after producing a DEEPX-compatible `.dxnn`.
4. Keep default live demo on CPU palm plus the existing MediaPipe-derived NPU hand landmark until palm-NPU detection produces accepted palms and nonzero hand-stage time.
