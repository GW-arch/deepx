# How to Compile the PINTO Hand Landmark Model on a PC

Date: 2026-06-16

This document is written as an execution handoff for an agent running on a developer PC. The goal is to compile the PINTO sparse hand-landmark ONNX model into a DEEPX `.dxnn` file without using the SNU compile server.

## Goal

Produce:

```text
models/vendor/pinto_hand_landmark_sparse.dxnn
```

from:

```text
models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx
models/dxcom/pinto_hand_landmark_sparse.json
```

## Important Constraints

- Do this on a **Linux x86_64/amd64 PC**, not on the Orange Pi board.
- Ubuntu **20.04, 22.04, or 24.04 x64** is the expected environment.
- WSL2 Ubuntu x86_64 is acceptable if DX-COM installs and runs there.
- `aarch64/arm64` is **not supported** for DX-COM compilation, so the Orange Pi cannot compile this locally.
- The NPU device is not required for compilation. It is required later for runtime testing of the generated `.dxnn`.
- DEEPX compiler access may require DEEPX portal credentials or a pre-downloaded DX-COM package.
- Keep compiler/runtime compatibility in mind. The target board currently reports DXRT `v3.2.0`; existing MediaPipe `.dxnn` metadata was compiled with DX-COM `v2.1.0-rc.4`.

## Step 1: Check PC Architecture

Run:

```bash
uname -m
lsb_release -a
```

Expected:

```text
x86_64
Ubuntu 20.04/22.04/24.04
```

Stop if `uname -m` is `aarch64` or `arm64`.

## Step 2: Install or Activate DX-COM

Use one of these two methods.

### Option A: DX-COM Wheel

Install the wheel that matches the local Python version:

```bash
python3 --version
python3 -m pip install dx_com-2.2.0-cp<VERSION>-cp<VERSION>-linux_x86_64.whl
```

Example for Python 3.11:

```bash
python3 -m pip install dx_com-2.2.0-cp311-cp311-linux_x86_64.whl
```

Verify:

```bash
python3 -c "import dx_com; print(dx_com.__version__)"
dxcom --version
```

### Option B: DX-COM Executable Package

Extract the compiler package:

```bash
tar xfz dx_com_M1_vx.x.x.tar.gz
./dx_com/dx_com --version
```

Use `./dx_com/dx_com` instead of `dxcom` in later commands.

## Step 3: Prepare This Repository

Clone or open the repo on the PC:

```bash
git clone git@github.com:GW-arch/deepx.git
cd deepx/air_drum_pad
```

Confirm the compile config exists:

```bash
test -f models/dxcom/pinto_hand_landmark_sparse.json
test -f models/dxnn_layout.pinto_hand_landmark_sparse.json
```

## Step 4: Get the PINTO ONNX Model

The ONNX model is intentionally not committed to git because vendor model files are ignored. Use one of the following methods.

### Option A: Copy from the Orange Pi

If the board already has the model:

```bash
mkdir -p models/vendor
scp orangepi@<BOARD_IP>:/home/orangepi/deepx/air_drum_pad/models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx models/vendor/
```

Expected checksum from the board copy used in this project:

```text
9fcea307b52350eec3366cd6ad4eb11f89cff58f2a482a60ec8704d0e012e63a  pinto_hand_landmark_sparse_Nx3x224x224.onnx
```

Verify:

```bash
sha256sum models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx
```

### Option B: Download from PINTO's ONNX Repository

Clone the source repository and locate the model:

```bash
tmpdir="$(mktemp -d)"
git clone --depth 1 https://github.com/PINTO0309/hand-gesture-recognition-using-onnx "$tmpdir/pinto-hand-onnx"
find "$tmpdir/pinto-hand-onnx" -name '*hand*landmark*sparse*Nx3x224x224*.onnx' -print
```

Copy it into this repo:

```bash
mkdir -p models/vendor
cp "$(find "$tmpdir/pinto-hand-onnx" -name '*hand*landmark*sparse*Nx3x224x224*.onnx' -print -quit)" \
  models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx
sha256sum models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx
```

The ONNX IO expected by this project is:

```text
input: input, shape [N, 3, 224, 224]
outputs:
  xyz_x21, shape [N, 63]
  hand_score, shape [N, 1]
  lefthand_0_or_righthand_1, shape [N, 1]
```

## Step 5: Prepare Calibration Data

DX-COM needs calibration images for INT8 compilation. The committed config currently contains the SNU server path:

```text
/home/dxs/dx_com_M1_v2.1.0-rc.4/calibration_dataset
```

On the PC, create a local config copy with a local calibration dataset path.

If using DX-All-Suite, sample calibration data is usually downloaded under the DX-COM install tree. Locate it:

```bash
find "$HOME" -type d -name calibration_dataset 2>/dev/null | head
```

Set the chosen path:

```bash
export CALIB_DIR="/absolute/path/to/calibration_dataset"
test -d "$CALIB_DIR"
```

Create a PC-local compile config:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

calib = os.environ["CALIB_DIR"]
src = Path("models/dxcom/pinto_hand_landmark_sparse.json")
dst = Path("models/dxcom/pinto_hand_landmark_sparse.local.json")
cfg = json.loads(src.read_text())
cfg["default_loader"]["dataset_path"] = calib
dst.write_text(json.dumps(cfg, indent=2) + "\n")
print(dst)
print("dataset_path =", calib)
PY
```

Sanity-check:

```bash
grep -n "dataset_path" models/dxcom/pinto_hand_landmark_sparse.local.json
```

Do not commit `pinto_hand_landmark_sparse.local.json` unless the path is made portable.

## Step 6: Compile with DX-COM

Create an output directory:

```bash
mkdir -p build/dxcom/pinto_hand_landmark_sparse
```

If using the wheel CLI:

```bash
dxcom \
  -m models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx \
  -c models/dxcom/pinto_hand_landmark_sparse.local.json \
  -o build/dxcom/pinto_hand_landmark_sparse \
  --gen_log
```

If using the executable package:

```bash
./dx_com/dx_com \
  -m models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx \
  -c models/dxcom/pinto_hand_landmark_sparse.local.json \
  -o build/dxcom/pinto_hand_landmark_sparse \
  --gen_log
```

Find the output:

```bash
find build/dxcom/pinto_hand_landmark_sparse -name '*.dxnn' -print
find build/dxcom/pinto_hand_landmark_sparse -name 'compiler.log' -print
```

Copy the final `.dxnn` into the expected project location:

```bash
mkdir -p models/vendor
cp "$(find build/dxcom/pinto_hand_landmark_sparse -name '*.dxnn' -print -quit)" \
  models/vendor/pinto_hand_landmark_sparse.dxnn
sha256sum models/vendor/pinto_hand_landmark_sparse.dxnn
```

## Step 7: If Compile Fails on Dynamic Batch

The source model name includes `Nx3x224x224`, and the ONNX input may contain a dynamic batch dimension. The committed JSON pins the input to `[1, 3, 224, 224]`, but if DX-COM still rejects the dynamic input shape, create a fixed-shape ONNX copy.

Try ONNX simplifier:

```bash
python3 -m pip install onnx onnxsim
python3 -m onnxsim \
  models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx \
  models/vendor/pinto_hand_landmark_sparse_1x3x224x224.onnx \
  --overwrite-input-shape input:1,3,224,224
```

Then compile the fixed model:

```bash
dxcom \
  -m models/vendor/pinto_hand_landmark_sparse_1x3x224x224.onnx \
  -c models/dxcom/pinto_hand_landmark_sparse.local.json \
  -o build/dxcom/pinto_hand_landmark_sparse_fixed \
  --gen_log
```

## Step 8: Transfer the Compiled Model to the Board

From the PC:

```bash
scp models/vendor/pinto_hand_landmark_sparse.dxnn \
  orangepi@<BOARD_IP>:/home/orangepi/deepx/air_drum_pad/models/vendor/
```

On the board:

```bash
cd /home/orangepi/deepx/air_drum_pad
ls -lh models/vendor/pinto_hand_landmark_sparse.dxnn
```

## Step 9: Board-Side Smoke Test

The app currently has a `pinto-cpu` backend. A NPU-specific PINTO runtime adapter still needs to be wired after the `.dxnn` exists. First verify that the board can load the compiled model with DXRT:

```bash
python3 - <<'PY'
from dx_engine import InferenceEngine
import numpy as np

path = "models/vendor/pinto_hand_landmark_sparse.dxnn"
ie = InferenceEngine(path)
print("inputs:", ie.get_input_tensors_info())
x = np.zeros((1, 3, 224, 224), dtype=np.float32)
outs = ie.run([x])
for i, out in enumerate(outs):
    arr = np.asarray(out)
    print(i, arr.shape, arr.dtype, arr.reshape(-1)[:5])
ie.dispose()
PY
```

If the input tensor info reports `uint8` or NHWC instead of `float32` NCHW, adjust the smoke input shape/dtype to match `ie.get_input_tensors_info()`.

## Success Criteria

Compilation is successful when all of the following are true:

- `models/vendor/pinto_hand_landmark_sparse.dxnn` exists.
- `compiler.log` exists and does not report a fatal compile error.
- The board can construct `InferenceEngine("models/vendor/pinto_hand_landmark_sparse.dxnn")`.
- A zero-input smoke inference returns three model outputs or otherwise returns outputs that can be mapped to:
  - landmarks `[1, 63]`
  - hand score `[1, 1]`
  - handedness `[1, 1]`

## Expected Follow-Up After Compile

After the `.dxnn` is available, implement a `pinto-npu` backend in `hand_tracker.py` by adapting the current `PintoOnnxHandLandmark` preprocessing/postprocessing to `dx_engine`.

The current CPU PINTO reference path is:

```bash
python3 tools/benchmark_dataset.py --backends pinto-cpu --no-compare --limit 10 --warmup 0
```

Previous board measurement:

```text
mean 88.91 ms
p50 86.57 ms
p95 97.13 ms
profile: palm=40.19 ms, hand=48.27 ms
```

The NPU follow-up should compare `pinto-npu` against both `pinto-cpu` and the current default `npu-full`.

## Troubleshooting Notes

- If `dxcom` is not found, activate the DX-COM Python environment or use the executable path `./dx_com/dx_com`.
- If the calibration dataset path fails, create `models/dxcom/pinto_hand_landmark_sparse.local.json` again with an absolute local image directory.
- If DX-COM reports unsupported dynamic input shape, use the fixed-shape ONNX fallback in Step 7.
- If the compiled model loads but output order differs, inspect it with `parse_model` or `dx_engine` and update `models/dxnn_layout.pinto_hand_landmark_sparse.json`.
- Do not assume the generated `.dxnn` is accuracy-equivalent to `pinto-cpu`; benchmark it on the board.
- Do not treat this as solving the palm-NPU problem. This only compiles the PINTO hand-landmark model.
