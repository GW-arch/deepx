# Session Report — Palm DXNN Local Recompile Validation

Date: 2026-06-16

## Summary

The locally compiled palm detector DXNN artifacts were transferred to the Orange Pi board and validated with the existing offline replay dataset. Both local DX-COM 2.3.0 variants produced accepted palm detections and reached the hand-landmark stage. The `minmax_local` variant is the representative full-NPU candidate for follow-up testing.

## Local Compile Environment

| Item | Value |
|------|-------|
| Host | x86_64, Ubuntu 22.04.4 LTS |
| DX-COM | `/home/jdaye54/dxcom-venv-cpu/bin/dxcom` |
| Version | DX-COM 2.3.0, target M1 |

Compile commands recorded by the local PC agent:

```bash
/home/jdaye54/dxcom-venv-cpu/bin/dxcom \
  -m models/vendor/palm_detection_lite.onnx \
  -c models/dxcom/palm_detection_lite_ema.local.json \
  -o build/dxcom/palm_detection_lite_ema_local \
  --gen_log

/home/jdaye54/dxcom-venv-cpu/bin/dxcom \
  -m models/vendor/palm_detection_lite.onnx \
  -c models/dxcom/palm_detection_lite_minmax.local.json \
  -o build/dxcom/palm_detection_lite_minmax_local \
  --gen_log
```

## Transferred Artifacts

Source bundle on board:

```text
/home/orangepi/deepx/palm_dxnn_transfer/
```

Runtime copies:

```text
/home/orangepi/deepx/air_drum_pad/models/vendor/palm_detection_lite_ema_local.dxnn
/home/orangepi/deepx/air_drum_pad/models/vendor/palm_detection_lite_minmax_local.dxnn
```

Checksums:

| File | SHA-256 |
|------|---------|
| `models/vendor/palm_detection_lite_ema_local.dxnn` | `020df4a9fb8a566cedf97901469750fe7ec0ac92fdbf72cb0255579f5a6adff2` |
| `models/vendor/palm_detection_lite_minmax_local.dxnn` | `72b9e913f294abf5c13380f086d02cc332dafe0d942aaa41d460e275a0768f1f` |
| `palm_detection_lite.onnx` | `dbbb495072e8a1a6b4b4372f90d3a5797f511f0c8837b90ee0ff2d1b9be741c0` |
| `palm_detection_lite_ema_local.compiler.log` | `52338aa09eafac544a45284291b02b28fd49a881d5d6211fb8914d9e191e10a7` |
| `palm_detection_lite_minmax_local.compiler.log` | `3daae75701ccfe17519f96ae18c6a589c234f6a88275fbb05aa0c5b2de498d5b` |

## Tensor Validation

Command shape:

```bash
python3 tools/debug_palm_outputs.py \
  --backend all \
  --image dataset/frame_000.png \
  --dxnn models/vendor/palm_detection_lite_minmax_local.dxnn \
  --score-thresh 0.5 \
  --dxnn-input-variant nhwc_u8
```

Board-side results:

| Candidate | Frame | TFLite detections | DXNN detections | DXNN score corr vs TFLite | DXNN box corr vs TFLite |
|-----------|-------|------------------:|----------------:|--------------------------:|------------------------:|
| `minmax_local` | `frame_000` | 2 | 2 | 0.9809 | 0.9937 |
| `minmax_local` | `frame_060` | 3 | 2 | 0.9780 | 0.9926 |
| `ema_local` | `frame_000` | 2 | 2 | 0.9809 | 0.9937 |
| `ema_local` | `frame_060` | 3 | 2 | 0.9780 | 0.9926 |

## Offline Benchmark

10-frame smoke with `minmax_local`:

```bash
python3 tools/benchmark_dataset.py \
  --backends npu-full \
  --palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn \
  --limit 10 \
  --warmup 0
```

Result:

| Backend | Frames | Mean | P95 | Profile |
|---------|-------:|-----:|----:|---------|
| `npu-full --palm-dxnn palm_detection_lite_minmax_local.dxnn` | 10 | 24.90 ms | 33.48 ms | palm 10.87 ms + hand 13.46 ms |

90-frame comparison:

```bash
python3 tools/benchmark_dataset.py \
  --backends cpu-baseline,npu-full \
  --palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn \
  --limit 90 \
  --warmup 0 \
  --csv build/benchmarks/palm_minmax_local_90_compare.csv \
  --json build/benchmarks/palm_minmax_local_90_compare.json
```

Latency:

| Backend | Frames | Mean | P50 | P95 | Profile |
|---------|-------:|-----:|----:|----:|---------|
| `cpu-baseline` | 90 | 83.47 ms | 83.04 ms | 83.30 ms | palm 38.80 ms + hand 44.27 ms |
| `npu-full --palm-dxnn palm_detection_lite_minmax_local.dxnn` | 90 | 22.91 ms | 22.49 ms | 25.71 ms | palm 9.83 ms + hand 12.62 ms |

Landmark agreement vs `cpu-baseline`:

| Hand | Matching frames | Mean error | Fingertip error | Mean max error | Worst max error |
|------|----------------:|-----------:|----------------:|---------------:|----------------:|
| Right | 89 | 0.0225 | 0.0262 | 0.0518 | 0.1045 |
| Left | 90 | 0.0166 | 0.0244 | 0.0347 | 0.0559 |

## Conclusion

`palm_detection_lite_minmax_local.dxnn` is the current best full-NPU candidate for the offline dataset: it runs palm and hand stages on NPU and reduces mean replay latency to `22.91 ms` while keeping landmark agreement close to the CPU-palm `npu-full` path.

Remaining validation:

- Live camera stability with `--palm-dxnn models/vendor/palm_detection_lite_minmax_local.dxnn`.
- Guided drum/piano hit-accuracy runs using the full-NPU candidate.
- Raw `dxtop` utilization capture while running the full-NPU path.
