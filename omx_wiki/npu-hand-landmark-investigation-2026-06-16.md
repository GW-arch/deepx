# NPU Hand Landmark Investigation 2026-06-16

Tags: npu, deepx, hand-tracking, mediapipe, pinto, benchmark
Category: session-log

## Summary

The current usable `npu-full` mode is CPU palm detection plus NPU MediaPipe hand landmark. A palm-detection-on-NPU mode exists with `--palm-dxnn models/vendor/palm_detection_lite.dxnn`, but it currently produces no accepted palm detections in the benchmark path, so the hand landmark stage is skipped.

Detailed record: `air_drum_pad/docs/SESSION_2026_06_16_NPU_PINTO_AND_USAGE.md`.

## Key Measurements

Default valid `npu-full`:

- Command: `python3 tools/benchmark_dataset.py --backends npu-full --no-compare --runs 5 --warmup 1`
- Mean latency: `50.10 ms`
- Profile: `palm=40.69 ms`, `hand=9.12 ms`
- `dxtop` observation: Core0 about `1.9%` to `2.1%`, NPU memory about `6.75 MiB`

Palm-NPU experimental path:

- Command: `python3 tools/benchmark_dataset.py --backends npu-full --palm-dxnn models/vendor/palm_detection_lite.dxnn --no-compare --runs 5 --warmup 1`
- Initial mean latency: `11.22 ms`
- Repeated means: `10.39 ms`, `10.88 ms`
- Profile in all measured runs: `hand=0.00 ms`
- `dxtop` observation: Core0 about `37.7%` to `46.4%`, NPU memory about `56.2 MiB`

Interpretation:

- The palm `.dxnn` is consuming NPU.
- The palm `.dxnn` path is not a valid tracker because no hand landmark inference happens.

## Decisions

- Keep landmark correction off by default.
- Keep default demo on CPU palm plus NPU hand landmark.
- Treat `--palm-dxnn` as experimental until output-level palm postprocess is fixed.
- Treat PINTO `.onnx` as adopted for CPU experimentation only until `.dxnn` compilation succeeds.

## External References

- PINTO model zoo 033: `https://github.com/PINTO0309/PINTO_model_zoo/tree/main/033_Hand_Detection_and_Tracking`
- PINTO ONNX source: `https://github.com/PINTO0309/hand-gesture-recognition-using-onnx`
- STMicro hand landmarks: `https://huggingface.co/STMicroelectronics/hand_landmarks`
- SNU DX-AS compile instructions: `https://sites.google.com/view/dxs-2603-snu/home/%EC%8B%A4%EC%8A%B55-dx-as-npu-%EC%BB%B4%ED%8C%8C%EC%9D%BC-%EB%B0%8F-%EC%B6%94%EB%A1%A0-%EC%8B%A4%EC%8A%B5`

## Blockers

- SNU compile server `43.203.143.33:443` timed out during this session.
- Local compile on the Orange Pi is blocked because the available DX-COM path targets `x86_64/amd64`, while this board is `aarch64`.

## Next Lookup Terms

- `palm-dxnn`
- `pinto-cpu`
- `hand_landmark_lite.dxnn`
- `pinto_hand_landmark_sparse`
- `dxtop`
