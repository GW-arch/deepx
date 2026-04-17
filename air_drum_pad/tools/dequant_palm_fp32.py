#!/usr/bin/env python3
"""Dequantize palm_detection_lite.tflite fp16 → fp32, then convert to ONNX.

Pipeline:
  1. flatc → JSON (already done externally)
  2. Patch JSON: FLOAT16 → FLOAT32, expand buffer data fp16→fp32
  3. flatc → binary TFLite
  4. tflite2onnx → ONNX
  5. Verify with onnxruntime + tflite_runtime numerical comparison

Usage:
    # Prerequisites: flatc, schema.fbs, JSON already generated
    python3 tools/dequant_palm_fp32.py
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from base64 import b64decode, b64encode
from pathlib import Path

import numpy as np


def patch_json_fp16_to_fp32(json_path: Path, out_path: Path) -> int:
    """Read TFLite JSON, change all FLOAT16 tensor types to FLOAT32,
    expand their buffer data from fp16 to fp32,
    and remove DEQUANTIZE ops that converted fp16→fp32.

    Returns number of tensors patched.
    """
    print(f"Reading {json_path} ...", flush=True)
    data = json.loads(json_path.read_text())

    # Build a set of buffer indices that belong to FLOAT16 tensors
    fp16_buffer_indices: set[int] = set()
    patched = 0

    for sg in data.get("subgraphs", []):
        for tensor in sg.get("tensors", []):
            if tensor.get("type") == "FLOAT16":
                tensor["type"] = "FLOAT32"
                buf_idx = tensor.get("buffer", -1)
                if buf_idx >= 0:
                    fp16_buffer_indices.add(buf_idx)
                patched += 1

    print(f"Patched {patched} tensor types FLOAT16→FLOAT32", flush=True)
    print(f"Buffer indices to expand: {len(fp16_buffer_indices)}", flush=True)

    # Expand buffer data: fp16 bytes → fp32 bytes
    buffers = data.get("buffers", [])
    expanded = 0
    for idx in fp16_buffer_indices:
        if idx >= len(buffers):
            continue
        buf = buffers[idx]
        buf_data = buf.get("data")
        if not buf_data:
            continue
        raw = bytes(buf_data)
        fp16_arr = np.frombuffer(raw, dtype=np.float16)
        fp32_arr = fp16_arr.astype(np.float32)
        buf["data"] = list(fp32_arr.tobytes())
        expanded += 1

    print(f"Expanded {expanded} buffers fp16→fp32", flush=True)

    # Remove DEQUANTIZE ops: they were converting fp16 weights to fp32.
    # Now that weights are fp32, these ops are identity and will error.
    # For each DEQUANTIZE op: redirect all consumers of its output
    # to use its input instead, then delete the op.
    opcodes = data.get("operator_codes", [])
    dequant_opcode_indices = set()
    for i, oc in enumerate(opcodes):
        bc = oc.get("builtin_code")
        # DEQUANTIZE can appear as "DEQUANTIZE" or numeric 6
        if bc == "DEQUANTIZE" or bc == 6:
            dequant_opcode_indices.add(i)
        # Also check deprecated_builtin_code
        dbc = oc.get("deprecated_builtin_code")
        if dbc == 6:
            dequant_opcode_indices.add(i)

    print(f"DEQUANTIZE opcode indices: {dequant_opcode_indices}", flush=True)

    for sg in data.get("subgraphs", []):
        ops = sg.get("operators", [])
        # Build input→output mapping for DEQUANTIZE ops
        # DEQUANTIZE has 1 input and 1 output
        remap: dict[int, int] = {}  # output_tensor_idx → input_tensor_idx
        new_ops = []
        removed = 0
        for op in ops:
            if op.get("opcode_index") in dequant_opcode_indices:
                inp = op.get("inputs", [])
                out = op.get("outputs", [])
                if len(inp) == 1 and len(out) == 1:
                    remap[out[0]] = inp[0]
                    removed += 1
                    continue
            new_ops.append(op)

        print(f"Removed {removed} DEQUANTIZE ops, remapping {len(remap)} tensor refs", flush=True)

        # Remap tensor references in remaining ops
        remapped_count = 0
        for op in new_ops:
            for key in ("inputs", "outputs"):
                arr = op.get(key, [])
                for i, tid in enumerate(arr):
                    if tid in remap:
                        arr[i] = remap[tid]
                        remapped_count += 1

        # Also remap subgraph outputs
        for key in ("inputs", "outputs"):
            arr = sg.get(key, [])
            for i, tid in enumerate(arr):
                if tid in remap:
                    arr[i] = remap[tid]
                    remapped_count += 1

        print(f"Remapped {remapped_count} tensor references", flush=True)
        sg["operators"] = new_ops

    print(f"Writing {out_path} ...", flush=True)
    out_path.write_text(json.dumps(data))
    print(f"Done ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return patched


def json_to_tflite(schema_fbs: Path, json_path: Path, out_dir: Path) -> Path:
    """flatc --binary: JSON → TFLite binary."""
    cmd = [
        "flatc", "--binary", "-o", str(out_dir),
        str(schema_fbs), str(json_path),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)
    # Output filename = json stem + .bin (flatc default)
    out = out_dir / json_path.with_suffix(".bin").name
    if not out.exists():
        # Try .tflite
        out = out_dir / json_path.with_suffix(".tflite").name
    if not out.exists():
        # List what was created
        created = list(out_dir.iterdir())
        print(f"Files in {out_dir}: {[f.name for f in created]}", flush=True)
        out = created[0] if created else out
    return out


def tflite2onnx_convert(tflite_path: Path, onnx_path: Path) -> None:
    """Run tflite2onnx conversion."""
    import tflite2onnx
    print(f"tflite2onnx: {tflite_path} → {onnx_path}", flush=True)
    tflite2onnx.convert(str(tflite_path), str(onnx_path))
    print(f"ONNX written: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)", flush=True)


def verify_onnx(
    onnx_path: Path,
    orig_tflite_path: Path,
) -> bool:
    """Verify ONNX model against original TFLite: same input → same output."""
    import onnxruntime as ort
    import tflite_runtime.interpreter as tflite

    print("\n--- Verification ---", flush=True)

    # TFLite reference
    tfl = tflite.Interpreter(model_path=str(orig_tflite_path))
    tfl.allocate_tensors()
    tfl_inp = tfl.get_input_details()[0]
    tfl_outs = tfl.get_output_details()

    # ONNX
    sess = ort.InferenceSession(str(onnx_path))
    ort_inp = sess.get_inputs()[0]
    ort_outs = sess.get_outputs()

    print(f"TFLite input: {tfl_inp['shape']} {tfl_inp['dtype']}", flush=True)
    print(f"ONNX   input: {ort_inp.shape} {ort_inp.type}", flush=True)

    # Random test input (NHWC for TFLite)
    test_nhwc = np.random.rand(1, 192, 192, 3).astype(np.float32)
    # NCHW for ONNX (tflite2onnx transposes layout)
    test_nchw = test_nhwc.transpose(0, 3, 1, 2)

    tfl.set_tensor(tfl_inp["index"], test_nhwc)
    tfl.invoke()
    tfl_results = [tfl.get_tensor(o["index"]) for o in tfl_outs]

    ort_results = sess.run(None, {ort_inp.name: test_nchw})

    ok = True
    for i, (tr, orr) in enumerate(zip(tfl_results, ort_results)):
        # Shapes might differ in layout (NHWC vs NCHW) — compare after flatten
        tr_flat = tr.flatten()
        orr_flat = orr.flatten()
        if tr_flat.shape != orr_flat.shape:
            print(f"  output[{i}] shape mismatch: TFLite {tr.shape} vs ONNX {orr.shape}")
            ok = False
            continue
        max_diff = np.abs(tr_flat - orr_flat).max()
        mean_diff = np.abs(tr_flat - orr_flat).mean()
        print(f"  output[{i}]: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if max_diff > 0.01:
            print(f"    WARNING: large difference!")
            ok = False

    return ok


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    schema_fbs = Path("/tmp/schema.fbs")
    json_orig = Path("/tmp/palm_json/palm_detection_lite.json")
    json_patched = Path("/tmp/palm_json/palm_detection_lite_fp32.json")
    tflite_orig = root / "models" / "vendor" / "palm_detection_lite.tflite"
    onnx_out = root / "models" / "vendor" / "palm_detection_lite.onnx"

    if not schema_fbs.exists():
        print(f"Missing {schema_fbs} — run: curl -sL -o /tmp/schema.fbs "
              "'https://raw.githubusercontent.com/tensorflow/tensorflow/v2.14.0/"
              "tensorflow/lite/schema/schema.fbs'")
        return 1

    if not json_orig.exists():
        print(f"Missing {json_orig} — run: flatc --json --strict-json --raw-binary "
              f"-o /tmp/palm_json {schema_fbs} -- {tflite_orig}")
        return 1

    # Step 1: Patch JSON
    patch_json_fp16_to_fp32(json_orig, json_patched)

    # Step 2: JSON → TFLite binary
    tflite_fp32 = json_to_tflite(schema_fbs, json_patched, Path("/tmp/palm_json"))
    print(f"FP32 TFLite: {tflite_fp32} ({tflite_fp32.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity: load with tflite_runtime
    import tflite_runtime.interpreter as tfl_rt
    intr = tfl_rt.Interpreter(model_path=str(tflite_fp32))
    intr.allocate_tensors()
    details = intr.get_tensor_details()
    fp16_left = sum(1 for d in details if "float16" in str(d["dtype"]))
    print(f"FP16 tensors remaining: {fp16_left}")
    if fp16_left > 0:
        print("WARNING: still has fp16 tensors — ONNX conversion may fail")

    # Step 3: tflite2onnx
    try:
        tflite2onnx_convert(tflite_fp32, onnx_out)
    except Exception as e:
        print(f"tflite2onnx failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Verify
    try:
        ok = verify_onnx(onnx_out, tflite_orig)
        if ok:
            print("\n✓ ONNX model verified — outputs match TFLite within tolerance")
        else:
            print("\n⚠ ONNX model has numerical differences (may still be usable for DX-COM)")
    except Exception as e:
        print(f"Verification error: {e}")

    print(f"\nOutput: {onnx_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
