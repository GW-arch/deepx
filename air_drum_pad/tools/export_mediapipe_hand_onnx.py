#!/usr/bin/env python3
"""
MediaPipe 공개 TFLite 손 랜드마크 → ONNX 변환 + 레이아웃 JSON 생성 + ORT 스모크 테스트.

원본: https://storage.googleapis.com/mediapipe-assets/hand_landmark_*.tflite

다음 단계(DX-COM): tools/compile_dxnn.sh 또는 DX-AllSuite 문서의 컴파일 명령.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# air_drum_pad/tools → parent is package root
_ROOT = Path(__file__).resolve().parents[1]

URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite",
    "full": "https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}\n  -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
        f.write(r.read())


def convert_tflite_to_onnx(tflite: Path, onnx_out: Path) -> None:
    from tflite2onnx import convert as t2o_convert

    t2o_convert(str(tflite), str(onnx_out))
    print(f"onnx written: {onnx_out} ({onnx_out.stat().st_size} bytes)", flush=True)


def onnx_io(onnx_path: Path) -> tuple[list[tuple[str, list[int | str]]], list[tuple[str, list[int | str]]]]:
    import onnx

    m = onnx.load(str(onnx_path))

    def _shape(t) -> list[int | str]:
        dims = []
        for d in t.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(str(d.dim_param))
            else:
                dims.append(-1)
        return dims

    ins = [(i.name, _shape(i)) for i in m.graph.input]
    outs = [(o.name, _shape(o)) for o in m.graph.output]
    return ins, outs


def write_layout(
    *,
    onnx_path: Path,
    layout_path: Path,
    dual_halves: bool,
    confidence_index: int | None,
    confidence_threshold: float,
    square_pad: bool,
) -> None:
    import onnx

    m = onnx.load(str(onnx_path))
    in0 = m.graph.input[0]
    name = in0.name
    shape = []
    for d in in0.type.tensor_type.shape.dim:
        if d.dim_value:
            shape.append(int(d.dim_value))
        else:
            shape.append(-1 if not d.dim_param else str(d.dim_param))

    layout: dict = {
        "input": {
            "tensor_name": name,
            "tensor_layout": "nchw" if len(shape) == 4 and int(shape[1] or 0) in (1, 3) else "auto",
            "color_order": "rgb",
            "square_pad": square_pad,
            "normalize": {"mode": "scale_255", "dtype": "float32"},
        },
        "inference": {"dual_horizontal_halves": dual_halves},
        "outputs": {
            "landmarks_tensor_index": 0,
            "points_per_hand": 21,
            "max_hands": 1,
        },
        "handedness": {"mode": "wrist_x_screen"},
        "confidence": {
            "tensor_index": confidence_index,
            "threshold": confidence_threshold,
        },
    }
    if len(shape) == 4 and (-1 in shape or any(isinstance(x, str) for x in shape)):
        # leave height/width from model if fixed; else user must fill
        pass

    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    print(f"layout written: {layout_path}", flush=True)


def ort_smoke(onnx_path: Path, in_name: str) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(onnx_path))
    in0 = next(i for i in m.graph.input if i.name == in_name)
    shape = []
    for d in in0.type.tensor_type.shape.dim:
        shape.append(int(d.dim_value) if d.dim_value else 1)
    x = np.random.rand(*shape).astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {in_name: x})
    print("ORT smoke OK:", [o.shape for o in outs], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("lite", "full"), default="lite")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "models" / "vendor",
        help="tflite/onnx/json 출력 디렉터리",
    )
    ap.add_argument(
        "--dual-halves-layout",
        action="store_true",
        help="max-hands=2 용 dual_horizontal_halves 레이아웃 JSON 을 추가로 씀",
    )
    ap.add_argument("--no-ort", action="store_true", help="onnxruntime 스모크 생략")
    ap.add_argument("--skip-download", action="store_true", help="이미 받은 tflite 재사용")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tflite = out_dir / f"hand_landmark_{args.variant}.tflite"
    onnx_path = out_dir / f"hand_landmark_{args.variant}.onnx"
    layout_path = out_dir / f"dxnn_layout.mediapipe_hand_{args.variant}.json"
    layout_dual = out_dir / f"dxnn_layout.mediapipe_hand_{args.variant}_dual.json"

    url = URLS[args.variant]
    if not args.skip_download or not tflite.is_file():
        download(url, tflite)
    else:
        print(f"reuse tflite {tflite}", flush=True)

    convert_tflite_to_onnx(tflite, onnx_path)
    ins, outs = onnx_io(onnx_path)
    print("ONNX inputs:", ins, flush=True)
    print("ONNX outputs:", outs, flush=True)

    # MediaPipe hand landmark: out[0]=63 lms, out[1]=presence scalar (typical)
    conf_idx = 1 if len(outs) >= 2 else None
    write_layout(
        onnx_path=onnx_path,
        layout_path=layout_path,
        dual_halves=False,
        confidence_index=conf_idx,
        confidence_threshold=0.5,
        square_pad=True,
    )
    if args.dual_halves_layout:
        write_layout(
            onnx_path=onnx_path,
            layout_path=layout_dual,
            dual_halves=True,
            confidence_index=conf_idx,
            confidence_threshold=0.5,
            square_pad=True,
        )

    if not args.no_ort:
        try:
            ort_smoke(onnx_path, ins[0][0])
        except Exception as e:  # pragma: no cover
            print("ORT smoke skipped/failed:", e, file=sys.stderr)

    print("\n다음: DX-COM 으로 .dxnn 생성 후", flush=True)
    print(f"  DXNN=... python3 main.py --backend npu --dxnn ... --dxnn-layout {layout_path}", flush=True)
    if args.dual_halves_layout:
        print(f"  (양손 근사) --dxnn-layout {layout_dual} --max-hands 2", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
