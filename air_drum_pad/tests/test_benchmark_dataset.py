from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools import benchmark_dataset as bd


class BenchmarkDatasetHelperTests(unittest.TestCase):
    def test_split_backends_normalizes_aliases(self) -> None:
        self.assertEqual(
            bd._split_backends("cpu_baseline,pinto_cpu,pinto_npu,npu_full,cpu"),
            ["cpu-baseline", "pinto-cpu", "pinto-npu", "npu-full", "cpu"],
        )
        with self.assertRaises(SystemExit):
            bd._split_backends("bogus")

    def test_stats_empty_and_non_empty(self) -> None:
        empty = bd._stats([])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["mean"])

        stats = bd._stats([1.0, 2.0, 3.0])
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(float(stats["mean"]), 2.0)
        self.assertAlmostEqual(float(stats["p50"]), 2.0)

    def test_paths_for_dataset_supports_single_file_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            for i in range(3):
                cv2.imwrite(str(root / f"frame_{i:03d}.png"), img)

            self.assertEqual(len(bd._paths_for_dataset(str(root), "frame_*.png", 2)), 2)
            self.assertEqual(
                bd._paths_for_dataset(str(root / "frame_000.png"), "ignored", 0),
                [root / "frame_000.png"],
            )

    def test_compare_landmarks_by_label(self) -> None:
        ref_lm = np.zeros((21, 3), dtype=np.float32)
        test_lm = ref_lm.copy()
        test_lm[:, 0] = 0.1
        all_hands = {
            "cpu-baseline": [[{"label": "Right", "landmarks": ref_lm}]],
            "npu-full": [[{"label": "Right", "landmarks": test_lm}]],
        }
        cmp = bd.compare_landmarks("cpu-baseline", all_hands)
        self.assertIn("npu-full", cmp)
        self.assertAlmostEqual(cmp["npu-full"]["Right"]["mean"]["mean"], 0.1, places=5)
        self.assertEqual(cmp["npu-full"]["Left"]["mean"]["n"], 0)

    def test_collect_error_records_and_overlay_manifest(self) -> None:
        ref_lm = np.zeros((21, 3), dtype=np.float32)
        test_lm = ref_lm.copy()
        test_lm[8, :2] = (0.2, 0.0)
        all_hands = {
            "ref": [[{"label": "Right", "landmarks": ref_lm}]],
            "test": [[{"label": "Right", "landmarks": test_lm}]],
        }
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as td:
            records = bd.collect_error_records("ref", all_hands, [Path("frame_000.png")])
            manifest = bd.save_debug_overlays(td, records, frames, top_k=1, min_error=0.0)
            self.assertEqual(len(manifest), 1)
            self.assertTrue(Path(manifest[0]["path"]).is_file())
            self.assertTrue((Path(td) / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
