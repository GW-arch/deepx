from __future__ import annotations

import unittest

import numpy as np

from tools import benchmark_dataset as bd
from tools.calibrate_npu_landmarks import _apply_correction, fit_correction


class CalibrateNpuLandmarksTests(unittest.TestCase):
    def test_fit_bias_correction_reduces_synthetic_offset(self) -> None:
        ref_lm = np.zeros((21, 3), dtype=np.float32)
        test_lm = np.zeros((21, 3), dtype=np.float32)
        for j in range(21):
            ref_lm[j, :2] = (0.4 + j * 0.001, 0.5 + j * 0.001)
            test_lm[j, :2] = ref_lm[j, :2] + (0.05, -0.02)

        ref = [[{"label": "Right", "landmarks": ref_lm}]]
        test = [[{"label": "Right", "landmarks": test_lm}]]
        before = bd.compare_landmarks("ref", {"ref": ref, "test": test})["test"]["Right"]["mean"]["mean"]

        corr = fit_correction(ref, test, kind="bias", ridge=0.0, min_samples=1)
        corrected = _apply_correction(test, corr)
        after = bd.compare_landmarks("ref", {"ref": ref, "test": corrected})["test"]["Right"]["mean"]["mean"]

        self.assertGreater(before, 0.05)
        self.assertLess(after, 1e-5)

    def test_affine_schema_has_21_transforms_per_label(self) -> None:
        ref_frames = []
        test_frames = []
        for i in range(8):
            ref_lm = np.zeros((21, 3), dtype=np.float32)
            test_lm = np.zeros((21, 3), dtype=np.float32)
            ref_lm[:, 0] = 0.2 + i * 0.01
            ref_lm[:, 1] = 0.3 + i * 0.01
            test_lm[:, 0] = ref_lm[:, 0] + 0.01
            test_lm[:, 1] = ref_lm[:, 1] - 0.01
            ref_frames.append([{"label": "Left", "landmarks": ref_lm}])
            test_frames.append([{"label": "Left", "landmarks": test_lm}])

        corr = fit_correction(ref_frames, test_frames, kind="affine", ridge=1e-4, min_samples=3)

        self.assertEqual(corr["type"], "affine_xy")
        self.assertEqual(len(corr["labels"]["Left"]), 21)
        self.assertEqual(np.asarray(corr["labels"]["Left"][0]["matrix"]).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
