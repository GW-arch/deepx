from __future__ import annotations

import math
import unittest

import numpy as np

from tools.palm_roi import palm_detection_to_roi, inverse_landmark_transform, warp_roi_affine


class PalmRoiTests(unittest.TestCase):
    def test_inverse_landmark_transform_center_no_rotation(self) -> None:
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[:, 0] = 0.5
        lm[:, 1] = 0.5
        out = inverse_landmark_transform(
            lm,
            center_x=320,
            center_y=240,
            roi_size=224,
            rotation=0.0,
            image_w=640,
            image_h=480,
        )
        self.assertAlmostEqual(float(out[0, 0]), 0.5)
        self.assertAlmostEqual(float(out[0, 1]), 0.5)

    def test_inverse_landmark_transform_top_left_no_rotation(self) -> None:
        lm = np.zeros((1, 3), dtype=np.float32)
        out = inverse_landmark_transform(
            lm,
            center_x=320,
            center_y=240,
            roi_size=224,
            rotation=0.0,
            image_w=640,
            image_h=480,
        )
        self.assertAlmostEqual(float(out[0, 0]), (320 - 112) / 640)
        self.assertAlmostEqual(float(out[0, 1]), (240 - 112) / 480)

    def test_warp_roi_affine_shape_and_dtype(self) -> None:
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        rgb[5:15, 10:20] = (255, 0, 0)
        out = warp_roi_affine(rgb, 15, 10, 20, 0.0, out_size=16)
        self.assertEqual(out.shape, (16, 16, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_palm_detection_to_roi_upright_hand_has_zero_rotation_and_upward_shift(self) -> None:
        det = np.zeros(19, dtype=np.float32)
        # box: x=[0.40,0.60], y=[0.40,0.60]
        det[1:5] = [0.40, 0.40, 0.60, 0.60]
        # wrist below middle MCP: fingers point upward in image coordinates.
        det[5 + 0 * 2] = 0.50
        det[5 + 0 * 2 + 1] = 0.60
        det[5 + 2 * 2] = 0.50
        det[5 + 2 * 2 + 1] = 0.40

        cx, cy, roi_size, rotation = palm_detection_to_roi(det, 640, 480)

        self.assertAlmostEqual(rotation, 0.0, places=5)
        self.assertAlmostEqual(cx, 320.0, places=4)
        self.assertLess(cy, 240.0)
        self.assertAlmostEqual(roi_size, max(0.20 * 640, 0.20 * 480) * 2.6, places=3)

    def test_palm_detection_to_roi_rightward_hand_aligns_to_y_axis(self) -> None:
        det = np.zeros(19, dtype=np.float32)
        det[1:5] = [0.40, 0.40, 0.60, 0.60]
        det[5 + 0 * 2] = 0.40
        det[5 + 0 * 2 + 1] = 0.50
        det[5 + 2 * 2] = 0.60
        det[5 + 2 * 2 + 1] = 0.50

        _cx, _cy, _roi_size, rotation = palm_detection_to_roi(det, 640, 480)

        self.assertAlmostEqual(rotation, math.pi / 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
