from __future__ import annotations

import unittest

import numpy as np

from tools.palm_roi import inverse_landmark_transform, warp_roi_affine


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


if __name__ == "__main__":
    unittest.main()
