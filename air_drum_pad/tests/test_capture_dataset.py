from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.capture_dataset import next_frame_index


class CaptureDatasetTests(unittest.TestCase):
    def test_next_frame_index_ignores_non_matching_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(next_frame_index(str(root)), 0)

            img = np.zeros((4, 4, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "frame_000.png"), img)
            cv2.imwrite(str(root / "frame_010.png"), img)
            (root / "frame_bad.png").write_text("not an image", encoding="utf-8")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(next_frame_index(str(root)), 11)


if __name__ == "__main__":
    unittest.main()
