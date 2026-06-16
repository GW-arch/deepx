from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import main as live_main
from tools import guided_eval


class CliDefaultTests(unittest.TestCase):
    def test_live_cli_does_not_apply_landmark_correction_by_default(self) -> None:
        with patch.object(sys, "argv", ["main.py"]):
            args = live_main.parse_args()

        self.assertEqual(args.backend, "npu-full")
        self.assertEqual(args.landmark_correction, "")

    def test_live_cli_accepts_explicit_landmark_correction(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["main.py", "--landmark-correction", "models/npu_landmark_correction.bias.json"],
        ):
            args = live_main.parse_args()

        self.assertEqual(args.landmark_correction, "models/npu_landmark_correction.bias.json")

    def test_live_cli_accepts_pinto_cpu_backend(self) -> None:
        with patch.object(sys, "argv", ["main.py", "--backend", "pinto-cpu"]):
            args = live_main.parse_args()

        self.assertEqual(args.backend, "pinto-cpu")
        self.assertTrue(args.hand_onnx.endswith("pinto_hand_landmark_sparse_Nx3x224x224.onnx"))

    def test_live_cli_accepts_pinto_npu_backend(self) -> None:
        with patch.object(sys, "argv", ["main.py", "--backend", "pinto-npu"]):
            args = live_main.parse_args()

        self.assertEqual(args.backend, "pinto-npu")

    def test_guided_eval_does_not_apply_landmark_correction_by_default(self) -> None:
        args = guided_eval.parse_args([])

        self.assertEqual(args.landmark_correction, "")


if __name__ == "__main__":
    unittest.main()
