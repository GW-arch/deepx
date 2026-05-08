from __future__ import annotations

import unittest

from tools import sweep_palm_redetect as sweep


class SweepPalmRedetectTests(unittest.TestCase):
    def test_parse_values(self) -> None:
        self.assertEqual(sweep.parse_values("0, 1,5"), [0, 1, 5])
        with self.assertRaises(SystemExit):
            sweep.parse_values("")
        with self.assertRaises(SystemExit):
            sweep.parse_values("-1")

    def test_compact_rows_includes_error_columns(self) -> None:
        result = {
            "palm_redetect_every": 2,
            "async_palm": False,
            "compare_ref": "cpu-baseline",
            "summary": {
                "npu-full": {
                    "latency_ms": {"mean": 10.0, "p95": 12.0, "min": 9.0, "max": 14.0},
                    "palm_ms": {"mean": 3.0},
                    "hand_ms": {"mean": 7.0},
                    "async_palm_ms": {"mean": 0.0},
                    "palm_wait_ms": {"mean": 0.0},
                    "modes": {"tracking": 4, "palm": 1},
                }
            },
            "landmark_comparison": {
                "npu-full": {
                    "Right": {
                        "mean": {"mean": 0.01, "n": 5},
                        "tips": {"mean": 0.02},
                        "max": {"max": 0.03},
                    },
                    "Left": {
                        "mean": {"mean": 0.04, "n": 5},
                        "tips": {"mean": 0.05},
                        "max": {"max": 0.06},
                    },
                }
            },
        }

        rows = sweep.compact_rows(result)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["palm_redetect_every"], 2)
        self.assertEqual(row["backend"], "npu-full")
        self.assertAlmostEqual(float(row["right_err_mean"]), 0.01)
        self.assertAlmostEqual(float(row["left_err_max"]), 0.06)


if __name__ == "__main__":
    unittest.main()
