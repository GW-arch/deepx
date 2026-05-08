from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strike_detector import (
    FINGERTIP_INDICES,
    InstrumentStrikeDetector,
    load_instrument_slots_json,
    sound_key_for_finger,
)


class _Lm:
    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Hand:
    def __init__(self, landmarks: list[_Lm]) -> None:
        self.landmark = landmarks


def _hand_with_index_pose(tip_y: float, tip_x: float) -> _Hand:
    """Build a minimal 21-landmark hand with a controlled index chain."""
    lms = [_Lm() for _ in range(21)]
    # index chain: MCP(5), PIP(6), TIP(8)
    lms[5] = _Lm(0.40, 0.40)
    lms[6] = _Lm(0.50, 0.50)
    lms[8] = _Lm(tip_x, tip_y)
    return _Hand(lms)


class StrikeDetectorTests(unittest.TestCase):
    def test_default_sound_mapping_uses_hand_and_finger_slot(self) -> None:
        self.assertEqual(sound_key_for_finger(0, FINGERTIP_INDICES[0]), "kick")
        self.assertEqual(sound_key_for_finger(0, FINGERTIP_INDICES[1]), "snare")
        self.assertEqual(sound_key_for_finger(1, FINGERTIP_INDICES[0]), "tom_m")

    def test_custom_sound_slots_wrap_when_max_hands_is_one(self) -> None:
        slots = tuple(f"s{i}" for i in range(10))
        self.assertEqual(
            sound_key_for_finger(0, FINGERTIP_INDICES[4], max_hands=1, sound_slots=slots),
            "s4",
        )

    def test_load_instrument_slots_json_validates_keys_and_length(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "slots.json"
            path.write_text(json.dumps({"slots": ["a"] * 10}), encoding="utf-8")
            self.assertEqual(load_instrument_slots_json(str(path), frozenset({"a"})), tuple(["a"] * 10))

            path.write_text(json.dumps({"slots": ["a"] * 9}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_instrument_slots_json(str(path), frozenset({"a"}))

            path.write_text(json.dumps({"slots": ["a"] * 9 + ["bad"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_instrument_slots_json(str(path), frozenset({"a"}))

    def test_update_finger_requires_downward_motion_and_angle_change(self) -> None:
        det = InstrumentStrikeDetector(
            vy_trigger=0.10,
            joint_dps_trigger=1.0,
            cooldown_s=0.0,
            min_tip_disp=0.0,
            min_conf=0.0,
            sound_mapper=lambda _h, _lm: "snare",
        )

        self.assertIsNone(det.update_finger(0, 8, 0.0, _hand_with_index_pose(0.40, 0.60), 1.0))
        hit = det.update_finger(0, 8, 0.1, _hand_with_index_pose(0.62, 0.75), 1.0)

        self.assertEqual(hit, ("h0_index", "snare"))

    def test_low_confidence_resets_tracking(self) -> None:
        det = InstrumentStrikeDetector(min_conf=0.5)
        self.assertIsNone(det.update_finger(0, 8, 0.0, _hand_with_index_pose(0.40, 0.60), 0.1))
        self.assertEqual(det._prev_y, {})


if __name__ == "__main__":
    unittest.main()
