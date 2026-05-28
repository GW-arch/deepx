from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strike_detector import (
    FINGERTIP_INDICES,
    FINGER_ANGLE_CHAIN,
    InstrumentStrikeDetector,
    PadStrikeDetector,
    PadZone,
    default_pad_zones,
    load_instrument_slots_json,
    load_pad_zones_json,
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


def _hand_with_tip_pose(tip_id: int, tip_y: float, tip_x: float) -> _Hand:
    """Build a minimal 21-landmark hand with a controlled chain for any fingertip."""
    lms = [_Lm() for _ in range(21)]
    a, b, c = FINGER_ANGLE_CHAIN[tip_id]
    lms[a] = _Lm(0.40, 0.40)
    lms[b] = _Lm(0.50, 0.50)
    lms[c] = _Lm(tip_x, tip_y)
    return _Hand(lms)


def _translated_hand_with_index_pose(
    *,
    wrist_y: float,
    tip_y: float,
    tip_x: float,
) -> _Hand:
    """Index pose with explicit wrist y for whole-hand ghost-strike tests."""
    lms = [_Lm() for _ in range(21)]
    lms[0] = _Lm(0.50, wrist_y)
    lms[5] = _Lm(0.40, wrist_y - 0.10)
    lms[6] = _Lm(0.50, wrist_y)
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

    def test_update_finger_rejects_whole_hand_translation_ghost(self) -> None:
        det = InstrumentStrikeDetector(
            vy_trigger=0.10,
            joint_dps_trigger=1.0,
            cooldown_s=0.0,
            min_tip_disp=0.0,
            min_conf=0.0,
            sound_mapper=lambda _h, _lm: "snare",
        )

        self.assertIsNone(
            det.update_finger(
                0,
                8,
                0.0,
                _translated_hand_with_index_pose(wrist_y=0.50, tip_y=0.60, tip_x=0.60),
                1.0,
            )
        )
        # Fingertip moves down in image space and the angle changes, but the
        # wrist moves down by the same amount.  This is a whole-hand/model jump,
        # not a local finger strike.
        self.assertIsNone(
            det.update_finger(
                0,
                8,
                0.1,
                _translated_hand_with_index_pose(wrist_y=0.55, tip_y=0.65, tip_x=0.72),
                1.0,
            )
        )

    def test_middle_finger_uses_more_sensitive_threshold(self) -> None:
        base_args = dict(
            vy_trigger=0.001,
            joint_dps_trigger=20.0,
            cooldown_s=0.0,
            min_tip_disp=0.0,
            min_conf=0.0,
            sound_mapper=lambda _h, _lm: "snare",
        )

        index_det = InstrumentStrikeDetector(**base_args)
        self.assertIsNone(index_det.update_finger(0, 8, 0.0, _hand_with_tip_pose(8, 0.550, 0.60), 1.0))
        self.assertIsNone(index_det.update_finger(0, 8, 0.1, _hand_with_tip_pose(8, 0.559, 0.61), 1.0))

        middle_det = InstrumentStrikeDetector(**base_args)
        self.assertIsNone(middle_det.update_finger(0, 12, 0.0, _hand_with_tip_pose(12, 0.550, 0.60), 1.0))
        self.assertEqual(
            middle_det.update_finger(0, 12, 0.1, _hand_with_tip_pose(12, 0.559, 0.61), 1.0),
            ("h0_middle", "snare"),
        )

    def test_low_confidence_resets_tracking(self) -> None:
        det = InstrumentStrikeDetector(min_conf=0.5)
        self.assertIsNone(det.update_finger(0, 8, 0.0, _hand_with_index_pose(0.40, 0.60), 0.1))
        self.assertEqual(det._prev_y, {})

    def test_default_pad_zones_returns_eight_pad_grid(self) -> None:
        pads = default_pad_zones()
        self.assertEqual([p.label for p in pads], ["kick", "snare", "hat", "ride", "tom_l", "tom_m", "crash", "clap"])
        self.assertEqual(len(pads), 8)
        self.assertTrue(all(0.0 <= p.x1 < p.x2 <= 1.0 for p in pads))
        self.assertTrue(all(0.0 <= p.y1 < p.y2 <= 1.0 for p in pads))

    def test_load_pad_zones_json_validates_keys_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pads.json"
            path.write_text(
                json.dumps(
                    {
                        "pads": [
                            {
                                "label": "Kick",
                                "sound": "kick",
                                "x1": 0.1,
                                "y1": 0.2,
                                "x2": 0.3,
                                "y2": 0.4,
                                "color": [1, 2, 3],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pads = load_pad_zones_json(str(path), frozenset({"kick"}))
            self.assertEqual(pads[0], PadZone("Kick", "kick", 0.1, 0.2, 0.3, 0.4, (1, 2, 3)))

            path.write_text(
                json.dumps({"pads": [{"label": "bad", "sound": "nope", "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_pad_zones_json(str(path), frozenset({"kick"}))

            path.write_text(
                json.dumps({"pads": [{"label": "bad", "sound": "kick", "x1": 0.8, "y1": 0.2, "x2": 0.3, "y2": 0.4}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_pad_zones_json(str(path), frozenset({"kick"}))

    def test_pad_strike_detector_returns_pad_under_striking_tip(self) -> None:
        pad = PadZone("snare pad", "snare", 0.70, 0.55, 0.90, 0.90, (10, 20, 30))
        det = PadStrikeDetector(
            [pad],
            vy_trigger=0.10,
            joint_dps_trigger=1.0,
            cooldown_s=0.0,
            min_conf=0.0,
        )

        self.assertIsNone(det.update_finger(0, 8, 0.0, _hand_with_index_pose(0.40, 0.60), 1.0))
        self.assertEqual(
            det.update_finger(0, 8, 0.1, _hand_with_index_pose(0.62, 0.75), 1.0),
            pad,
        )

    def test_pad_strike_detector_rejects_whole_hand_translation_ghost(self) -> None:
        pad = PadZone("clap", "clap", 0.50, 0.50, 0.90, 0.90, (10, 20, 30))
        det = PadStrikeDetector(
            [pad],
            vy_trigger=0.10,
            joint_dps_trigger=1.0,
            cooldown_s=0.0,
            min_conf=0.0,
        )

        self.assertIsNone(
            det.update_finger(
                1,
                8,
                0.0,
                _translated_hand_with_index_pose(wrist_y=0.50, tip_y=0.60, tip_x=0.60),
                1.0,
            )
        )
        self.assertIsNone(
            det.update_finger(
                1,
                8,
                0.1,
                _translated_hand_with_index_pose(wrist_y=0.55, tip_y=0.65, tip_x=0.72),
                1.0,
            )
        )

    def test_pad_strike_detector_ignores_hits_outside_pad_and_enforces_cooldown(self) -> None:
        pad = PadZone("snare pad", "snare", 0.70, 0.55, 0.90, 0.95, (10, 20, 30))
        det = PadStrikeDetector(
            [pad],
            vy_trigger=0.10,
            joint_dps_trigger=0.0,
            cooldown_s=0.20,
            min_conf=0.0,
        )

        self.assertIsNone(det.update_finger(0, 8, 0.0, _hand_with_index_pose(0.40, 0.60), 1.0))
        self.assertIsNone(det.update_finger(0, 8, 0.1, _hand_with_index_pose(0.62, 0.50), 1.0))
        self.assertEqual(det.update_finger(0, 8, 0.2, _hand_with_index_pose(0.70, 0.75), 1.0), pad)
        self.assertIsNone(det.update_finger(0, 8, 0.3, _hand_with_index_pose(0.78, 0.76), 1.0))
        self.assertEqual(det.update_finger(0, 8, 0.5, _hand_with_index_pose(0.86, 0.77), 1.0), pad)

    def test_pad_strike_detector_uses_middle_finger_sensitivity(self) -> None:
        pad = PadZone("middle pad", "snare", 0.50, 0.50, 0.75, 0.75, (10, 20, 30))
        det = PadStrikeDetector(
            [pad],
            vy_trigger=0.001,
            joint_dps_trigger=20.0,
            cooldown_s=0.0,
            min_conf=0.0,
        )

        self.assertIsNone(det.update_finger(0, 12, 0.0, _hand_with_tip_pose(12, 0.550, 0.60), 1.0))
        self.assertEqual(
            det.update_finger(0, 12, 0.1, _hand_with_tip_pose(12, 0.559, 0.61), 1.0),
            pad,
        )


if __name__ == "__main__":
    unittest.main()
