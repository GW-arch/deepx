from __future__ import annotations

import unittest

from drumkit_audio import (
    PIANO_DEFAULT_SLOTS,
    _hat_closed,
    _kick,
    _piano_tone,
    note_name_to_midi,
    piano_slots_from_inter_hand_distance,
)


class DrumkitAudioTests(unittest.TestCase):
    def test_piano_default_left_hand_descends_thumb_to_pinky(self) -> None:
        self.assertEqual(PIANO_DEFAULT_SLOTS[:5], ("G4", "F4", "E4", "D4", "C4"))
        self.assertEqual(PIANO_DEFAULT_SLOTS[5:], ("A4", "B4", "C5", "D5", "E5"))

    def test_dynamic_piano_left_hand_descends_thumb_to_pinky(self) -> None:
        slots = piano_slots_from_inter_hand_distance(0.30)
        self.assertEqual(len(slots), 10)
        left_midis = [note_name_to_midi(s) for s in slots[:5]]
        self.assertEqual(left_midis, sorted(left_midis, reverse=True))

    def test_piano_tone_defaults_to_half_second(self) -> None:
        self.assertEqual(len(_piano_tone(1000, 440.0)), 500)

    def test_short_drum_hits_have_longer_audible_samples(self) -> None:
        self.assertGreaterEqual(len(_kick(1000)), 280)
        self.assertGreaterEqual(len(_hat_closed(1000)), 140)


if __name__ == "__main__":
    unittest.main()
