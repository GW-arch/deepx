from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.guided_eval import (
    Cue,
    Event,
    build_block_cues,
    build_cues,
    default_targets,
    load_cues_json,
    match_events,
    piano_finger_id_for_mapping,
    physical_side_from_screen_x,
    parse_targets,
    scored_cues_for_protocol,
    write_outputs,
)


class GuidedEvalTests(unittest.TestCase):
    def test_build_cues_repeats_targets_at_bpm(self) -> None:
        cues = build_cues(["kick", "snare"], bpm=120.0, repeats=2, lead_in_s=1.0)
        self.assertEqual([c.target for c in cues], ["kick", "snare", "kick", "snare"])
        self.assertEqual([c.cue_id for c in cues], [1, 2, 3, 4])
        self.assertEqual([c.t_s for c in cues], [1.0, 1.5, 2.0, 2.5])

    def test_build_block_cues_adds_ready_time_and_repeated_beats(self) -> None:
        cues = build_block_cues(
            ["kick", "snare"],
            bpm=60.0,
            repeats=1,
            lead_in_s=2.0,
            beats_per_target=3,
            block_ready_s=1.5,
        )
        self.assertEqual([c.target for c in cues], ["kick", "kick", "kick", "snare", "snare", "snare"])
        self.assertEqual([c.cue_id for c in cues], [1, 2, 3, 4, 5, 6])
        self.assertEqual([c.t_s for c in cues], [3.5, 4.5, 5.5, 8.0, 9.0, 10.0])

    def test_scored_cues_skip_block_warmup_beats(self) -> None:
        cues = build_block_cues(
            ["kick", "snare"],
            bpm=60.0,
            repeats=1,
            lead_in_s=0.0,
            beats_per_target=4,
            block_ready_s=0.0,
        )
        scored = scored_cues_for_protocol(
            cues,
            protocol="blocks",
            beats_per_target=4,
            warmup_beats=2,
        )
        self.assertEqual([c.cue_id for c in scored], [3, 4, 7, 8])
        self.assertEqual(
            scored_cues_for_protocol(cues[:2], protocol="sequence", beats_per_target=4, warmup_beats=2),
            cues[:2],
        )

    def test_physical_side_mapping_matches_mirrored_camera_setup(self) -> None:
        self.assertEqual(physical_side_from_screen_x(0.25, mirror=True), 0)
        self.assertEqual(physical_side_from_screen_x(0.75, mirror=True), 1)
        self.assertEqual(physical_side_from_screen_x(0.25, mirror=False), 1)
        self.assertEqual(physical_side_from_screen_x(0.75, mirror=False), 0)

    def test_mirrored_piano_finger_order_swaps_thumb_and_pinky(self) -> None:
        self.assertEqual(piano_finger_id_for_mapping(4, mirror_finger_order=True), 20)
        self.assertEqual(piano_finger_id_for_mapping(8, mirror_finger_order=True), 16)
        self.assertEqual(piano_finger_id_for_mapping(12, mirror_finger_order=True), 12)
        self.assertEqual(piano_finger_id_for_mapping(16, mirror_finger_order=True), 8)
        self.assertEqual(piano_finger_id_for_mapping(20, mirror_finger_order=True), 4)
        self.assertEqual(piano_finger_id_for_mapping(4, mirror_finger_order=False), 4)

    def test_parse_targets_normalizes_by_mode(self) -> None:
        self.assertEqual(parse_targets("Kick, SNARE", "drum"), ("kick", "snare"))
        self.assertEqual(parse_targets("c4, g5", "piano"), ("C4", "G5"))

    def test_default_targets_are_mode_specific(self) -> None:
        self.assertIn("kick", default_targets("drum"))
        self.assertEqual(default_targets("piano")[:5], ("C4", "D4", "E4", "F4", "G4"))
        with self.assertRaises(ValueError):
            default_targets("bad")

    def test_match_events_reports_accuracy_and_latency(self) -> None:
        cues = [Cue(1, 1.0, "kick"), Cue(2, 2.0, "snare")]
        events = [
            Event(1, 1.10, "kick"),
            Event(2, 1.40, "hat"),
            Event(3, 2.05, "snare"),
        ]
        result = match_events(cues, events, pre_window_s=0.20, post_window_s=0.70)

        self.assertEqual([m.outcome for m in result.matches], ["hit", "hit"])
        self.assertEqual(result.summary["tp"], 2)
        self.assertEqual(result.summary["fp"], 1)
        self.assertEqual(result.summary["fn"], 0)
        self.assertAlmostEqual(result.summary["precision"], 2 / 3)
        self.assertAlmostEqual(result.summary["recall"], 1.0)
        self.assertAlmostEqual(result.summary["mean_latency_ms"], 75.0)
        self.assertAlmostEqual(result.summary["mean_abs_latency_ms"], 75.0)

    def test_match_events_marks_misses_and_early_hits(self) -> None:
        cues = [Cue(1, 1.0, "kick"), Cue(2, 2.0, "kick")]
        events = [Event(1, 0.90, "kick")]
        result = match_events(cues, events, pre_window_s=0.20, post_window_s=0.30)

        self.assertEqual([m.outcome for m in result.matches], ["hit", "miss"])
        self.assertAlmostEqual(result.matches[0].latency_s or 0.0, -0.10)
        self.assertEqual(result.summary["tp"], 1)
        self.assertEqual(result.summary["fp"], 0)
        self.assertEqual(result.summary["fn"], 1)
        self.assertAlmostEqual(result.summary["recall"], 0.5)

    def test_load_sequence_json_and_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            seq_path = Path(td) / "sequence.json"
            seq_path.write_text(
                '{"sequence": [{"t_s": 1.25, "target": "c4"}, {"t_s": 2.5, "target": "g5"}]}',
                encoding="utf-8",
            )
            cues = load_cues_json(str(seq_path), "piano")
            self.assertEqual(cues, [Cue(1, 1.25, "C4"), Cue(2, 2.5, "G5")])

            result = match_events(cues, [Event(1, 1.30, "C4")])
            out_dir = Path(td) / "out"
            write_outputs(
                out_dir,
                cues=cues,
                events=[Event(1, 1.30, "C4", "left pinky", 12)],
                matches=result.matches,
                summary=result.summary,
                metadata={"mode": "piano", "backend": "cpu"},
            )
            self.assertTrue((out_dir / "cues.csv").is_file())
            self.assertTrue((out_dir / "events.csv").is_file())
            self.assertTrue((out_dir / "matches.csv").is_file())
            self.assertIn("cue-to-detection latency", (out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
