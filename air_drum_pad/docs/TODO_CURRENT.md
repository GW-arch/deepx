# Current TODO — AI Air-Drum Pad

Date: 2026-05-26

This file replaces the completed historical TODO list from `TODO_2026_05_18.md`.
The previous interface/performance items are implemented and summarized in
`SESSION_2026_05_26_TODO_IMPLEMENTATION.md`.

## Status summary

- Runtime/code TODOs from the prior list are complete.
- Current open work is mainly evaluation, final-report placeholders, and future enhancements.
- After any report placeholder is filled, regenerate both English/Korean DOCX and PDF exports with:

```bash
cd air_drum_pad
python3 tools/export_report_documents.py
```

## P0 — Project demo submission

- [ ] Confirm demo submission requirements: format, duration, deadline, upload location, and whether source code/report links are required.
- [ ] Record a working demo video showing both modes:
  - drum pad mode with visible pad hits and audible sounds,
  - piano mode with the final left/right note mapping and audible notes.
- [ ] Include a short narration or captions explaining the camera setup, hand tracking, strike detection, and mode switching.
- [ ] Export the demo video in the required format (for example `.mp4`) and store the final filename/link in this TODO.
- [ ] Do a final playback check on another machine before submission.

## P0 — Finish report deliverables

- [ ] Fill report metadata in both language versions:
  - `docs/FINAL_REPORT_AI_AIR_DRUM_PAD.md`
  - `docs/FINAL_REPORT_AI_AIR_DRUM_PAD_KO.md`
  - replace `FILLME` for authors, affiliation, and acknowledgments.
- [ ] Capture final live **drum mode** screenshot and replace:
  - `FILLME_live_drum_mode_screenshot.png`
- [ ] Capture final live **piano mode** screenshot and replace:
  - `FILLME_live_piano_mode_screenshot.png`
- [ ] Capture or generate the E2E latency measurement setup figure and replace:
  - `FILLME_latency_measurement_setup.png`
- [ ] Add the final hit-accuracy result figure/table and replace:
  - `FILLME_hit_accuracy_results.png`
- [ ] Regenerate report exports after the above changes:
  - `docs/FINAL_REPORT_AI_AIR_DRUM_PAD.docx`
  - `docs/FINAL_REPORT_AI_AIR_DRUM_PAD.pdf`
  - `docs/FINAL_REPORT_AI_AIR_DRUM_PAD_KO.docx`
  - `docs/FINAL_REPORT_AI_AIR_DRUM_PAD_KO.pdf`

## P1 — Evaluation tasks

- [ ] Measure end-to-end audio latency with a high-speed camera or synchronized audio/video capture.
- [ ] Run tempo-controlled hit-accuracy trials for both drum and piano modes.
- [ ] Report true positives, false positives, false negatives, mean latency, standard deviation, and P95 latency.
- [ ] Capture at least one hold-out dataset under different lighting/camera-distance conditions.
- [ ] Evaluate NPU landmark correction on training vs. hold-out captures before using it as a default.
- [ ] Validate `--palm-redetect-every` and `--async-palm` settings against drift and hit accuracy, not only latency.

## P2 — Product and interaction improvements

- [ ] Add per-user calibration for pad positions, handedness, and strike thresholds.
- [ ] Add optional MIDI output so the system can drive external instruments or DAWs.
- [ ] Add a visual/audio metronome mode for controlled evaluation and practice.
- [ ] Add a small user study comparing playability of drum mode vs. piano mode.
- [ ] Consider UI controls for loading/saving pad layouts without editing JSON manually.

## P3 — NPU / model pipeline improvements

- [ ] Investigate palm detector alternatives that quantize well for NPU deployment.
- [ ] Revisit palm `.dxnn` conversion only if score-head quality can be preserved.
- [ ] Profile NPU dispatch overhead for the hand landmark model (`~8 ms` NPU vs `~5 ms` CPU TFLite was previously observed).
- [ ] Document exact DX-RT/DX-COM versions, model checksums, and vendor asset acquisition steps for reproducibility.

## P4 — Documentation hygiene

- [ ] Update `docs/NEXT_SESSION_NPU_PALM.md` where it still contains historical `커밋: TBD` text.
- [ ] Keep `docs/FINAL_REPORT_AI_AIR_DRUM_PAD.md` and `docs/FINAL_REPORT_AI_AIR_DRUM_PAD_KO.md` synchronized when report content changes.
- [ ] If project knowledge will continue to accumulate, create a persistent `omx_wiki/` index for mapping conventions, evaluation results, and session decisions.
