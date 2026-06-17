# Forced-audio demo shot plan

This demo path does not use hand landmark decoding. Record a clean video while following the predefined timing, then mux the generated piano or drum audio onto the video.

## Recording setup

- Record in landscape, preferably 1080p or 720p.
- Keep both hands and the pad area visible for the full take.
- Start recording, hold still for about 2 seconds, then begin on the first hit after the count-in.
- Use the guide WAV in one earbud or another device. The original video audio will be discarded in the final MP4.
- If the first visible strike is slightly early or late, rerender with `--audio-offset`, for example `--audio-offset 0.08`.

## Piano take

Sequence file: `demo_sequences/piano_airplane_melody.json`

- Length: about 18 seconds.
- Tempo: 120 BPM, one note every 0.5 seconds.
- Melody: public-domain "Mary Had a Little Lamb" style, close to the familiar Korean "airplane" melody shape.
- Start first visible strike at `2.0s`.

Main strike notes:

```text
E D C D | E E E - | D D D - | E G G -
E D C D | E E E E | D D E D | C - - -
```

## Drum take

Sequence file: `demo_sequences/drum_simple_groove.json`

- Length: about 22 seconds.
- Tempo: 100 BPM.
- Start first visible strike at `2.0s`.
- For the visual take, alternate a low pad and a snare pad on the main beat. Hats and crashes are forced in the generated audio, so exact pad identity is less important than timing.

Main visible rhythm:

```text
kick snare kick snare | repeat for 8 bars
```

## Generate guide audio

```bash
python3 tools/render_forced_audio_demo.py \
  --sequence demo_sequences/piano_airplane_melody.json \
  --guide-output docs/demo_guide_piano_airplane.wav

python3 tools/render_forced_audio_demo.py \
  --sequence demo_sequences/drum_simple_groove.json \
  --guide-output docs/demo_guide_drum_groove.wav
```

## Mux final videos

```bash
python3 tools/render_forced_audio_demo.py \
  --sequence demo_sequences/piano_airplane_melody.json \
  --input docs/raw_piano_take.mp4 \
  --output docs/demo_piano_airplane_forced_audio.mp4

python3 tools/render_forced_audio_demo.py \
  --sequence demo_sequences/drum_simple_groove.json \
  --input docs/raw_drum_take.mp4 \
  --output docs/demo_drum_groove_forced_audio.mp4
```

## Record with live visual cues

Use this when the performer needs to see the camera and the target key/pad on
the connected monitor. The recorded raw video includes the UI overlay, then the
same script muxes the clean forced audio without click sounds.

```bash
python3 tools/record_forced_audio_take.py \
  --sequence demo_sequences/piano_airplane_melody.json \
  --raw-output docs/raw_piano_take_cued.mp4 \
  --output docs/demo_piano_airplane_cued_forced_audio.mp4

python3 tools/record_forced_audio_take.py \
  --sequence demo_sequences/drum_simple_groove.json \
  --raw-output docs/raw_drum_take_cued.mp4 \
  --output docs/demo_drum_groove_cued_forced_audio.mp4
```

## Record actual decoder-based audio first

Use this first for the real demo take. The sequence is only an on-screen cue;
the final audio is generated from actual decoded strike events. If the detected
audio is weak or wrong, keep the raw video and rerun the forced-audio mux from
the previous section.

```bash
python3 tools/record_decoder_demo_take.py \
  --sequence demo_sequences/piano_airplane_melody.json \
  --raw-output docs/raw_piano_decoder_take.mp4 \
  --output docs/demo_piano_decoder_audio.mp4 \
  --backend cpu

python3 tools/record_decoder_demo_take.py \
  --sequence demo_sequences/drum_simple_groove.json \
  --raw-output docs/raw_drum_decoder_take.mp4 \
  --output docs/demo_drum_decoder_audio.mp4 \
  --backend cpu
```
