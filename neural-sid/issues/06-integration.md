# Issue 6: End-to-End Integration

## Description
Wire everything together: train → export → pack → run.

## Tasks
- [ ] Create a single `build.sh` that runs the full pipeline
- [ ] Generate synthetic corpus
- [ ] Train model (small, fast: 50 epochs, hidden=128)
- [ ] Export weights
- [ ] Bake weights into .prg (inline for initial build)
- [ ] Pack into .d64
- [ ] Document manual VICE testing steps
- [ ] Create a Python SID register playback tool (play corpus through libsidplayfp or simple audio)

## Acceptance Criteria
- `./build.sh` runs from clean state and produces a .d64
- .d64 runs in VICE and produces audio from the SID
- Audio is recognisably "musical" (not random noise)

## Stretch Goals
- Record SID output from VICE as .wav for comparison
- Side-by-side: original corpus vs generated output
