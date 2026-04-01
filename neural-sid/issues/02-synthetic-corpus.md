# Issue 2: Synthetic Corpus Generator

## Description
Test and extend the synthetic SID register corpus generator.

## Tasks
- [ ] Verify `synthetic_corpus.py` produces valid 25-byte frames
- [ ] Add more patterns: vibrato, portamento, drum patterns (noise channel)
- [ ] Add multi-voice patterns (bass + lead + arpeggio simultaneously)
- [ ] Add a "trivial" mode: constant frames and alternating pairs (for unit testing the model)
- [ ] Validate output can be loaded by `train.py`

## Acceptance Criteria
- Output file is valid: size is divisible by 25
- Patterns are musically sensible when played back through a SID emulator
- Trivial patterns are learnable by even a tiny model
