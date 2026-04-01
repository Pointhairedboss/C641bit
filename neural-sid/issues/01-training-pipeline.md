# Issue 1: Training Pipeline

## Description
Complete and test the Python training pipeline for the BinaryMLP.

## Tasks
- [ ] Verify `model.py` BinaryMLP with straight-through estimator trains correctly
- [ ] Test `train.py` end-to-end with synthetic corpus
- [ ] Validate that binarised weights preserve learned patterns
- [ ] Add learning rate scheduling (cosine or step)
- [ ] Add gradient clipping for stability with STE
- [ ] Experiment with temperature scaling on output layer
- [ ] Add TensorBoard or simple CSV logging

## Acceptance Criteria
- Training on synthetic arpeggio corpus converges (val loss decreasing)
- Binarised model predicts constant-input sequences correctly
- Model can learn simple patterns: constant frame, alternating frames, arpeggio cycle

## Notes
- The straight-through estimator can be unstable with large learning rates
- Start with lr=1e-3 and reduce if loss explodes
- Batch size 64 is reasonable for the corpus sizes we're working with
