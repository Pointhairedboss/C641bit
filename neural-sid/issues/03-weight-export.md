# Issue 3: Weight Export & Verification

## Tasks
- [ ] Test `export.py` round-trip: export then read back and compare
- [ ] Verify packed binary matches expected format from SPEC.md
- [ ] Add a hex dump mode for debugging
- [ ] Write a C64-side weight loader test (read header, print dimensions)
- [ ] Test with different model sizes (128, 256 hidden)

## Acceptance Criteria
- `--verify` flag passes without assertion errors
- Exported file size matches `model.packed_size_bytes()` + header
- File can be read by the d64 packer without errors
