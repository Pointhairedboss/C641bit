# Issue 4: .d64 Disk Image Packer

## Tasks
- [ ] Test `d64pack.py` produces valid .d64 images
- [ ] Verify images load in VICE emulator (`x64 -autostart`)
- [ ] Test with files of various sizes (small weights, full 170KB disk)
- [ ] Handle edge case: file too large for disk (error gracefully)
- [ ] Verify directory listing shows correct file names and sizes in VICE

## Acceptance Criteria
- .d64 opens in VICE and shows correct directory
- PRG files auto-load and execute
- SEQ files are readable from BASIC (for manual testing)
