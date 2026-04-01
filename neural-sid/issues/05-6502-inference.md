# Issue 5: 6502 Inference Engine

## Description
Get the assembly code compiling and running in VICE.

## Tasks
- [ ] Install ACME and verify `main.asm` assembles cleanly
- [ ] Fix any syntax issues in the popcount table generation
- [ ] Write a Python script to generate a test .prg with known weights baked in
- [ ] Verify XOR+popcount produces correct results for known inputs
- [ ] Measure actual cycle count per inference pass in VICE monitor
- [ ] Optimise hot loop if needed (unroll, zero-page tricks)
- [ ] Test IRQ handler fires at 50Hz and writes to SID

## Acceptance Criteria
- `make` produces a valid .prg
- .prg runs in VICE without crashing
- SID produces audible output
- With known constant weights, output is deterministic and correct

## Notes
- ACME syntax: `!for` loop may need adjustment for popcount table
- The `!align` directive may behave differently across ACME versions
- Test in VICE with monitor: set breakpoints in xor_popcount to verify
- Consider: if ACME popcount table generation is too complex, generate it offline in Python and `!binary` include it
