; =============================================================================
; fastload.asm - Minimal 1541 Fast Loader
; =============================================================================
; TODO: Implement a basic fast loader for paging weights from disk.
;
; For the initial build, weights are compiled directly into the .prg.
; This module will be needed when the model exceeds ~40KB (too large for RAM).
;
; Approach options:
;   1. Custom 2-bit serial protocol (classic demo scene approach)
;   2. Burst mode (1571/1581 only)
;   3. Simple optimised KERNAL replacement (slowest but simplest)
;
; Since KERNAL is banked out, we need to either:
;   a. Temporarily bank KERNAL in for LOAD, then bank out again
;   b. Write our own serial bus routines (preferred for speed)
;
; For now, this file is a placeholder.
; =============================================================================

; fast_load: Load a file from disk into memory
;   Input:  A/X = filename pointer (lo/hi)
;           Y = filename length
;           zp_load_addr = destination address (2 bytes)
;   Output: carry clear = success, carry set = error
;
; Temporary implementation: bank KERNAL in, use standard LOAD, bank out

fast_load:
    ; Bank KERNAL ROM in temporarily
    LDA #$37                ; BASIC + KERNAL + I/O
    STA PROCESSOR_PORT
    
    ; TODO: Set up KERNAL LOAD call
    ; LDA #$01              ; device 8
    ; LDX zp_load_addr
    ; LDY zp_load_addr + 1
    ; JSR $FFD5             ; KERNAL LOAD
    
    ; Bank ROMs back out
    LDA #$35                ; I/O only, no ROMs
    STA PROCESSOR_PORT
    
    RTS
