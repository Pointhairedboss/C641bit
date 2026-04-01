; =============================================================================
; sid_output.asm - SID Chip Interface
; =============================================================================
; Handles initialising the SID and writing predicted register values.
; =============================================================================

; sid_init: Clear all SID registers
sid_init:
    LDX #24
    LDA #$00
-   STA SID_BASE,X
    DEX
    BPL -
    
    ; Set volume to max
    LDA #$0F
    STA SID_BASE + $18      ; mode/volume register
    
    RTS


; sid_write_output: Copy SID_OUTPUT to the actual SID registers
;   Called from IRQ handler when zp_frame_ready is set.
;
;   We write all 25 registers. Some notes:
;   - ADSR registers should be written BEFORE the control register
;     to avoid retriggering the envelope
;   - We write frequency and pulse width first, then ADSR, then control
;   - Filter and volume last

sid_write_output:
    ; Voice 1: freq, pw, adsr, control
    LDA SID_OUTPUT + 0      ; freq lo
    STA SID_BASE + $00
    LDA SID_OUTPUT + 1      ; freq hi
    STA SID_BASE + $01
    LDA SID_OUTPUT + 2      ; pulse width lo
    STA SID_BASE + $02
    LDA SID_OUTPUT + 3      ; pulse width hi
    STA SID_BASE + $03
    LDA SID_OUTPUT + 5      ; attack/decay
    STA SID_BASE + $05
    LDA SID_OUTPUT + 6      ; sustain/release
    STA SID_BASE + $06
    LDA SID_OUTPUT + 4      ; control (gate, waveform) — LAST for this voice
    STA SID_BASE + $04
    
    ; Voice 2
    LDA SID_OUTPUT + 7      ; freq lo
    STA SID_BASE + $07
    LDA SID_OUTPUT + 8      ; freq hi
    STA SID_BASE + $08
    LDA SID_OUTPUT + 9      ; pulse width lo
    STA SID_BASE + $09
    LDA SID_OUTPUT + 10     ; pulse width hi
    STA SID_BASE + $0A
    LDA SID_OUTPUT + 12     ; attack/decay
    STA SID_BASE + $0C
    LDA SID_OUTPUT + 13     ; sustain/release
    STA SID_BASE + $0D
    LDA SID_OUTPUT + 11     ; control
    STA SID_BASE + $0B
    
    ; Voice 3
    LDA SID_OUTPUT + 14     ; freq lo
    STA SID_BASE + $0E
    LDA SID_OUTPUT + 15     ; freq hi
    STA SID_BASE + $0F
    LDA SID_OUTPUT + 16     ; pulse width lo
    STA SID_BASE + $10
    LDA SID_OUTPUT + 17     ; pulse width hi
    STA SID_BASE + $11
    LDA SID_OUTPUT + 19     ; attack/decay
    STA SID_BASE + $13
    LDA SID_OUTPUT + 20     ; sustain/release
    STA SID_BASE + $14
    LDA SID_OUTPUT + 18     ; control
    STA SID_BASE + $12
    
    ; Filter
    LDA SID_OUTPUT + 21     ; filter cutoff lo
    STA SID_BASE + $15
    LDA SID_OUTPUT + 22     ; filter cutoff hi
    STA SID_BASE + $16
    LDA SID_OUTPUT + 23     ; resonance + filter enable
    STA SID_BASE + $17
    
    ; Mode + Volume
    ; Preserve at least some minimum volume so we can hear something
    LDA SID_OUTPUT + 24
    AND #$F0                ; keep mode bits
    ORA #$0F                ; force max volume
    STA SID_BASE + $18
    
    RTS
