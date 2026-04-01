; =============================================================================
; NEURAL SID - 1-Bit Neural Network Music Generator for Commodore 64
; =============================================================================
; Runs a binary MLP (XOR + popcount) to predict SID register frames.
; All ROMs banked out, no screen, output is 100% SID chip.
;
; Assembler: ACME (https://sourceforge.net/projects/acme-crossass/)
; Build:     acme -f cbm -o neural-sid.prg main.asm
; =============================================================================

; --- Memory Map ---
; $0002-$00FF  Zero page: inference variables
; $0100-$01FF  Stack
; $0200-$07FF  Code
; $0800-$9FFF  Weight storage (38 KB)
; $A000-$BFFF  Weight overflow / BASIC ROM area (8 KB)
; $C000-$CFFF  Activation buffers (4 KB)
; $D000-$D3FF  VIC-II (we don't touch it)
; $D400-$D418  SID registers (our output!)
; $D800-$DBFF  Colour RAM (ignored)
; $DC00-$DCFF  CIA 1 (timer for IRQ)
; $DD00-$DDFF  CIA 2
; $E000-$FFFF  Weight overflow / KERNAL ROM area (8 KB)

; --- Constants ---
SID_BASE        = $D400
CIA1_ICR        = $DC0D
CIA1_TIMER_A_LO = $DC04
CIA1_TIMER_A_HI = $DC05
CIA1_CTRL_A     = $DC0E
PROCESSOR_PORT  = $01
IRQ_VECTOR      = $FFFE    ; hardware IRQ vector (when KERNAL banked out)

; Model dimensions (must match exported weights)
INPUT_SIZE      = 200       ; bits (25 SID registers × 8)
HIDDEN_SIZE     = 128       ; neurons per hidden layer
OUTPUT_SIZE     = 200       ; bits
INPUT_BYTES     = 25        ; INPUT_SIZE / 8
HIDDEN_BYTES    = 16        ; HIDDEN_SIZE / 8
OUTPUT_BYTES    = 25        ; OUTPUT_SIZE / 8

; Memory addresses for buffers
WEIGHTS_BASE    = $0800     ; weight data loaded here
ACTIVATION_BUF  = $C000     ; current layer activations
NEXT_ACT_BUF   = $C100     ; next layer activations
SID_STATE       = $C200     ; current SID register snapshot (25 bytes)
SID_OUTPUT      = $C219     ; predicted SID registers (25 bytes)
INPUT_BITS      = $C300     ; bit-decomposed input (25 bytes packed)
OUTPUT_BITS     = $C319     ; bit-decomposed output (25 bytes packed)

; Zero page variables
zp_weight_ptr   = $02       ; 2 bytes - pointer to current weight row
zp_input_ptr    = $04       ; 2 bytes - pointer to current input vector
zp_accum        = $06       ; 1 byte  - dot product accumulator
zp_neuron_count = $07       ; 1 byte  - neuron counter
zp_layer_count  = $08       ; 1 byte  - layer counter
zp_byte_count   = $09       ; 1 byte  - byte loop counter
zp_temp         = $0A       ; 1 byte  - temp
zp_frame_ready  = $0B       ; 1 byte  - flag: new frame ready for SID
zp_threshold    = $0C       ; 1 byte  - activation threshold

; --- Program start ---
*= $0200

    ; Entry point
    SEI                     ; disable interrupts during setup
    
    ; Bank out BASIC and KERNAL ROMs
    ; Bit pattern: xxxx x 1 0 0
    ;   Bit 0: LORAM  (0 = BASIC ROM out)
    ;   Bit 1: HIRAM  (0 = KERNAL ROM out)  
    ;   Bit 2: CHAREN (1 = I/O visible at $D000)
    LDA #$35                ; %00110101 - I/O visible, ROMs banked out
    STA PROCESSOR_PORT
    
    ; Clear SID
    JSR sid_init
    
    ; TODO: Load weights from disk via fastloader
    ; For now, assume weights are pre-loaded at WEIGHTS_BASE
    ; (In development, the .prg can include weights in-line)
    
    ; Set up IRQ for 50Hz (PAL) frame rate
    ; PAL: 985248 cycles/sec ÷ 50 = 19705 cycles per frame
    LDA #<19705
    STA CIA1_TIMER_A_LO
    LDA #>19705
    STA CIA1_TIMER_A_HI
    
    ; Point IRQ vector to our handler
    ; (With KERNAL banked out, we use the hardware vector at $FFFE)
    LDA #<irq_handler
    STA IRQ_VECTOR
    LDA #>irq_handler
    STA IRQ_VECTOR + 1
    
    ; Enable CIA1 Timer A interrupt
    LDA #$81               ; bit 7 = set, bit 0 = Timer A
    STA CIA1_ICR
    
    ; Start Timer A: continuous, counting system clock
    LDA #$11               ; bit 0 = start, bit 4 = force load
    STA CIA1_CTRL_A
    
    ; Seed initial SID state (silence)
    LDX #24
-   LDA #$00
    STA SID_STATE,X
    DEX
    BPL -
    
    ; Set initial gate on voice 1 with pulse wave for something to start from
    LDA #$41               ; pulse + gate
    STA SID_STATE + 4
    LDA #$10               ; freq hi = some audible note
    STA SID_STATE + 1
    LDA #$08               ; pulse width mid
    STA SID_STATE + 3
    LDA #$09               ; attack=0, decay=9
    STA SID_STATE + 5
    
    ; Enable interrupts and enter main loop
    LDA #$00
    STA zp_frame_ready
    CLI
    
main_loop:
    ; Run inference (this takes multiple frames)
    JSR run_inference
    
    ; Signal that a new frame is ready
    LDA #$01
    STA zp_frame_ready
    
    ; Wait until the IRQ handler has written it to the SID
-   LDA zp_frame_ready
    BNE -
    
    ; Copy output back to input for next prediction
    LDX #24
-   LDA SID_OUTPUT,X
    STA SID_STATE,X
    DEX
    BPL -
    
    JMP main_loop

; --- IRQ Handler ---
irq_handler:
    PHA
    TXA
    PHA
    TYA
    PHA
    
    ; Acknowledge CIA1 interrupt
    LDA CIA1_ICR
    
    ; If a new frame is ready, write it to SID
    LDA zp_frame_ready
    BEQ +
    
    JSR sid_write_output
    
    LDA #$00
    STA zp_frame_ready

+   PLA
    TAY
    PLA
    TAX
    PLA
    RTI

; --- Include modules ---
!source "inference.asm"
!source "sid_output.asm"
; !source "fastload.asm"     ; uncomment when fastloader is ready

; --- Popcount lookup table ---
; 256 bytes: popcount_table[i] = number of 1-bits in byte i
!align 255, 0               ; align to page boundary for fast indexed access
popcount_table:
!for i, 0, 255 {
    ; ACME doesn't have a built-in popcount, so we compute it
    ; popcount(i) = sum of bits
    !byte ((i>>7)&1)+((i>>6)&1)+((i>>5)&1)+((i>>4)&1)+((i>>3)&1)+((i>>2)&1)+((i>>1)&1)+(i&1)
}
