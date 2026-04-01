; =============================================================================
; inference.asm - Binary Neural Network Forward Pass
; =============================================================================
; XOR + popcount implementation for 1-bit weight inference.
;
; The key insight: with binary inputs and binary weights,
; the dot product reduces to:
;   dot = total_bits - 2 × popcount(input XOR weights)
;
; We don't even need the full dot product. For the sign activation,
; we only need to know: is popcount(XOR) > half the input size?
;   If yes → activation = 0 (representing -1)
;   If no  → activation = 1 (representing +1)
;
; This means the entire forward pass is: XOR, popcount, compare.
; =============================================================================

; run_inference: Execute full forward pass
;   Input:  SID_STATE (25 bytes)
;   Output: SID_OUTPUT (25 bytes)
;
; Walks through all layers, reading weights from WEIGHTS_BASE.
; Layer structure in memory:
;   Layer 0: INPUT_BYTES × HIDDEN_SIZE rows  (25 bytes × 128 rows = 3200 bytes)
;   Layer 1: HIDDEN_BYTES × HIDDEN_SIZE rows (16 bytes × 128 rows = 2048 bytes)
;   Layer 2: HIDDEN_BYTES × OUTPUT_SIZE rows (16 bytes × 200 rows = 3200 bytes)
;   Total: 8448 bytes (~8.25 KB)

run_inference:
    ; --- Layer 1: Input (200 bits) → Hidden (128 neurons) ---
    ; Input is SID_STATE (25 bytes, already packed as 200 bits)
    ; Weights start at WEIGHTS_BASE
    ; Output goes to ACTIVATION_BUF (16 bytes = 128 bits)
    
    LDA #<SID_STATE
    STA zp_input_ptr
    LDA #>SID_STATE
    STA zp_input_ptr + 1
    
    LDA #<WEIGHTS_BASE
    STA zp_weight_ptr
    LDA #>WEIGHTS_BASE
    STA zp_weight_ptr + 1
    
    LDA #HIDDEN_SIZE
    STA zp_neuron_count
    
    LDA #INPUT_BYTES         ; 25 bytes per weight row
    STA zp_byte_count
    
    LDA #(INPUT_SIZE / 2)    ; threshold = 100 (half of 200 bits)
    STA zp_threshold
    
    ; Output: build activation byte at ACTIVATION_BUF
    LDX #0                   ; output byte index
    LDY #0                   ; bit index within byte (0-7)
    LDA #0
    STA ACTIVATION_BUF       ; clear first output byte
    
    JSR process_layer
    
    ; --- Layer 2: Hidden (128 bits) → Hidden (128 neurons) ---
    LDA #<ACTIVATION_BUF
    STA zp_input_ptr
    LDA #>ACTIVATION_BUF
    STA zp_input_ptr + 1
    
    ; Weight pointer continues from where layer 1 ended
    ; (zp_weight_ptr was advanced by process_layer)
    
    LDA #HIDDEN_SIZE
    STA zp_neuron_count
    
    LDA #HIDDEN_BYTES        ; 16 bytes per weight row
    STA zp_byte_count
    
    LDA #(HIDDEN_SIZE / 2)   ; threshold = 64
    STA zp_threshold
    
    ; Output to NEXT_ACT_BUF, then swap
    LDX #0
    LDY #0
    LDA #0
    STA NEXT_ACT_BUF
    
    JSR process_layer_to_next
    
    ; Swap buffers: copy NEXT_ACT_BUF → ACTIVATION_BUF
    LDX #(HIDDEN_BYTES - 1)
-   LDA NEXT_ACT_BUF,X
    STA ACTIVATION_BUF,X
    DEX
    BPL -
    
    ; --- Layer 3: Hidden (128 bits) → Output (200 neurons) ---
    LDA #<ACTIVATION_BUF
    STA zp_input_ptr
    LDA #>ACTIVATION_BUF
    STA zp_input_ptr + 1
    
    LDA #OUTPUT_SIZE
    STA zp_neuron_count
    
    LDA #HIDDEN_BYTES        ; 16 bytes per weight row
    STA zp_byte_count
    
    LDA #(HIDDEN_SIZE / 2)   ; threshold = 64
    STA zp_threshold
    
    LDX #0
    LDY #0
    LDA #0
    STA OUTPUT_BITS
    
    JSR process_layer_to_output
    
    ; Pack OUTPUT_BITS (25 bytes) → SID_OUTPUT (25 bytes)
    ; (they're already in packed bit format, just copy)
    LDX #(OUTPUT_BYTES - 1)
-   LDA OUTPUT_BITS,X
    STA SID_OUTPUT,X
    DEX
    BPL -
    
    RTS


; process_layer: Compute one layer's worth of neurons
;   Uses: zp_input_ptr, zp_weight_ptr, zp_neuron_count, zp_byte_count, zp_threshold
;   Output: bits packed into ACTIVATION_BUF
;
; For each neuron:
;   1. XOR input bytes with weight row bytes
;   2. Sum popcount of each XOR result
;   3. If total popcount > threshold: output bit = 0, else output bit = 1
;   4. Advance weight pointer to next row

process_layer:
    LDX #0                   ; output byte index in ACTIVATION_BUF
    STX zp_temp              ; bit position within byte (0-7, counts down in shifts)
    LDA #0
    STA ACTIVATION_BUF

.neuron_loop:
    ; Compute XOR + popcount for this neuron
    JSR xor_popcount         ; result in A = total popcount
    
    ; Compare with threshold
    CMP zp_threshold
    BCS .output_zero         ; popcount >= threshold → mismatches dominate → output 0
    
    ; Output 1: set bit in current byte
    LDA ACTIVATION_BUF,X
    ORA .bit_masks,Y
    STA ACTIVATION_BUF,X
    JMP .next_neuron
    
.output_zero:
    ; Bit is already 0 (cleared), nothing to do
    
.next_neuron:
    INY                      ; next bit position
    CPY #8
    BNE +
    LDY #0                   ; reset bit position
    INX                      ; next output byte
    LDA #0
    STA ACTIVATION_BUF,X    ; clear next byte
+
    DEC zp_neuron_count
    BNE .neuron_loop
    
    RTS


; process_layer_to_next: Same as process_layer but outputs to NEXT_ACT_BUF
process_layer_to_next:
    LDX #0
    LDY #0
    LDA #0
    STA NEXT_ACT_BUF

.neuron_loop2:
    JSR xor_popcount
    
    CMP zp_threshold
    BCS .output_zero2
    
    LDA NEXT_ACT_BUF,X
    ORA .bit_masks,Y
    STA NEXT_ACT_BUF,X
    JMP .next_neuron2
    
.output_zero2:

.next_neuron2:
    INY
    CPY #8
    BNE +
    LDY #0
    INX
    LDA #0
    STA NEXT_ACT_BUF,X
+
    DEC zp_neuron_count
    BNE .neuron_loop2
    
    RTS


; process_layer_to_output: Same but outputs to OUTPUT_BITS
process_layer_to_output:
    LDX #0
    LDY #0
    LDA #0
    STA OUTPUT_BITS

.neuron_loop3:
    JSR xor_popcount
    
    CMP zp_threshold
    BCS .output_zero3
    
    LDA OUTPUT_BITS,X
    ORA .bit_masks,Y
    STA OUTPUT_BITS,X
    JMP .next_neuron3
    
.output_zero3:

.next_neuron3:
    INY
    CPY #8
    BNE +
    LDY #0
    INX
    LDA #0
    STA OUTPUT_BITS,X
+
    DEC zp_neuron_count
    BNE .neuron_loop3
    
    RTS


; xor_popcount: XOR input with weight row and sum popcounts
;   Input:  zp_input_ptr → input vector
;           zp_weight_ptr → current weight row
;           zp_byte_count = bytes per row
;   Output: A = total popcount
;   Side effect: advances zp_weight_ptr past this row
;
; This is the HOT LOOP. Every cycle matters here.

xor_popcount:
    LDA #0
    STA zp_accum             ; reset accumulator
    
    ; Save Y (bit position)
    TYA
    PHA
    
    LDY #0                   ; byte index
    
.xor_loop:
    LDA (zp_input_ptr),Y    ; load input byte
    EOR (zp_weight_ptr),Y   ; XOR with weight byte
    TAX                      ; use as index
    LDA popcount_table,X    ; lookup popcount (0-8)
    CLC
    ADC zp_accum
    STA zp_accum
    
    INY
    CPY zp_byte_count
    BNE .xor_loop
    
    ; Advance weight pointer past this row
    CLC
    LDA zp_weight_ptr
    ADC zp_byte_count
    STA zp_weight_ptr
    LDA zp_weight_ptr + 1
    ADC #0
    STA zp_weight_ptr + 1
    
    ; Restore Y, return popcount in A
    PLA
    TAY
    LDA zp_accum
    RTS


; Bit mask table for setting individual bits (MSB first)
.bit_masks:
    !byte $80, $40, $20, $10, $08, $04, $02, $01
