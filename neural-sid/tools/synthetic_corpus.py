"""
Generate synthetic SID register sequences for testing.

Produces predictable patterns that a small MLP should be able to learn:
- Arpeggios (cycling frequency values)
- Pulse width sweeps
- Filter sweeps
- Combined patterns

Output: raw binary file of 25-byte frames at 50Hz.

SID register layout per frame (25 bytes):
  Voice 1: freq_lo, freq_hi, pw_lo, pw_hi, control, attack_decay, sustain_release
  Voice 2: freq_lo, freq_hi, pw_lo, pw_hi, control, attack_decay, sustain_release
  Voice 3: freq_lo, freq_hi, pw_lo, pw_hi, control, attack_decay, sustain_release
  Filter:  fc_lo, fc_hi, res_filt, mode_vol
"""

import argparse
from pathlib import Path

import numpy as np


# SID frequency table (C-2 to B-7, PAL values)
# Subset: one octave for arpeggios
NOTE_FREQS = {
    "C4": 4291, "D4": 4817, "E4": 5407, "F4": 5728,
    "G4": 6430, "A4": 7217, "B4": 8101,
    "C5": 8583, "D5": 9634, "E5": 10814, "F5": 11457,
    "G5": 12860, "A5": 14434, "B5": 16203,
}


def freq_to_bytes(freq: int) -> tuple[int, int]:
    """Convert 16-bit frequency to (lo, hi) bytes."""
    return freq & 0xFF, (freq >> 8) & 0xFF


def pw_to_bytes(pw: int) -> tuple[int, int]:
    """Convert 12-bit pulse width to (lo, hi) bytes."""
    return pw & 0xFF, (pw >> 8) & 0x0F


def make_frame(
    v1_freq: int = 0, v1_pw: int = 0x800, v1_ctrl: int = 0, v1_ad: int = 0, v1_sr: int = 0,
    v2_freq: int = 0, v2_pw: int = 0x800, v2_ctrl: int = 0, v2_ad: int = 0, v2_sr: int = 0,
    v3_freq: int = 0, v3_pw: int = 0x800, v3_ctrl: int = 0, v3_ad: int = 0, v3_sr: int = 0,
    fc: int = 0, res_filt: int = 0, mode_vol: int = 0x0F,
) -> bytes:
    """Construct a 25-byte SID register frame."""
    frame = bytearray(25)
    
    # Voice 1
    f1_lo, f1_hi = freq_to_bytes(v1_freq)
    p1_lo, p1_hi = pw_to_bytes(v1_pw)
    frame[0], frame[1] = f1_lo, f1_hi
    frame[2], frame[3] = p1_lo, p1_hi
    frame[4] = v1_ctrl
    frame[5] = v1_ad
    frame[6] = v1_sr
    
    # Voice 2
    f2_lo, f2_hi = freq_to_bytes(v2_freq)
    p2_lo, p2_hi = pw_to_bytes(v2_pw)
    frame[7], frame[8] = f2_lo, f2_hi
    frame[9], frame[10] = p2_lo, p2_hi
    frame[11] = v2_ctrl
    frame[12] = v2_ad
    frame[13] = v2_sr
    
    # Voice 3
    f3_lo, f3_hi = freq_to_bytes(v3_freq)
    p3_lo, p3_hi = pw_to_bytes(v3_pw)
    frame[14], frame[15] = f3_lo, f3_hi
    frame[16], frame[17] = p3_lo, p3_hi
    frame[18] = v3_ctrl
    frame[19] = v3_ad
    frame[20] = v3_sr
    
    # Filter
    fc_lo, fc_hi = fc & 0x07, (fc >> 3) & 0xFF
    frame[21] = fc_lo
    frame[22] = fc_hi
    frame[23] = res_filt
    frame[24] = mode_vol
    
    return bytes(frame)


def gen_arpeggio(
    notes: list[str],
    speed: int = 6,         # frames per note
    duration_sec: float = 30.0,
    waveform: int = 0x41,   # pulse + gate
    ad: int = 0x09,
    sr: int = 0x00,
) -> list[bytes]:
    """Generate an arpeggio pattern cycling through notes."""
    n_frames = int(duration_sec * 50)
    freqs = [NOTE_FREQS[n] for n in notes]
    frames = []
    
    for i in range(n_frames):
        note_idx = (i // speed) % len(freqs)
        freq = freqs[note_idx]
        frames.append(make_frame(
            v1_freq=freq, v1_ctrl=waveform, v1_ad=ad, v1_sr=sr,
            mode_vol=0x0F,
        ))
    
    return frames


def gen_pw_sweep(
    freq: int = 4291,       # C4
    sweep_speed: int = 16,  # PW increment per frame
    duration_sec: float = 30.0,
    waveform: int = 0x41,   # pulse + gate
) -> list[bytes]:
    """Generate a pulse width sweep on voice 1."""
    n_frames = int(duration_sec * 50)
    frames = []
    pw = 0x400  # start at 25%
    direction = 1
    
    for _ in range(n_frames):
        frames.append(make_frame(
            v1_freq=freq, v1_pw=pw, v1_ctrl=waveform,
            v1_ad=0x09, v1_sr=0x00, mode_vol=0x0F,
        ))
        pw += sweep_speed * direction
        if pw >= 0xF00:
            direction = -1
        elif pw <= 0x100:
            direction = 1
    
    return frames


def gen_filter_sweep(
    freq: int = 4291,
    sweep_speed: int = 8,
    duration_sec: float = 30.0,
) -> list[bytes]:
    """Generate a filter cutoff sweep with sawtooth wave."""
    n_frames = int(duration_sec * 50)
    frames = []
    fc = 0x100
    direction = 1
    
    for _ in range(n_frames):
        frames.append(make_frame(
            v1_freq=freq, v1_ctrl=0x21, v1_ad=0x09, v1_sr=0x00,  # sawtooth + gate
            fc=fc, res_filt=0xF1, mode_vol=0x1F,  # high resonance, lowpass, voice 1 filtered
        ))
        fc += sweep_speed * direction
        if fc >= 0x700:
            direction = -1
        elif fc <= 0x020:
            direction = 1
    
    return frames


def gen_combined(duration_sec: float = 60.0) -> list[bytes]:
    """Generate a combined pattern: arpeggio + PW sweep + filter sweep."""
    n_frames = int(duration_sec * 50)
    arp_notes = ["C4", "E4", "G4", "C5"]
    arp_freqs = [NOTE_FREQS[n] for n in arp_notes]
    frames = []
    
    pw = 0x400
    pw_dir = 1
    fc = 0x200
    fc_dir = 1
    
    for i in range(n_frames):
        # Voice 1: arpeggio
        note_idx = (i // 6) % len(arp_freqs)
        v1_freq = arp_freqs[note_idx]
        
        # Voice 2: bass note (one octave down, steady)
        v2_freq = arp_freqs[0] // 2
        
        frames.append(make_frame(
            v1_freq=v1_freq, v1_pw=pw, v1_ctrl=0x41, v1_ad=0x09, v1_sr=0x00,
            v2_freq=v2_freq, v2_ctrl=0x21, v2_ad=0x0C, v2_sr=0x0A,
            fc=fc, res_filt=0xA1, mode_vol=0x1F,
        ))
        
        pw += 8 * pw_dir
        if pw >= 0xE00:
            pw_dir = -1
        elif pw <= 0x200:
            pw_dir = 1
        
        fc += 4 * fc_dir
        if fc >= 0x600:
            fc_dir = -1
        elif fc <= 0x080:
            fc_dir = 1
    
    return frames


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SID corpus")
    parser.add_argument("--output", type=Path, default=Path("../data/synthetic.bin"))
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds")
    parser.add_argument(
        "--pattern",
        choices=["arpeggio", "pw_sweep", "filter_sweep", "combined", "all"],
        default="all",
        help="Pattern type to generate",
    )
    args = parser.parse_args()
    
    all_frames: list[bytes] = []
    duration = args.duration
    
    if args.pattern in ("arpeggio", "all"):
        print("Generating arpeggio (C-E-G-C)...")
        all_frames.extend(gen_arpeggio(
            ["C4", "E4", "G4", "C5"], duration_sec=duration / 4 if args.pattern == "all" else duration
        ))
    
    if args.pattern in ("pw_sweep", "all"):
        print("Generating pulse width sweep...")
        all_frames.extend(gen_pw_sweep(
            duration_sec=duration / 4 if args.pattern == "all" else duration
        ))
    
    if args.pattern in ("filter_sweep", "all"):
        print("Generating filter sweep...")
        all_frames.extend(gen_filter_sweep(
            duration_sec=duration / 4 if args.pattern == "all" else duration
        ))
    
    if args.pattern in ("combined", "all"):
        print("Generating combined pattern...")
        all_frames.extend(gen_combined(
            duration_sec=duration / 4 if args.pattern == "all" else duration
        ))
    
    # Write raw binary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        for frame in all_frames:
            f.write(frame)
    
    total_bytes = len(all_frames) * 25
    print(f"\nWrote {len(all_frames)} frames ({total_bytes:,} bytes) to {args.output}")
    print(f"Duration: {len(all_frames) / 50:.1f}s at 50Hz")


if __name__ == "__main__":
    main()
