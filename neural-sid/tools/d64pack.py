"""
Pack C64 .prg and weight files into a standard .d64 disk image.

Creates a CBM DOS formatted disk image (35 tracks, 683 sectors)
with the inference program and weight data as files.

.d64 format:
  - 35 tracks, variable sectors per track (21/19/18/17)
  - 256 bytes per sector
  - Track 18 = directory + BAM
  - Total: 174,848 bytes (683 sectors × 256 bytes)
"""

import argparse
import struct
from pathlib import Path


# Sectors per track for standard 35-track .d64
SECTORS_PER_TRACK = [
    0,   # track 0 doesn't exist
    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,  # 1-17
    19, 19, 19, 19, 19, 19, 19,  # 18-24
    18, 18, 18, 18, 18, 18,  # 25-30
    17, 17, 17, 17, 17,  # 31-35
]

TOTAL_SECTORS = sum(SECTORS_PER_TRACK[1:])  # 683
D64_SIZE = TOTAL_SECTORS * 256  # 174,848

DIR_TRACK = 18
DIR_SECTOR = 1
BAM_TRACK = 18
BAM_SECTOR = 0

# CBM DOS file types
FILE_PRG = 0x82  # PRG, closed
FILE_SEQ = 0x81  # SEQ, closed


def track_sector_offset(track: int, sector: int) -> int:
    """Calculate byte offset in .d64 for a given track/sector."""
    offset = 0
    for t in range(1, track):
        offset += SECTORS_PER_TRACK[t] * 256
    offset += sector * 256
    return offset


def petscii_pad(name: str, length: int = 16) -> bytes:
    """Convert ASCII name to PETSCII, padded with $A0."""
    name = name.upper()[:length]
    result = bytearray(name.encode("ascii"))
    result.extend([0xA0] * (length - len(result)))
    return bytes(result)


class D64Image:
    """Builder for a CBM DOS .d64 disk image."""
    
    def __init__(self, disk_name: str = "NEURAL SID", disk_id: str = "NS"):
        self.data = bytearray(D64_SIZE)
        self.disk_name = disk_name
        self.disk_id = disk_id
        self.files: list[tuple[str, int, bytes]] = []  # (name, type, data)
        
        # Track BAM allocation
        self.bam_free = {}
        for t in range(1, 36):
            self.bam_free[t] = list(range(SECTORS_PER_TRACK[t]))
        
        # Reserve track 18 sector 0 (BAM) and sector 1 (first dir sector)
        self._alloc_sector(BAM_TRACK, BAM_SECTOR)
        self._alloc_sector(DIR_TRACK, DIR_SECTOR)
    
    def _alloc_sector(self, track: int, sector: int) -> None:
        """Mark a sector as allocated."""
        if sector in self.bam_free.get(track, []):
            self.bam_free[track].remove(sector)
    
    def _next_free_sector(self, prefer_track: int = 1) -> tuple[int, int]:
        """Find next free sector, starting from preferred track.
        Skip track 18 (directory)."""
        # Search outward from prefer_track
        for t in range(prefer_track, 36):
            if t == 18:
                continue
            if self.bam_free[t]:
                s = self.bam_free[t][0]
                self._alloc_sector(t, s)
                return t, s
        for t in range(1, prefer_track):
            if t == 18:
                continue
            if self.bam_free[t]:
                s = self.bam_free[t][0]
                self._alloc_sector(t, s)
                return t, s
        raise RuntimeError("Disk full!")
    
    def _write_sector(self, track: int, sector: int, data: bytes) -> None:
        """Write 256 bytes to a sector."""
        offset = track_sector_offset(track, sector)
        self.data[offset : offset + 256] = data[:256].ljust(256, b"\x00")
    
    def add_file(self, name: str, file_type: int, content: bytes) -> None:
        """Add a file to the disk."""
        self.files.append((name, file_type, content))
    
    def add_prg(self, name: str, prg_data: bytes) -> None:
        """Add a PRG file (includes 2-byte load address header)."""
        self.add_file(name, FILE_PRG, prg_data)
    
    def add_seq(self, name: str, data: bytes) -> None:
        """Add a SEQ file."""
        self.add_file(name, FILE_SEQ, data)
    
    def _write_file_chain(self, content: bytes) -> tuple[int, int, int]:
        """Write file data as a chain of sectors. Returns (first_track, first_sector, n_sectors)."""
        chunks = []
        offset = 0
        while offset < len(content):
            chunk = content[offset : offset + 254]  # 254 bytes per sector (2 bytes for chain link)
            chunks.append(chunk)
            offset += 254
        
        if not chunks:
            chunks = [b""]
        
        # Allocate sectors
        sectors = []
        for _ in chunks:
            sectors.append(self._next_free_sector())
        
        # Write chain
        for i, (chunk, (t, s)) in enumerate(zip(chunks, sectors)):
            sector_data = bytearray(256)
            if i < len(sectors) - 1:
                next_t, next_s = sectors[i + 1]
                sector_data[0] = next_t
                sector_data[1] = next_s
            else:
                sector_data[0] = 0x00  # last sector
                sector_data[1] = len(chunk) + 1  # bytes used in last sector
            sector_data[2 : 2 + len(chunk)] = chunk
            self._write_sector(t, s, bytes(sector_data))
        
        return sectors[0][0], sectors[0][1], len(sectors)
    
    def _write_bam(self) -> None:
        """Write the Block Availability Map (track 18, sector 0)."""
        bam = bytearray(256)
        
        # Pointer to first directory sector
        bam[0] = DIR_TRACK
        bam[1] = DIR_SECTOR
        bam[2] = 0x41  # DOS version (A)
        bam[3] = 0x00  # unused
        
        # BAM entries for tracks 1-35 (4 bytes each, starting at offset 4)
        for t in range(1, 36):
            offset = 4 + (t - 1) * 4
            free_count = len(self.bam_free[t])
            bam[offset] = free_count
            
            # Bitmap: 3 bytes, bit set = free
            bitmap = 0
            for s in range(SECTORS_PER_TRACK[t]):
                if s in self.bam_free[t]:
                    bitmap |= 1 << s
            bam[offset + 1] = bitmap & 0xFF
            bam[offset + 2] = (bitmap >> 8) & 0xFF
            bam[offset + 3] = (bitmap >> 16) & 0xFF
        
        # Disk name and ID at offset 144
        disk_name = petscii_pad(self.disk_name, 16)
        bam[144:160] = disk_name
        bam[160] = 0xA0
        bam[161] = 0xA0
        disk_id = self.disk_id.upper().encode("ascii")[:2].ljust(2, b" ")
        bam[162:164] = disk_id
        bam[164] = 0xA0
        bam[165:167] = b"2A"  # DOS type
        bam[167:171] = b"\xA0" * 4
        
        self._write_sector(BAM_TRACK, BAM_SECTOR, bytes(bam))
    
    def _write_directory(self) -> None:
        """Write directory entries."""
        # Each directory sector holds 8 entries of 32 bytes each
        # First directory sector: track 18, sector 1
        dir_data = bytearray(256)
        dir_data[0] = 0x00  # no next directory sector
        dir_data[1] = 0xFF
        
        for i, (name, file_type, _) in enumerate(self.files):
            if i >= 8:
                break  # TODO: multi-sector directory
            
            file_info = self.file_locations[i]
            if i == 0:
                off = 2  # first entry starts at byte 2 (after chain link)
                entry = dir_data
            else:
                off = i * 32
                entry = dir_data
            
            entry[off + 0] = file_type
            entry[off + 1] = file_info[0]  # first track
            entry[off + 2] = file_info[1]  # first sector
            entry[off + 3 : off + 19] = petscii_pad(name, 16)
            # File size in sectors (little-endian u16 at offset 28-29 relative to entry start)
            n_sectors = file_info[2]
            entry[off + 28] = n_sectors & 0xFF
            entry[off + 29] = (n_sectors >> 8) & 0xFF
        
        self._write_sector(DIR_TRACK, DIR_SECTOR, bytes(dir_data))
    
    def build(self) -> bytes:
        """Build the complete .d64 image."""
        # Write all file data first
        self.file_locations = []
        for name, file_type, content in self.files:
            t, s, n = self._write_file_chain(content)
            self.file_locations.append((t, s, n))
            print(f"  {name}: {len(content)} bytes, {n} sectors, starts at T{t}/S{s}")
        
        # Write directory and BAM
        self._write_directory()
        self._write_bam()
        
        return bytes(self.data)


def main():
    parser = argparse.ArgumentParser(description="Create .d64 disk image")
    parser.add_argument("--prg", type=Path, help="C64 .prg file to include")
    parser.add_argument("--weights", type=Path, help="Weight .bin file to include")
    parser.add_argument("--output", type=Path, required=True, help="Output .d64 path")
    parser.add_argument("--disk-name", type=str, default="NEURAL SID")
    args = parser.parse_args()
    
    img = D64Image(disk_name=args.disk_name)
    
    if args.prg:
        prg_data = args.prg.read_bytes()
        img.add_prg("NEURAL SID", prg_data)
        print(f"Added PRG: {args.prg} ({len(prg_data)} bytes)")
    
    if args.weights:
        weight_data = args.weights.read_bytes()
        img.add_seq("WEIGHTS", weight_data)
        print(f"Added weights: {args.weights} ({len(weight_data)} bytes)")
    
    d64_data = img.build()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(d64_data)
    print(f"\nWrote {args.output} ({len(d64_data):,} bytes)")


if __name__ == "__main__":
    main()
