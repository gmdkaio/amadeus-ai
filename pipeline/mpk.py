"""Reader for MAGES .mpk archives (format v2, used by the Steam Steins;Gate releases).

Layout:
    0x00  magic   b"MPK\\0"
    0x04  u16 minor, u16 major version
    0x08  u64 entry count
    0x40  entry table, 256 bytes per entry:
          u32 compressed flag, u32 index, u64 offset, u64 size,
          u64 uncompressed size, 224 byte null-terminated name

Usage:
    python -m pipeline.mpk list sg0 chara --filter 'CRS_*.png'
    python -m pipeline.mpk extract sg voice out_dir --filter 'CRS_*'
"""

import argparse
import fnmatch
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"MPK\x00"
HEADER_SIZE = 0x40
ENTRY_SIZE = 256
ENTRY_FORMAT = "<IIQQQ"


@dataclass(frozen=True)
class Entry:
    index: int
    name: str
    offset: int
    size: int
    compressed: bool


class MpkArchive:
    def __init__(self, path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        try:
            self.entries = self._read_table()
        except Exception:
            self._file.close()
            raise

    def _read_table(self):
        header = self._file.read(HEADER_SIZE)
        if header[:4] != MAGIC:
            raise ValueError(f"{self.path} is not an MPK archive")

        minor, major = struct.unpack_from("<HH", header, 4)
        if major != 2:
            raise ValueError(f"{self.path}: MPK v{major}.{minor} not supported, only v2")

        (count,) = struct.unpack_from("<Q", header, 8)
        table = self._file.read(count * ENTRY_SIZE)
        if len(table) != count * ENTRY_SIZE:
            raise ValueError(f"{self.path}: entry table is truncated")

        entries = []
        for i in range(count):
            raw = table[i * ENTRY_SIZE:(i + 1) * ENTRY_SIZE]
            compressed, index, offset, size, _ = struct.unpack_from(ENTRY_FORMAT, raw)
            name = raw[32:].split(b"\x00", 1)[0].decode("utf-8")
            if not name:
                # Unnamed slots show up in some shader.mpk tables, nothing to extract
                continue
            entries.append(Entry(index, name, offset, size, bool(compressed)))
        return entries

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._file.close()

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def filter(self, pattern):
        """Entries whose name matches a glob, case-insensitive."""
        pattern = pattern.lower()
        return [e for e in self.entries if fnmatch.fnmatchcase(e.name.lower(), pattern)]

    def find(self, name):
        """Entry by exact name, case-insensitive."""
        name = name.lower()
        for entry in self.entries:
            if entry.name.lower() == name:
                return entry
        raise KeyError(f"{name} not in {self.path.name}")

    def _check(self, entry):
        if entry.compressed:
            # None of the Steam archives use this, so the codec is unverified
            raise NotImplementedError(f"{entry.name} is compressed, not supported yet")

    def read(self, entry):
        self._check(entry)
        self._file.seek(entry.offset)
        return self._file.read(entry.size)

    def extract(self, entry, out_dir):
        """Copy an entry to out_dir/<name> in chunks and return the path."""
        self._check(entry)
        out_path = Path(out_dir) / entry.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self._file.seek(entry.offset)
        remaining = entry.size
        with open(out_path, "wb") as out:
            while remaining:
                chunk = self._file.read(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError(f"{self.path.name}: {entry.name} runs past end of file")
                out.write(chunk)
                remaining -= len(chunk)
        return out_path


def main():
    from pipeline.games import archive_path

    parser = argparse.ArgumentParser(prog="python -m pipeline.mpk")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in ("list", "extract"):
        p = sub.add_parser(command)
        p.add_argument("game", help="sg, sg0, phenogram or darling")
        p.add_argument("archive", help="archive name, e.g. voice or chara")
        if command == "extract":
            p.add_argument("out_dir", type=Path)
        p.add_argument("--filter", default="*", help="glob on entry names, case-insensitive")

    args = parser.parse_args()

    with MpkArchive(archive_path(args.game, args.archive)) as archive:
        entries = archive.filter(args.filter)
        if args.command == "list":
            for entry in entries:
                print(f"{entry.index:6} {entry.size:12} {entry.name}")
            print(f"{len(entries)} of {len(archive)} entries")
        else:
            for entry in entries:
                archive.extract(entry, args.out_dir)
            print(f"extracted {len(entries)} entries to {args.out_dir}")


if __name__ == "__main__":
    main()
