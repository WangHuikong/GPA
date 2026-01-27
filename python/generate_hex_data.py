#!/usr/bin/env python3
"""
Generate the hex data sequence:
00010000, 00030002, ... , fffffffe

Each line is 8 lowercase hex digits: odd word then even word.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def write_hex_data(output_path: Path) -> None:
    last_even = 0xFFFE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        for even in range(0, last_even + 1, 2):
            odd = even + 1
            line = f"{odd:04x}{even:04x}"
            if even != last_even:
                line += "\n"
            handle.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hex data sequence to a file."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="hex_data.txt",
        help="Output file path (default: hex_data.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_hex_data(Path(args.output))


if __name__ == "__main__":
    main()
