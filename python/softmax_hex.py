#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def softmax_fp32(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    max_val = np.max(values)
    exps = np.exp(values - max_val)
    sum_val = np.sum(exps, dtype=np.float32)
    return (exps / sum_val).astype(np.float32, copy=False)


def write_hex_lines(values: np.ndarray, output_path: Path) -> None:
    hex_values = values.view(np.uint32)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        for value in hex_values:
            handle.write(f"{int(value):08x}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute FP32 softmax for 0..N-1 and write IEEE-754 hex lines."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=32000,
        help="Number of values to generate (default: 32000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("softmax_hex.txt"),
        help="Output txt path (default: softmax_hex.txt).",
    )
    args = parser.parse_args()

    values = np.arange(args.count, dtype=np.float32)
    softmax_values = softmax_fp32(values)
    write_hex_lines(softmax_values, args.output)


if __name__ == "__main__":
    main()
