#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def softmax_details_fp32(values: np.ndarray) -> tuple[np.ndarray, np.float32, np.ndarray]:
    values = values.astype(np.float32, copy=False)
    max_val = np.max(values)
    exps = np.exp(values - max_val).astype(np.float32, copy=False)
    sum_val = np.sum(exps, dtype=np.float32)
    softmax = (exps / sum_val).astype(np.float32, copy=False)
    return exps, np.float32(sum_val), softmax


def softmax_fp32(values: np.ndarray) -> np.ndarray:
    _, _, softmax = softmax_details_fp32(values)
    return softmax


def write_hex_lines(values: np.ndarray, output_path: Path) -> None:
    hex_values = values.view(np.uint32)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        for value in hex_values:
            handle.write(f"{int(value):08x}\n")


def float_to_hex(value: np.float32) -> str:
    bits = np.float32(value).view(np.uint32)
    return f"{int(bits):08x}"


def format_fp32(value: np.float32) -> str:
    return f"{np.float32(value):.8e}"


def write_csv_details(
    exps: np.ndarray,
    sum_val: np.float32,
    softmax: np.ndarray,
    output_path: Path,
) -> None:
    sum_hex = float_to_hex(sum_val)
    sum_fp32 = format_fp32(sum_val)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        for exp_val, soft_val in zip(exps, softmax):
            handle.write(
                f"{float_to_hex(exp_val)},{format_fp32(exp_val)},"
                f"{sum_hex},{sum_fp32},"
                f"{float_to_hex(soft_val)},{format_fp32(soft_val)}\n"
            )


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
        "--hex-output",
        dest="hex_output",
        type=Path,
        default=Path("softmax_hex.txt"),
        help="Output txt path for softmax hex (default: softmax_hex.txt).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("softmax_details.csv"),
        help="Output csv path for exp/sum/softmax details.",
    )
    args = parser.parse_args()

    values = np.arange(args.count, dtype=np.float32)
    exps, sum_val, softmax_values = softmax_details_fp32(values)
    write_hex_lines(softmax_values, args.hex_output)
    write_csv_details(exps, sum_val, softmax_values, args.csv_output)


if __name__ == "__main__":
    main()
