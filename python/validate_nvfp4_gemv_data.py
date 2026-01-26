#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

M = 256
N = 1
K = 1024
BLOCK = 16


def fp4_e2m1_table() -> np.ndarray:
    """NVFP4-style E2M1 values (bias=1, no inf/NaN)."""
    table = np.zeros(16, dtype=np.float32)
    for code in range(16):
        sign = (code >> 3) & 0x1
        exp = (code >> 1) & 0x3
        mant = code & 0x1
        if exp == 0:
            value = 0.0 if mant == 0 else 0.5
        else:
            value = (1.0 + 0.5 * mant) * (2.0 ** (exp - 1))
        if sign:
            value = -value
        table[code] = value
    return table


def fp8_e4m3_table() -> np.ndarray:
    """FP8 E4M3 values (bias=7, exp=0xF reserved)."""
    table = np.empty(256, dtype=np.float32)
    table.fill(np.nan)
    for code in range(256):
        sign = (code >> 7) & 0x1
        exp = (code >> 3) & 0xF
        mant = code & 0x7
        if exp == 0:
            if mant == 0:
                value = 0.0
            else:
                value = (mant / 8.0) * (2.0 ** (1 - 7))
        elif exp == 0xF:
            value = np.nan
        else:
            value = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        if sign:
            value = -value
        table[code] = value
    return table


def read_hex_words(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as handle:
        return np.array([int(line.strip(), 16) for line in handle], dtype=np.uint32)


def unpack_u4(words: np.ndarray) -> np.ndarray:
    vals = np.empty(words.size * 8, dtype=np.uint8)
    for idx in range(8):
        vals[idx::8] = (words >> (4 * idx)) & 0xF
    return vals


def unpack_u8(words: np.ndarray) -> np.ndarray:
    vals = np.empty(words.size * 4, dtype=np.uint8)
    for idx in range(4):
        vals[idx::4] = (words >> (8 * idx)) & 0xFF
    return vals


def unpack_u16(words: np.ndarray) -> np.ndarray:
    vals = np.empty(words.size * 2, dtype=np.uint16)
    for idx in range(2):
        vals[idx::2] = (words >> (16 * idx)) & 0xFFFF
    return vals


def load_inputs(data_dir: Path) -> dict[str, np.ndarray]:
    required = {
        "a_fp4.txt": (M * K) // 8,
        "a_scale_fp8.txt": (M * (K // BLOCK)) // 4,
        "b_fp4.txt": (N * K) // 8,
        "b_scale_fp8.txt": (N * (K // BLOCK)) // 4,
        "c_fp16.txt": (M * N) // 2,
    }
    data = {}
    for name, expected_words in required.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing data file: {path}")
        words = read_hex_words(path)
        if words.size != expected_words:
            raise ValueError(
                f"{name}: expected {expected_words} lines, got {words.size}"
            )
        data[name] = words
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate NVFP4 GEMV output for 256x1x1024."
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "nvfp4_gemv_256x1x1024"),
        help="Directory containing nvfp4 gemv txt files",
    )
    parser.add_argument(
        "--show-mismatches",
        action="store_true",
        help="Print a few mismatching indices",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    try:
        inputs = load_inputs(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    a_codes = unpack_u4(inputs["a_fp4.txt"]).reshape(M, K)
    b_codes = unpack_u4(inputs["b_fp4.txt"]).reshape(N, K)
    a_scale_codes = unpack_u8(inputs["a_scale_fp8.txt"]).reshape(M, K // BLOCK)
    b_scale_codes = unpack_u8(inputs["b_scale_fp8.txt"]).reshape(N, K // BLOCK)
    c_ref = unpack_u16(inputs["c_fp16.txt"]).view(np.float16).reshape(M, N)

    fp4_table = fp4_e2m1_table()
    fp8_table = fp8_e4m3_table()

    a = fp4_table[a_codes]
    b = fp4_table[b_codes]
    a_scale = fp8_table[a_scale_codes]
    b_scale = fp8_table[b_scale_codes]

    if not np.isfinite(a_scale).all() or not np.isfinite(b_scale).all():
        print("Scale contains NaN/Inf values.", file=sys.stderr)
        return 3

    a_scaled = (a.reshape(M, K // BLOCK, BLOCK) * a_scale[..., None]).reshape(M, K)
    b_scaled = (b.reshape(N, K // BLOCK, BLOCK) * b_scale[..., None]).reshape(K)

    c_calc = (a_scaled @ b_scaled).astype(np.float16).reshape(M, 1)

    diff = c_calc.astype(np.float32) - c_ref.astype(np.float32)
    mismatch_mask = diff != 0
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    max_abs = float(np.max(np.abs(diff)))

    print(f"max_abs_diff: {max_abs}")
    print(f"mismatch_count: {mismatch_count}")

    if mismatch_count and args.show_mismatches:
        indices = np.argwhere(mismatch_mask)
        for idx in indices[:8]:
            i, j = int(idx[0]), int(idx[1])
            print(
                f"[{i},{j}] calc={c_calc[i, j]} ref={c_ref[i, j]} diff={diff[i, j]}"
            )

    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
