#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

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
    """FP8 E4M3 values (bias=7, IEEE-like, exp=0xF reserved)."""
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


def pack_u4_to_u32(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.uint8).ravel()
    if vals.size % 8 != 0:
        raise ValueError("u4 values length must be a multiple of 8")
    vals = vals.reshape(-1, 8)
    words = np.zeros(vals.shape[0], dtype=np.uint32)
    for idx in range(8):
        words |= (vals[:, idx].astype(np.uint32) & 0xF) << (4 * idx)
    return words


def pack_u8_to_u32(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.uint8).ravel()
    if vals.size % 4 != 0:
        raise ValueError("u8 values length must be a multiple of 4")
    vals = vals.reshape(-1, 4)
    words = np.zeros(vals.shape[0], dtype=np.uint32)
    for idx in range(4):
        words |= vals[:, idx].astype(np.uint32) << (8 * idx)
    return words


def pack_u16_to_u32(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.uint16).ravel()
    if vals.size % 2 != 0:
        raise ValueError("u16 values length must be a multiple of 2")
    vals = vals.reshape(-1, 2)
    words = np.zeros(vals.shape[0], dtype=np.uint32)
    for idx in range(2):
        words |= vals[:, idx].astype(np.uint32) << (16 * idx)
    return words


def write_hex_lines(path: Path, words: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as handle:
        for word in words:
            handle.write(f"{int(word):08x}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NVFP4 GEMV test data (256x1x1024)."
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "nvfp4_gemv_256x1x1024"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp4_table = fp4_e2m1_table()
    fp8_table = fp8_e4m3_table()

    # Repeat row patterns to make manual comparison easier.
    a_row_codes = np.tile(np.arange(16, dtype=np.uint8), K // 16)
    a_codes = np.tile(a_row_codes, (M, 1))

    b_row_codes = np.tile(np.arange(15, -1, -1, dtype=np.uint8), K // 16)
    b_codes = b_row_codes.reshape(1, K)

    scale_a_row_codes = np.tile(np.arange(0x20, 0x30, dtype=np.uint8), (K // BLOCK) // 16)
    scale_a_codes = np.tile(scale_a_row_codes, (M, 1))

    scale_b_row_codes = np.tile(np.arange(0x18, 0x28, dtype=np.uint8), (K // BLOCK) // 16)
    scale_b_codes = scale_b_row_codes.reshape(1, K // BLOCK)

    a_dequant = fp4_table[a_codes]
    b_dequant = fp4_table[b_codes]
    scale_a_dequant = fp8_table[scale_a_codes]
    scale_b_dequant = fp8_table[scale_b_codes]

    a_scaled = a_dequant.reshape(M, K // BLOCK, BLOCK) * scale_a_dequant[..., None]
    b_scaled = b_dequant.reshape(N, K // BLOCK, BLOCK) * scale_b_dequant[..., None]
    a_scaled = a_scaled.reshape(M, K).astype(np.float32)
    b_scaled = b_scaled.reshape(K).astype(np.float32)

    c_fp16 = (a_scaled @ b_scaled).astype(np.float16)

    write_hex_lines(out_dir / "a_fp4.txt", pack_u4_to_u32(a_codes))
    write_hex_lines(out_dir / "a_scale_fp8.txt", pack_u8_to_u32(scale_a_codes))
    write_hex_lines(out_dir / "b_fp4.txt", pack_u4_to_u32(b_codes))
    write_hex_lines(out_dir / "b_scale_fp8.txt", pack_u8_to_u32(scale_b_codes))
    write_hex_lines(out_dir / "c_fp16.txt", pack_u16_to_u32(c_fp16.view(np.uint16)))


if __name__ == "__main__":
    main()
