#!/usr/bin/env python3
import argparse
import numpy as np

E4M3_BIAS = 7
E8M0_BIAS = 127


def write_hex_u32(path, data_u32):
    with open(path, "w", encoding="utf-8") as handle:
        for value in data_u32:
            handle.write(f"{int(value) & 0xFFFFFFFF:08x}\n")


def fp8_e4m3_to_f32(values_u8):
    values_u8 = np.asarray(values_u8, dtype=np.uint8)
    sign = (values_u8 >> 7) & 0x1
    exp = (values_u8 >> 3) & 0xF
    mant = values_u8 & 0x7

    out = np.empty(values_u8.shape, dtype=np.float32)

    mask_zero = exp == 0
    mask_mant_zero = mant == 0
    out[mask_zero & mask_mant_zero] = np.float32(0.0)
    mask_sub = mask_zero & ~mask_mant_zero
    if np.any(mask_sub):
        sub = mant[mask_sub].astype(np.float32) / np.float32(8.0)
        out[mask_sub] = np.ldexp(sub, 1 - E4M3_BIAS).astype(np.float32)

    mask_exp_max = exp == 0xF
    out[mask_exp_max & mask_mant_zero] = np.float32(np.inf)
    out[mask_exp_max & ~mask_mant_zero] = np.float32(np.nan)

    mask_norm = ~(mask_zero | mask_exp_max)
    if np.any(mask_norm):
        frac = np.float32(1.0) + mant[mask_norm].astype(np.float32) / np.float32(8.0)
        out[mask_norm] = np.ldexp(
            frac, exp[mask_norm].astype(np.int32) - E4M3_BIAS
        ).astype(np.float32)

    sign_mask = sign == 1
    out[sign_mask] = -out[sign_mask]
    return out


def fp8_e8m0_to_f32(values_u8):
    values_u8 = np.asarray(values_u8, dtype=np.uint8)
    out = np.empty(values_u8.shape, dtype=np.float32)
    mask_zero = values_u8 == 0
    mask_inf = values_u8 == 0xFF
    mask_norm = ~(mask_zero | mask_inf)

    out[mask_zero] = np.float32(0.0)
    out[mask_inf] = np.float32(np.inf)
    if np.any(mask_norm):
        out[mask_norm] = np.exp2(
            values_u8[mask_norm].astype(np.int32) - E8M0_BIAS
        ).astype(np.float32)
    return out


def generate_e4m3(rng, shape):
    sign = rng.integers(0, 2, size=shape, dtype=np.uint8)
    exp = rng.integers(0, 0xF, size=shape, dtype=np.uint8)
    mant = rng.integers(0, 8, size=shape, dtype=np.uint8)
    return (sign << 7) | (exp << 3) | mant


def generate_e8m0(rng, shape):
    return rng.integers(1, 0xFF, size=shape, dtype=np.uint8)


def mxfp8_gemv(a_u8, scale_a_u8, b_u8, scale_b_u8, group_size):
    k = a_u8.size
    if k % group_size != 0:
        raise ValueError("K must be a multiple of group_size")
    groups = k // group_size
    n = b_u8.shape[1]
    out = np.zeros((n,), dtype=np.float32)

    for group in range(groups):
        start = group * group_size
        end = start + group_size

        a_vals = fp8_e4m3_to_f32(a_u8[start:end])
        b_vals = fp8_e4m3_to_f32(b_u8[start:end, :])
        scale_a = fp8_e8m0_to_f32(scale_a_u8[group])
        scale_b = fp8_e8m0_to_f32(scale_b_u8[group])[None, :]

        with np.errstate(over="ignore", invalid="ignore"):
            a_vals *= scale_a
            b_vals *= scale_b
            out += a_vals.astype(np.float32) @ b_vals.astype(np.float32)

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate MXFP8 inputs, write hex files, then compute GEMV output. "
            "Each line stores one 32-bit hex value. Input files are overwritten."
        )
    )
    parser.add_argument("--a", required=True, help="A input hex file (overwrite)")
    parser.add_argument(
        "--scale-a", required=True, help="Scale A hex file (overwrite)"
    )
    parser.add_argument("--b", required=True, help="B input hex file (overwrite)")
    parser.add_argument(
        "--scale-b", required=True, help="Scale B hex file (overwrite)"
    )
    parser.add_argument("--out", required=True, help="Output hex file")
    parser.add_argument("--k", type=int, default=4096, help="K dimension")
    parser.add_argument("--n", type=int, default=4096, help="N dimension")
    parser.add_argument(
        "--group-size", type=int, default=32, help="Group size for scaling"
    )
    parser.add_argument(
        "--init",
        choices=["random", "zeros"],
        default="random",
        help="Initialize inputs with random values or zeros",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for input generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.k % args.group_size != 0:
        raise ValueError("K must be a multiple of group-size")
    groups = args.k // args.group_size

    if args.init == "zeros":
        a_u8 = np.zeros((args.k,), dtype=np.uint8)
        scale_a_u8 = np.zeros((groups,), dtype=np.uint8)
        b_u8 = np.zeros((args.k, args.n), dtype=np.uint8)
        scale_b_u8 = np.zeros((groups, args.n), dtype=np.uint8)
    else:
        rng = np.random.default_rng(args.seed)
        a_u8 = generate_e4m3(rng, (args.k,))
        scale_a_u8 = generate_e8m0(rng, (groups,))
        b_u8 = generate_e4m3(rng, (args.k, args.n))
        scale_b_u8 = generate_e8m0(rng, (groups, args.n))

    write_hex_u32(args.a, a_u8.astype(np.uint32))
    write_hex_u32(args.scale_a, scale_a_u8.astype(np.uint32))
    write_hex_u32(args.b, b_u8.astype(np.uint32).ravel())
    write_hex_u32(args.scale_b, scale_b_u8.astype(np.uint32).ravel())

    out = mxfp8_gemv(a_u8, scale_a_u8, b_u8, scale_b_u8, args.group_size)
    out_u32 = np.ascontiguousarray(out.astype(np.float32)).view(np.uint32)
    write_hex_u32(args.out, out_u32)


if __name__ == "__main__":
    main()
