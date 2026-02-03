#!/usr/bin/env python3
import argparse
import numpy as np

E4M3_BIAS = 7
E8M0_BIAS = 127


def write_hex_u32(path, data_u32):
    with open(path, "w", encoding="utf-8") as handle:
        for value in data_u32:
            handle.write(f"{int(value) & 0xFFFFFFFF:08x}\n")


def write_hex_f32(path, data_f32):
    data_u32 = np.ascontiguousarray(data_f32, dtype=np.float32).view(np.uint32)
    write_hex_u32(path, data_u32)


def write_hex_u8_packed4(path, data_u8):
    data_u8 = np.asarray(data_u8, dtype=np.uint8).ravel()
    if data_u8.size % 4 != 0:
        raise ValueError("Packed fp8 output requires length multiple of 4")
    packed = data_u8.reshape(-1, 4).astype(np.uint32)
    packed = (
        packed[:, 0]
        | (packed[:, 1] << 8)
        | (packed[:, 2] << 16)
        | (packed[:, 3] << 24)
    )
    write_hex_u32(path, packed)


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


def float_to_fp8_e4m3(values_f32):
    values_f32 = np.asarray(values_f32, dtype=np.float32)
    out = np.zeros(values_f32.shape, dtype=np.uint8)

    sign = values_f32 < 0
    abs_vals = np.abs(values_f32)
    max_finite = np.float32(240.0)

    mask_zero = abs_vals == 0
    mask_nan = np.isnan(abs_vals)
    mask_inf = np.isinf(abs_vals) | (abs_vals > max_finite)

    if np.any(mask_nan):
        out[mask_nan] = np.uint8(0x7F)
    if np.any(mask_inf):
        out[mask_inf] = np.uint8(0x78)

    mask_work = ~(mask_zero | mask_nan | mask_inf)
    if np.any(mask_work):
        vals = abs_vals[mask_work]
        exp = np.floor(np.log2(vals)).astype(np.int32)

        encoded = np.zeros(vals.shape, dtype=np.uint8)
        mask_sub = exp < -6
        if np.any(mask_sub):
            sub_vals = vals[mask_sub]
            mant = np.rint(sub_vals / np.float32(2**-6)).astype(np.int32)
            mant = np.clip(mant, 0, 7)
            encoded[mask_sub] = mant.astype(np.uint8)

        mask_norm = ~mask_sub
        if np.any(mask_norm):
            exp_norm = exp[mask_norm]
            exp_norm = np.clip(exp_norm, -6, 7)
            scaled = vals[mask_norm] / np.exp2(exp_norm.astype(np.float32))
            mant = np.rint((scaled - np.float32(1.0)) * np.float32(8.0)).astype(
                np.int32
            )
            carry = mant == 8
            if np.any(carry):
                mant[carry] = 0
                exp_norm[carry] += 1

            overflow = exp_norm > 7
            if np.any(overflow):
                exp_norm[overflow] = 7
                mant[overflow] = 7

            encoded[mask_norm] = (
                ((exp_norm + E4M3_BIAS) << 3) | mant
            ).astype(np.uint8)

        out[mask_work] = encoded

    if np.any(sign):
        out[sign] |= np.uint8(0x80)
    return out


def generate_e4m3(rng, shape):
    sign = rng.integers(0, 2, size=shape, dtype=np.uint8)
    exp = rng.integers(0, 0xF, size=shape, dtype=np.uint8)
    mant = rng.integers(0, 8, size=shape, dtype=np.uint8)
    return (sign << 7) | (exp << 3) | mant


def generate_e8m0(rng, shape):
    return rng.integers(1, 0xFF, size=shape, dtype=np.uint8)


def generate_ones_ramp_fp8(k, n, group_size):
    groups = k // group_size
    a_u8 = np.full((k,), 0x38, dtype=np.uint8)
    scale_a_u8 = np.full((groups,), 0x7F, dtype=np.uint8)

    desired = np.arange(1, k * n + 1, dtype=np.float32).reshape(k, n)
    cols = np.arange(n, dtype=np.int64)[None, :]
    row_max = (np.arange(groups, dtype=np.int64) * group_size + (group_size - 1))[
        :, None
    ]
    max_desired = (row_max * n + cols + 1).astype(np.float32)

    max_finite = np.float32(240.0)
    ratio = max_desired / max_finite
    scale_pow = np.maximum(0, np.ceil(np.log2(ratio))).astype(np.int32)
    scale_b = np.exp2(scale_pow).astype(np.float32)
    scale_b_u8 = np.clip(scale_pow + E8M0_BIAS, 1, 254).astype(np.uint8)

    scale_b_full = np.repeat(scale_b, group_size, axis=0)
    b_vals = desired / scale_b_full
    b_u8 = float_to_fp8_e4m3(b_vals)

    return a_u8, scale_a_u8, b_u8, scale_b_u8, desired


def generate_ones_1p5_fp8(k, n, group_size):
    groups = k // group_size
    a_u8 = np.full((k,), 0x38, dtype=np.uint8)
    scale_a_u8 = np.full((groups,), 0x7F, dtype=np.uint8)
    b_u8 = np.full((k, n), 0x3C, dtype=np.uint8)
    scale_b_u8 = np.full((groups, n), 0x7F, dtype=np.uint8)
    return a_u8, scale_a_u8, b_u8, scale_b_u8


def generate_ones_ramp_f32(k, n, group_size):
    groups = k // group_size
    a = np.ones((k,), dtype=np.float32)
    scale_a = np.ones((groups,), dtype=np.float32)
    b = np.arange(1, k * n + 1, dtype=np.float32).reshape(k, n)
    scale_b = np.ones((groups, n), dtype=np.float32)
    return a, scale_a, b, scale_b, b


def generate_ones_1p5_f32(k, n, group_size):
    groups = k // group_size
    a = np.ones((k,), dtype=np.float32)
    scale_a = np.ones((groups,), dtype=np.float32)
    b = np.full((k, n), 1.5, dtype=np.float32)
    scale_b = np.ones((groups, n), dtype=np.float32)
    return a, scale_a, b, scale_b


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


def gemv_f32(a, scale_a, b, scale_b, group_size):
    k = a.size
    if k % group_size != 0:
        raise ValueError("K must be a multiple of group_size")
    groups = k // group_size
    n = b.shape[1]
    out = np.zeros((n,), dtype=np.float32)

    for group in range(groups):
        start = group * group_size
        end = start + group_size

        a_vals = a[start:end] * scale_a[group]
        b_vals = b[start:end, :] * scale_b[group][None, :]
        out += a_vals.astype(np.float32) @ b_vals.astype(np.float32)

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate MXFP8 GEMV inputs, write hex files, then compute output. "
            "Each line stores one 32-bit hex value. Input files are overwritten. "
            "For fp8 encoding, four 8-bit values are packed per line "
            "(little-endian: first value in lowest byte)."
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
        "--pattern",
        choices=["ones_1p5", "ones_ramp", "random", "zeros"],
        default="ones_1p5",
        help="Input pattern for generated data",
    )
    parser.add_argument(
        "--encoding",
        choices=["fp8", "f32"],
        default="fp8",
        help="Input encoding format written to hex files",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for random pattern"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exact ones/ramp after scaling in fp8 mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.k % args.group_size != 0:
        raise ValueError("K must be a multiple of group-size")
    groups = args.k // args.group_size

    if args.encoding == "f32":
        if args.pattern == "ones_1p5":
            a_f32, scale_a_f32, b_f32, scale_b_f32 = generate_ones_1p5_f32(
                args.k, args.n, args.group_size
            )
        elif args.pattern == "ones_ramp":
            a_f32, scale_a_f32, b_f32, scale_b_f32, _ = generate_ones_ramp_f32(
                args.k, args.n, args.group_size
            )
        elif args.pattern == "zeros":
            a_f32 = np.zeros((args.k,), dtype=np.float32)
            scale_a_f32 = np.zeros((groups,), dtype=np.float32)
            b_f32 = np.zeros((args.k, args.n), dtype=np.float32)
            scale_b_f32 = np.zeros((groups, args.n), dtype=np.float32)
        else:
            rng = np.random.default_rng(args.seed)
            a_f32 = rng.standard_normal((args.k,), dtype=np.float32)
            scale_a_f32 = rng.standard_normal((groups,), dtype=np.float32)
            b_f32 = rng.standard_normal((args.k, args.n), dtype=np.float32)
            scale_b_f32 = rng.standard_normal((groups, args.n), dtype=np.float32)

        write_hex_f32(args.a, a_f32)
        write_hex_f32(args.scale_a, scale_a_f32)
        write_hex_f32(args.b, b_f32.ravel())
        write_hex_f32(args.scale_b, scale_b_f32.ravel())

        out = gemv_f32(a_f32, scale_a_f32, b_f32, scale_b_f32, args.group_size)
        write_hex_f32(args.out, out.astype(np.float32))
        return

    if args.pattern == "ones_1p5":
        a_u8, scale_a_u8, b_u8, scale_b_u8 = generate_ones_1p5_fp8(
            args.k, args.n, args.group_size
        )
        desired = None
    elif args.pattern == "ones_ramp":
        a_u8, scale_a_u8, b_u8, scale_b_u8, desired = generate_ones_ramp_fp8(
            args.k, args.n, args.group_size
        )
    elif args.pattern == "zeros":
        a_u8 = np.zeros((args.k,), dtype=np.uint8)
        scale_a_u8 = np.zeros((groups,), dtype=np.uint8)
        b_u8 = np.zeros((args.k, args.n), dtype=np.uint8)
        scale_b_u8 = np.zeros((groups, args.n), dtype=np.uint8)
        desired = None
    else:
        rng = np.random.default_rng(args.seed)
        a_u8 = generate_e4m3(rng, (args.k,))
        scale_a_u8 = generate_e8m0(rng, (groups,))
        b_u8 = generate_e4m3(rng, (args.k, args.n))
        scale_b_u8 = generate_e8m0(rng, (groups, args.n))
        desired = None

    write_hex_u8_packed4(args.a, a_u8)
    write_hex_u8_packed4(args.scale_a, scale_a_u8)
    write_hex_u8_packed4(args.b, b_u8.ravel())
    write_hex_u8_packed4(args.scale_b, scale_b_u8.ravel())

    if args.strict and args.pattern == "ones_ramp":
        b_f32 = fp8_e4m3_to_f32(b_u8)
        scale_b_f32 = fp8_e8m0_to_f32(scale_b_u8)
        scale_b_full = np.repeat(scale_b_f32, args.group_size, axis=0)
        actual = b_f32 * scale_b_full
        if not np.array_equal(actual, desired):
            raise ValueError(
                "ones_ramp is not exactly representable in fp8; "
                "use --encoding f32 or reduce k/n."
            )

    out = mxfp8_gemv(a_u8, scale_a_u8, b_u8, scale_b_u8, args.group_size)
    out_u32 = np.ascontiguousarray(out.astype(np.float32)).view(np.uint32)
    write_hex_u32(args.out, out_u32)


if __name__ == "__main__":
    main()
