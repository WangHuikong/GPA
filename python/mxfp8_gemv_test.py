#!/usr/bin/env python3
import argparse
import numpy as np

E4M3_BIAS = 7
E8M0_BIAS = 127


def load_hex_u32(path, expected_count=None):
    def line_iter():
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                if text.startswith(("0x", "0X")):
                    text = text[2:]
                yield int(text, 16)

    count = expected_count if expected_count is not None else -1
    data = np.fromiter(line_iter(), dtype=np.uint32, count=count)
    if expected_count is not None and data.size != expected_count:
        raise ValueError(
            f"{path} has {data.size} entries, expected {expected_count}"
        )
    return data


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
        a_vals *= fp8_e8m0_to_f32(scale_a_u8[group])

        b_vals = fp8_e4m3_to_f32(b_u8[start:end, :])
        b_vals *= fp8_e8m0_to_f32(scale_b_u8[group])[None, :]

        out += a_vals.astype(np.float32) @ b_vals.astype(np.float32)

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute MXFP8 GEMV output from hex text inputs. "
            "Each line stores one 32-bit hex value."
        )
    )
    parser.add_argument("--a", required=True, help="A input hex file")
    parser.add_argument("--scale-a", required=True, help="Scale A hex file")
    parser.add_argument("--b", required=True, help="B input hex file")
    parser.add_argument("--scale-b", required=True, help="Scale B hex file")
    parser.add_argument("--out", required=True, help="Output hex file")
    parser.add_argument("--k", type=int, default=4096, help="K dimension")
    parser.add_argument("--n", type=int, default=4096, help="N dimension")
    parser.add_argument(
        "--group-size", type=int, default=32, help="Group size for scaling"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.k % args.group_size != 0:
        raise ValueError("K must be a multiple of group-size")
    groups = args.k // args.group_size

    a_u32 = load_hex_u32(args.a, expected_count=args.k)
    scale_a_u32 = load_hex_u32(args.scale_a, expected_count=groups)
    b_u32 = load_hex_u32(args.b, expected_count=args.k * args.n)
    scale_b_u32 = load_hex_u32(args.scale_b, expected_count=groups * args.n)

    a_u8 = a_u32.astype(np.uint8)
    scale_a_u8 = scale_a_u32.astype(np.uint8)
    b_u8 = b_u32.astype(np.uint8).reshape(args.k, args.n)
    scale_b_u8 = scale_b_u32.astype(np.uint8).reshape(groups, args.n)

    out = mxfp8_gemv(a_u8, scale_a_u8, b_u8, scale_b_u8, args.group_size)
    out_u32 = np.ascontiguousarray(out.astype(np.float32)).view(np.uint32)
    write_hex_u32(args.out, out_u32)


if __name__ == "__main__":
    main()
