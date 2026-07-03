"""
Performs per-group symmetric INT4 quantization by calculating weights and scales etc.
"""

import numpy as np


def qmax(num_bits: int) -> int:
    """Largest representable unsigned code, e.g. 15 for 4-bit."""
    return (1 << num_bits) - 1


def zero_point(num_bits: int) -> int:
    """
    Fixed symmetric zero-point, e.g. 8 for 4-bit.

    Storage is unsigned in [0, qmax]. The zero-point is the code that
    represents 0.0. For 4-bit: codes [0,15] represent signed range
    [-8, 7] via (code - 8).
    """
    return 1 << (num_bits - 1)


def quantize(w: np.ndarray, group_size: int, num_bits: int = 4):
    """
    Symmetric group-wise RTN quantization.

    Args:
        w: (K, N) fp32/fp16 weight matrix.
        group_size: number of rows (K) per group. Must divide K evenly.
        num_bits: bit width of quantized weights (4 for W4A16).

    Returns:
        qweight: (K, N) int32, unsigned codes in [0, qmax(num_bits)].
        scales:  (K // group_size, N) fp32, one scale per group per column.
    """
    K, N = w.shape
    if K % group_size != 0:
        raise ValueError(f"K={K} not divisible by group_size={group_size}")
    num_groups = K // group_size
    zp = zero_point(num_bits)
    qm = qmax(num_bits)
    denom = min(zp, qm - zp)

    w = w.astype(np.float32)
    qweight = np.empty((K, N), dtype=np.int32)
    scales = np.empty((num_groups, N), dtype=np.float32)

    for g in range(num_groups):
        row_lo, row_hi = g * group_size, (g + 1) * group_size
        w_group = w[row_lo:row_hi, :]

        max_abs = np.abs(w_group).max(axis=0)
        scale = np.where(max_abs > 0, max_abs / denom, 1.0).astype(np.float32)

        q = np.round(w_group / scale) + zp
        q = np.clip(q, 0, qm)

        qweight[row_lo:row_hi, :] = q.astype(np.int32)
        scales[g, :] = scale

    return qweight, scales


def dequantize(
    qweight: np.ndarray, scales: np.ndarray, group_size: int, num_bits: int = 4
) -> np.ndarray:
    """
    Inverse of quantize(). Reconstructs fp32 weights.

    Args:
        qweight: (K, N) int32 unsigned codes in [0, qmax(num_bits)].
        scales:  (num_groups, N) fp32.
        group_size: rows per group (must match what quantize() used).
        num_bits: bit width used during quantization.

    Returns:
        w_hat: (K, N) fp32 reconstructed weights.
    """
    K, N = qweight.shape
    num_groups = K // group_size
    if scales.shape != (num_groups, N):
        raise ValueError(f"scales shape {scales.shape} != expected {(num_groups, N)}")
    zp = zero_point(num_bits)

    w_hat = np.empty((K, N), dtype=np.float32)
    for g in range(num_groups):
        row_lo, row_hi = g * group_size, (g + 1) * group_size
        w_hat[row_lo:row_hi, :] = (
            qweight[row_lo:row_hi, :].astype(np.float32) - zp
        ) * scales[g, :]

    return w_hat


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, N, group_size = 256, 64, 128
    w = rng.normal(size=(K, N)).astype(np.float32)

    qweight, scales = quantize(w, group_size, num_bits=4)
    w_hat = dequantize(qweight, scales, group_size, num_bits=4)

    err = np.abs(w - w_hat)
    for g in range(K // group_size):
        row_lo, row_hi = g * group_size, (g + 1) * group_size
        group_err = err[row_lo:row_hi, :]
        max_abs = np.abs(w[row_lo:row_hi, :]).max(axis=0)
        denom = min(zero_point(4), qmax(4) - zero_point(4))
        bound = max_abs / denom / 2 + 1e-6
        assert (group_err <= bound[None, :] + 1e-5).all(), (
            f"group {g} exceeded expected RTN error bound"
        )

    print(
        f"qweight range: [{qweight.min()}, {qweight.max()}] (expected [0, {qmax(4)}])"
    )
    print(f"max abs reconstruction error: {err.max():.6f}")
    print(f"mean abs reconstruction error: {err.mean():.6f}")
    print("round-trip sanity check passed.")
