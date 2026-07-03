"""
Matmul correctness reference
"""

import numpy as np

from packing import elements_per_word, unpack
from quantize import dequantize


def dequantize_weights(
    packed_qweight: np.ndarray, scales: np.ndarray, group_size: int, num_bits: int = 4
) -> np.ndarray:
    """
    Unpack + dequantize in one step: packed int4 codes -> fp32 weights.

    Args:
        packed_qweight: (K // elements_per_word, N) uint32.
        scales: (K // group_size, N) fp32.
        group_size: rows per quantization group (must match quantize()).
        num_bits: bit width per element (must match quantize()/pack()).

    Returns:
        w_hat: (K, N) fp32.
    """
    num_groups = scales.shape[0]
    K = num_groups * group_size

    epw = elements_per_word(num_bits)
    if K % epw != 0:
        raise ValueError(
            f"K={K} (from scales.shape[0]={num_groups} * "
            f"group_size={group_size}) not divisible by "
            f"elements_per_word={epw}"
        )

    qweight = unpack(packed_qweight, K, num_bits=num_bits)
    return dequantize(qweight, scales, group_size, num_bits=num_bits)


def reference_matmul(
    activations: np.ndarray,
    packed_qweight: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    num_bits: int = 4,
) -> np.ndarray:
    """
    Ground-truth W4A16 GEMM: fp16 activations x packed int4 weights -> fp16.

    Args:
        activations: (M, K), any float dtype (will be cast to fp16).
        packed_qweight: (K // elements_per_word, N) uint32.
        scales: (K // group_size, N) fp32.
        group_size: rows per quantization group.
        num_bits: bit width per weight element.

    Returns:
        output: (M, N) fp16.
    """
    w_fp32 = dequantize_weights(packed_qweight, scales, group_size, num_bits)

    # Round-trip through fp16 first so operands match what the kernel
    # actually reads (fp16 activations, fp16-dequantized weights), then
    # upcast to fp32 purely for the accumulation, matching tensor-core
    # semantics (fp16 in, fp32 accumulate, fp16 out).
    a_fp16 = activations.astype(np.float16)
    w_fp16 = w_fp32.astype(np.float16)

    out_fp32 = a_fp16.astype(np.float32) @ w_fp16.astype(np.float32)
    return out_fp32.astype(np.float16)


if __name__ == "__main__":
    from packing import pack
    from quantize import quantize

    rng = np.random.default_rng(0)
    M, K, N, group_size, num_bits = 32, 256, 64, 128, 4

    activations = rng.normal(size=(M, K)).astype(np.float32)
    w_true = rng.normal(size=(K, N)).astype(np.float32)

    qweight, scales = quantize(w_true, group_size, num_bits=num_bits)
    packed = pack(qweight, num_bits=num_bits)

    w_hat_direct = dequantize(qweight, scales, group_size, num_bits=num_bits)
    w_hat_via_pack = dequantize_weights(packed, scales, group_size, num_bits=num_bits)
    assert np.array_equal(w_hat_direct, w_hat_via_pack), (
        "dequantize_weights(pack(q)) != dequantize(q) — packing changed values"
    )

    out = reference_matmul(activations, packed, scales, group_size, num_bits)
    assert out.shape == (M, N), f"output shape {out.shape} != {(M, N)}"
    assert out.dtype == np.float16, f"output dtype {out.dtype} != float16"

    out_unquantized = activations.astype(np.float16).astype(np.float32) @ w_true.astype(
        np.float16
    ).astype(np.float32)
    rel_err = np.abs(out.astype(np.float32) - out_unquantized) / (
        np.abs(out_unquantized) + 1e-3
    )
    print(f"output shape: {out.shape}, dtype: {out.dtype}")
    print(f"median relative error vs unquantized fp16 matmul: {np.median(rel_err):.4f}")
    print(f"max relative error vs unquantized fp16 matmul: {rel_err.max():.4f}")
    assert np.median(rel_err) < 0.5, (
        "median relative error implausibly large — check pipeline"
    )

    print("reference pipeline sanity checks passed.")
