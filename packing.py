"""
Take INT4 quantized weights and pack them into bytes (two elements per byte).
"""

import numpy as np


def elements_per_word(num_bits: int, word_bits: int = 32) -> int:
    """How many num_bits-wide codes fit in one word_bits-wide word."""
    if word_bits % num_bits != 0:
        raise ValueError(f"word_bits={word_bits} not divisible by num_bits={num_bits}")
    return word_bits // num_bits


def pack(qweight: np.ndarray, num_bits: int = 4) -> np.ndarray:
    """
    Pack unsigned num_bits-wide codes into 32-bit words along the K axis.

    Args:
        qweight: (K, N) int-typed array, values in [0, 2^num_bits - 1].
        num_bits: bit width per element (4 for W4A16).

    Returns:
        packed: (K // elements_per_word, N) uint32.
    """
    K, N = qweight.shape
    epw = elements_per_word(num_bits)
    if K % epw != 0:
        raise ValueError(f"K={K} not divisible by elements_per_word={epw}")

    qm = (1 << num_bits) - 1
    if qweight.min() < 0 or qweight.max() > qm:
        raise ValueError(
            f"qweight values must be in [0, {qm}] for num_bits={num_bits}, "
            f"got range [{qweight.min()}, {qweight.max()}]"
        )

    num_words = K // epw
    grouped = qweight.reshape(num_words, epw, N).astype(np.uint32)

    packed = np.zeros((num_words, N), dtype=np.uint32)
    for j in range(epw):
        # OR (|) our elements into place (packed is all zeros)
        # Use the j'th index to appropriately apply bitshift via j * num_bits
        # w0 lands in [0:4), w1 lands in [4,8), etc., w7 lands in [28,32)
        # packed so far:      0000 0000 0000 0000 0000 0000 1010 0011   (slots 0,1 filled)
        # new value (j=2)
        # shifted into place:  0000 0000 0000 0000 0000 1101 0000 0000
        #                   ------------------------------------------  OR
        # result:              0000 0000 0000 0000 0000 1101 1010 0011
        packed |= grouped[:, j, :] << np.uint32(j * num_bits)

    return packed


def unpack(packed: np.ndarray, K: int, num_bits: int = 4) -> np.ndarray:
    """
    Inverse of pack(). Reconstructs the unpacked qweight array.

    Args:
        packed: (K // elements_per_word, N) uint32.
        K: number of rows the caller expects (must match what pack() used).
        num_bits: bit width per element (must match what pack() used).

    Returns:
        qweight: (K, N) int32, values in [0, 2^num_bits - 1].
    """
    epw = elements_per_word(num_bits)
    num_words, N = packed.shape
    if num_words * epw != K:
        raise ValueError(
            f"packed has {num_words} words * {epw} elements/word = "
            f"{num_words * epw}, expected K={K}"
        )

    mask = np.uint32((1 << num_bits) - 1)
    grouped = np.zeros((num_words, epw, N), dtype=np.int32)
    for j in range(epw):
        grouped[:, j, :] = ((packed >> np.uint32(j * num_bits)) & mask).astype(np.int32)

    return grouped.reshape(K, N)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, N, num_bits = 256, 64, 4
    qm = (1 << num_bits) - 1

    qweight = rng.integers(0, qm + 1, size=(K, N), dtype=np.int32)

    packed = pack(qweight, num_bits=num_bits)
    qweight_hat = unpack(packed, K, num_bits=num_bits)

    expected_words = K // elements_per_word(num_bits)
    assert packed.shape == (expected_words, N), (
        f"packed shape {packed.shape} != expected {(expected_words, N)}"
    )
    assert packed.dtype == np.uint32
    assert np.array_equal(qweight, qweight_hat), "round-trip mismatch (not bit-exact)"

    w, n = 0, 0
    word = int(packed[w, n])
    for j in range(elements_per_word(num_bits)):
        expected = int(qweight[w * elements_per_word(num_bits) + j, n])
        actual = (word >> (j * num_bits)) & qm
        assert actual == expected, (
            f"bit layout mismatch at j={j}: {actual} != {expected}"
        )

    print(f"packed shape: {packed.shape}, dtype: {packed.dtype}")
    print(f"word[0,0] = 0x{word:08x}")
    print("round-trip bit-exact check passed.")
