#pragma once
#ifndef GUPPY_COMMON_UNPACK_CUH_
#define GUPPY_COMMON_UNPACK_CUH_

#include <cstdint>

#ifdef __CUDACC__
#define W4A16_INLINE __device__ __forceinline__
#else
#define W4A16_INLINE inline
#endif

constexpr int kNumBits = 4;
constexpr int kElementsPerWord = 32 / kNumBits; // 8
constexpr int kZeroPoint = 1 << (kNumBits - 1); // 8
constexpr uint32_t kCodeMask = (1u << kNumBits) - 1; // 0xF

static_assert(32 % kNumBits == 0, "kNumBits must evenly divide 32");

W4A16_INLINE int unpack_code(uint32_t word, int j) {
    return static_cast<int>((word >> (j * kNumBits)) & kCodeMask);
}

W4A16_INLINE float dequantize_code(int code, float scale) {
    return static_cast<float>(code - kZeroPoint) * scale;
}

W4A16_INLINE float unpack_dequantize(uint32_t word, int j, float scale) {
    return dequantize_code(unpack_code(word, j), scale);
}

#endif  // GUPPY_COMMON_UNPACK_CUH_
