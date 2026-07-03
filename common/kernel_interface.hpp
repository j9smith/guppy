#pragma once
#ifndef GUPPY_COMMON_KERNEL_INTERFACE_HPP_
#define GUPPY_COMMON_KERNEL_INTERFACE_HPP_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

void launch_w4a16_matmul(const half* d_activations, const uint32_t* d_packed_qweight,
                         const float* d_scales, half* d_output, int M, int N, int K, int group_size,
                         cudaStream_t stream = 0);

#endif  // GUPPY_COMMON_KERNEL_INTERFACE_HPP_