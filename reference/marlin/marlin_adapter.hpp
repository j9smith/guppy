// reference/marlin/marlin_adapter.hpp
#pragma once
#include "marlin.cuh"

inline void launch_marlin_w4a16_matmul(const half* d_activations, const void* d_B_opaque,
                                       const void* d_scales_opaque, half* d_output, int M, int N,
                                       int K, int group_size, cudaStream_t stream = 0) {
    static int* d_locks = nullptr;
    static size_t locks_capacity = 0;
    const size_t needed = static_cast<size_t>(N) / 128 * 16;
    if (needed > locks_capacity) {
        if (d_locks) cudaFree(d_locks);
        cudaMalloc(&d_locks, needed * sizeof(int));
        locks_capacity = needed;
    }
    cudaMemset(d_locks, 0, needed * sizeof(int));

    marlin_cuda(d_activations, d_B_opaque, d_output, d_scales_opaque, M, N, K, d_locks, group_size,
                /*dev=*/0, stream);
}