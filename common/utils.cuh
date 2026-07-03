#pragma once
#ifndef GUPPY_COMMON_UTILS_CUH_
#define GUPPY_COMMON_UTILS_CUH_

#include <cstdlib>   
#include <iostream>  

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                       << " - " << cudaGetErrorString(err) << "\n";        \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)


__host__ __device__ constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#endif  // GUPPY_COMMON_UTILS_CUH_