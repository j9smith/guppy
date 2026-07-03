// Ground-truth kernel

#include "../../common/utils.cuh"
#include "../../common/unpack.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>


__global__ void w4a16_matmul_kernel(
    const half* __restrict__ activations, 
    const uint32_t* __restrict__ packed_qweight,
    const float* __restrict__ scales,
    half* __restrict__ output,
    int M, int N, int K, int group_size
) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float acc = 0.0;
        for (int i = 0; i < K; i++) {
            int word_idx = i / kElementsPerWord;
            int j = i % kElementsPerWord;
            int group_idx = i / group_size;
            float w;

            w = __half2float(__float2half(unpack_dequantize(packed_qweight[word_idx * N + y], j, scales[group_idx * N + y])));

            acc += __half2float(activations[x * K + i]) * w;
        }
        output[x * N + y] = __float2half(acc);
    }
}

void launch_w4a16_matmul(
    const half* d_activations, const uint32_t* d_packed_qweight,
    const float* d_scales, half* d_output,
    int M, int N, int K, int group_size, cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid(ceil_div(M, block.x), ceil_div(N, block.y));

        w4a16_matmul_kernel<<<grid, block, 0, stream>>>(
            d_activations, d_packed_qweight, d_scales, d_output,
            M, N, K, group_size);

        CUDA_CHECK(cudaGetLastError());
    }