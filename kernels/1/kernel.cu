// Tiling + shared mem

#include "../../common/utils.cuh"
#include "../../common/unpack.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define BM 16
#define BN 16
// BK must divide evenly into group_size so that
// each chunk stays within one group, requiring only
// one scale per column. Must also be a multiple of
// 8 to load packed weights
#define BK 16 

__global__ void w4a16_matmul_kernel(
    const half* __restrict__ activations, 
    const uint32_t* __restrict__ packed_qweight,
    const float* __restrict__ scales,
    half* __restrict__ output,
    int M, int N, int K, int group_size
) {
    int num_chunks = K / BK;
    __shared__ half acts[BM * BK];
    __shared__ uint32_t weights[(BK / 8) * BN]; 
    __shared__ float tile_scales[BN];

    const uint j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint i = blockIdx.y * blockDim.y + threadIdx.y;

    // Block local thread id, used for the cooperative loading phase
    const uint block_tid = blockDim.x * threadIdx.y + threadIdx.x; 

    float acc = 0.0;

    for (int c = 0; c < num_chunks; c++) {
        // Load activations into smem
        int chunk_act_row_id = block_tid / BK; // local row within acts chunk this thread loads
        int chunk_act_col_id = block_tid % BK; // local col within acts chunk this thread loads
        int glob_act_row_id = blockIdx.y * BM + chunk_act_row_id; // the global M row being loaded
        int glob_act_col_id = c * BK + chunk_act_col_id; // the global K column being loaded (offset by chunk)

        acts[block_tid] = activations[glob_act_row_id * K + glob_act_col_id];
        
        // Load weights into smem
        if (block_tid < (BK * BN) / 8) { 
            int chunk_w_row_id = block_tid / BN;
            int chunk_w_col_id = block_tid % BN;

            int glob_w_row_id = c * (BK / 8) + chunk_w_row_id;
            int glob_w_col_id = blockIdx.x * BN + chunk_w_col_id;

            int w_idx = glob_w_row_id * N + glob_w_col_id; 
            weights[block_tid] = packed_qweight[w_idx];
        }

        // Load scales into smem
        int group_idx = (c * BK) / group_size; // which group this chunk belongs to
        if (block_tid < BN) {
            // One scale per output col, valid for the whole chunk
            // because group_size % BK == 0
            tile_scales[block_tid] = scales[group_idx * N + (blockIdx.x * BN + block_tid % BN)];
        }

        __syncthreads();
        // Shared mem is populated, now we compute
        for (int k = 0; k < BK; k++) {
            int word_row = k / 8; // k = 0 through to 7, this = 0; k = 8 to 15, = 1, etc.
            int word_col = threadIdx.x; // 0 < word_col < blockDim.x - 1
            int nibble_idx = k % 8; // offset within word
            
            uint32_t word = weights[word_row * BN + word_col]; 
            float scale = tile_scales[word_col];

            // Round the dequantised weight through fp16 (fp32 has higher precision -> error)
            float w = __half2float(__float2half(unpack_dequantize(word, nibble_idx, scale)));
            float a = acts[threadIdx.y * BK + k];

            acc += w * a;
        }
        __syncthreads();
    }
    output[i * N + j] = __float2half(acc);
}

void launch_w4a16_matmul(
    const half* d_activations, const uint32_t* d_packed_qweight,
    const float* d_scales, half* d_output,
    int M, int N, int K, int group_size, cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));

        w4a16_matmul_kernel<<<grid, block, 0, stream>>>(
            d_activations, d_packed_qweight, d_scales, d_output,
            M, N, K, group_size);

        CUDA_CHECK(cudaGetLastError());
    }