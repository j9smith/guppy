// Register tiling

#include "../../common/utils.cuh"
#include "../../common/unpack.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// (BM/TM) * (BN/TN) should be a multiple of 32 to avoid half-empty warps
#define BM 64
#define BN 64
#define TM 8
#define TN 8

// BK must divide evenly into group_size so that
// each chunk stays within one group, requiring only
// one scale per column. Must also be a multiple of
// 8 to load packed weights
#define BK 16
#define NUM_THREADS 64 // must match block size in kernel launch

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

    const uint block_tid = blockDim.x * threadIdx.y + threadIdx.x;
    int subtile_row_idx = (block_tid / (BN/TN)) * TM;
    int subtile_col_idx = (block_tid % (BN/TN)) * TN;

    float acc[TM][TN] = {0.0f};

    for (int c = 0; c < num_chunks; c++) {
        // Load activations into smem
        for (int idx = block_tid; idx < BM * BK; idx += NUM_THREADS) {
            int chunk_act_row_id = idx / BK; 
            int chunk_act_col_id = idx % BK; 
            int glob_act_row_id = blockIdx.y * BM + chunk_act_row_id; 
            int glob_act_col_id = c * BK + chunk_act_col_id; 

            acts[idx] = (glob_act_row_id < M)
                ? activations[glob_act_row_id * K + glob_act_col_id]
                : __float2half(0.0f);
        }

        // Load weights into smem
        for (int idx = block_tid; idx < (BK * BN) / 8; idx += NUM_THREADS) {
            int chunk_w_row_id = idx / BN;
            int chunk_w_col_id = idx % BN;

            int glob_w_row_id = c * (BK / 8) + chunk_w_row_id;
            int glob_w_col_id = blockIdx.x * BN + chunk_w_col_id;

            int w_idx = glob_w_row_id * N + glob_w_col_id;

            weights[idx] = (glob_w_col_id < N)
            ? packed_qweight[w_idx]
            : 0u;
        }

        int group_idx = (c * BK) / group_size; 
        if (block_tid < BN) {
            int glob_scale_col = blockIdx.x * BN + block_tid;
            tile_scales[block_tid] = (glob_scale_col < N)
                ? scales[group_idx * N + glob_scale_col]
                : 0.0f;
        }

        __syncthreads();

        // Shared mem is populated, now we compute
        for (int k = 0; k < BK; k++) {
            float acts_reg[TM];
            float w_reg[TN];

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                // Walking the row one element at a time (across TM rows)
                // Each row's activ is offset by BK, then k indexes into the current column (k)
                acts_reg[tm] = acts[(subtile_row_idx + tm) * BK + k];
            }

            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                uint32_t word = weights[(k/8) * BN + (subtile_col_idx + tn)];
                int nibble = k % 8;
                float scale = tile_scales[subtile_col_idx + tn];
                
                w_reg[tn] = __half2float(__float2half(unpack_dequantize(word, nibble, scale)));
            }

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    acc[tm][tn] += acts_reg[tm] * w_reg[tn];
                }
            }
        }
        __syncthreads();
    }
    
    // Write to output
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int row = blockIdx.y * BM + subtile_row_idx + tm;
            int col = blockIdx.x * BN + subtile_col_idx + tn;
            if (row < M && col < N) {
            output[row * N + col] = __float2half(acc[tm][tn]);
            }
        }
    }
}

void launch_w4a16_matmul(
    const half* d_activations, const uint32_t* d_packed_qweight,
    const float* d_scales, half* d_output,
    int M, int N, int K, int group_size, cudaStream_t stream) {
        dim3 block(BM/TM, BN/TN);
        dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

        w4a16_matmul_kernel<<<grid, block, 0, stream>>>(
            d_activations, d_packed_qweight, d_scales, d_output,
            M, N, K, group_size);

        CUDA_CHECK(cudaGetLastError());
    }