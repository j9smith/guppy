// Register tiling + vectorized lop3/hfma2 dequant (Stage 3b)

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

// Returns exact integers (q - 8) as half2, no scale applied
__device__ __forceinline__ half2 unpack_pair(uint32_t word, int i) {
    uint32_t combined;
    uint32_t shifted = word >> (4 * i);

    // AND mask to slice only 4 lower bits of each half word (shifted)
    // then OR them into the 10-bit mantissa of a 16-bit float
    // 0x6400 bias gives exponent 2^10, so mantissa least-significant bit
    // lands at 1 (in [1024, 2048)).
    // so we recover our weight as integer+1024 in fp16
    asm volatile(
        // lop3.b32 d, a, b, c, immLut
        // bitwise logical operation on inputs a, b, c then store result in d
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3
        // immLut 0xea = (a & b) | c
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(combined)
        : "r"(shifted), "r"(0x000F000Fu), "r"(0x64006400u), "n"(0xea)
    );

    // our zero-point is shifted by 8 for symmetry
    // instead of 0x6400 (1024), we subtract 0x6408 (1024+8=1032)
    // to restore our unscaled weight
    const half2 magic = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    return __hsub2(*reinterpret_cast<half2*>(&combined), magic);
}

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
    int subtile_row_idx = (block_tid / (BN / TN)) * TM;
    int subtile_col_idx = (block_tid % (BN / TN)) * TN;

    // accumulate in fp32
    float acc[TM][TN] = { 0.0f };

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

        // Load this chunk's scales into registers
        float col_scale[TN];
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            col_scale[tn] = tile_scales[subtile_col_idx + tn];
        }

        // Shared mem is populated, now we compute
        // Iterate over words (weights) in chunk
        for (int kw = 0; kw < BK; kw += 8) {
            // Walk columns and pull out their words
            uint32_t word[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                word[tn] = weights[(kw / 8) * BN + (subtile_col_idx + tn)];
            }

            // fp16 partial accumulator
            half2 part[TM][TN];
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    // overwrite existing (garbage) acc values with 0.0
                    part[tm][tn] = __float2half2_rn(0.0f);
                }
            }

            // Each lop3 unpacks 2 weights (nibbles i and i+4)
            // We need to run 4 times to unpack entire word (8 weights)
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int k = kw + i;

                half2 acts_reg[TM];
                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    half a0 = acts[(subtile_row_idx + tm) * BK + k];
                    // lop3 unpacks weight i and i+4, so pair activs k and k+4
                    half a1 = acts[(subtile_row_idx + tm) * BK + k + 4];

                    // store 2 halves in half2 (2xfp16 in 1 32bit register)
                    acts_reg[tm] = __halves2half2(a0, a1);
                }

                half2 w_h2[TN];
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    // 2xfp16 in 32bit register -> 2xfp32 in 2x32bit registers
                    // scales in 32 bit, so weights need to be 32 bit
                    float2 wf = __half22float2(unpack_pair(word[tn], i));

                    // narrow fp32 products back into fp16, then repack
                    w_h2[tn] = __halves2half2(__float2half(wf.x * col_scale[tn]),
                                              __float2half(wf.y * col_scale[tn]));
                }

                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < TN; tn++) {
                        // FMA over two side by side fp16 in 32 bit register, in one instruction
                        // __hfma2(a, b, c) = a * b + c
                        // https://docs.nvidia.com/cuda/archive/9.1/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html
                        part[tm][tn] = __hfma2(acts_reg[tm], w_h2[tn], part[tm][tn]);
                    }
                }
            }

            // add fp16 partials into the fp32 accumulator
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    // __low2float (bits 0-15)/__high2float (bits 16-31) operate on packed fp16 pairs
                    acc[tm][tn] += __low2float(part[tm][tn]) + __high2float(part[tm][tn]);
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
    dim3 block(BM / TM, BN / TN);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

    w4a16_matmul_kernel<<<grid, block, 0, stream>>>(
        d_activations, d_packed_qweight, d_scales, d_output,
        M, N, K, group_size);

    CUDA_CHECK(cudaGetLastError());
}
