# Tiling

| Metric | Value |
|---|---|
| Compute (SM) Throughput | 62.18% |
| Memory Throughput | 32.25% |
| L1/TEX Cache Throughput | 35.14% |
| L2 Cache Throughput | 8.42% |
| DRAM Throughput | 2.46% |

```
running M16_N16_K128_g128_b4_s0          ... PASS  median=  0.0062 ms      10.50 GFLOPS      0.91 GB/s  (max_abs_err=0 max_rel_err=0)
running M128_N256_K1024_g128_b4_s1       ... PASS  median=  0.0452 ms    1486.29 GFLOPS     10.34 GB/s  (max_abs_err=0.0625 max_rel_err=0.01083)
running M64_N128_K512_g128_b4_s0         ... PASS  median=  0.0153 ms     546.70 GFLOPS      7.61 GB/s  (max_abs_err=0.01562 max_rel_err=0.0009728)
```

- First implementation assumed that number of threads was always greater than number of elements. This failed as soon as that was not the case. **Fix:** add a strided loop that iterates $+\text{num\_threads}$. Adding runtime constant (`num_threads = blockDim.x * blockDim.y`) reduced flops by around 200 (can't perform loop unrolling at compile time). Replaced with compile time constant (`#define NUM_THREADS 256`) -- performance improved back to baseline.
- Also identified `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=568` -- 568 write conflicts to shared memory. This is caused by two consecutive 2-byte (`half`) activations writing to the 4-byte bank. The hardware is unable to service two concurrent writes to the same bank in one cycle, and so these writes are serialised. Because we observe L1 throughput at only 35%, this suggests that at any one point, only a fraction of blocks are performing writes. This means that although it may take an extra cycle to write per block, the SM is not sitting idle, and is instead able to issue work. Therefore, the latency is isolated to the block and does not materially propagate to the SM level. 
- Warp state statistics shows warps stall most on stall barrier (`__syncthreads`; ~2.9 cycles per instruction) and stall long scoreboard (load from global memory; ~1.8 cycles per instruction). Stall short scoreboard (load from local memory) lower at only ~0.5 cycles per instruction.
- For each FMA, we read one value from each of activations and weights shared memory to compute one element: 2 reads per FMA. Low arithmetic intensity. This motivates the next kernel -- if we instead compute a tile of outputs, like a $2\times 2$ block, we would load 2 activations and 2 weights (4 reads) to compute 4 outputs -- 1 read per FMA. At $4 \times 4$, we load 8 and compute 16 -- 0.5 reads per FMA, etc. 