# Tiling

```
running M16_N4096_K11008_g128_b4_s0      ... PASS  median=  0.8299 ms    1738.53 GFLOPS     29.44 GB/s  (max_abs_err=0.25 max_rel_err=0.04158)
running M16_N16_K128_g128_b4_s0          ... PASS  median=  0.0056 ms      11.60 GFLOPS      1.01 GB/s  (max_abs_err=0 max_rel_err=0)
running M128_N256_K1024_g128_b4_s1       ... PASS  median=  0.0445 ms    1507.12 GFLOPS     10.49 GB/s  (max_abs_err=0.0625 max_rel_err=0.01083)
running M64_N100_K256_g128_b4_s2         ... PASS  median=  0.0098 ms     335.19 GFLOPS      6.05 GB/s  (max_abs_err=0 max_rel_err=0)
running M64_N128_K192_g64_b4_s2          ... PASS  median=  0.0076 ms     414.78 GFLOPS      7.22 GB/s  (max_abs_err=0 max_rel_err=0)
running M1024_N4096_K4096_g128_b4_s0     ... PASS  median= 16.1679 ms    2125.18 GFLOPS      1.59 GB/s  (max_abs_err=0.125 max_rel_err=1.601)
running M256_N4096_K4096_g128_b4_s0      ... PASS  median=  3.9807 ms    2157.91 GFLOPS      3.29 GB/s  (max_abs_err=0.125 max_rel_err=0.2177)
running M16_N4096_K4096_g128_b4_s0       ... PASS  median=  0.3169 ms    1693.98 GFLOPS     28.95 GB/s  (max_abs_err=0.125 max_rel_err=0.02526)
running M64_N128_K512_g128_b4_s0         ... PASS  median=  0.0158 ms     530.66 GFLOPS      7.38 GB/s  (max_abs_err=0.01562 max_rel_err=0.0009728)
running M17_N128_K256_g128_b4_s2         ... PASS  median=  0.0096 ms     116.44 GFLOPS      3.18 GB/s  (max_abs_err=0 max_rel_err=0)
running M256_N4096_K11008_g128_b4_s0     ... PASS  median= 11.1222 ms    2075.62 GFLOPS      2.85 GB/s  (max_abs_err=0.25 max_rel_err=10.98)
running M1_N4096_K4096_g128_b4_s0        ... PASS  median=  0.3171 ms     105.82 GFLOPS     28.16 GB/s  (max_abs_err=0.0625 max_rel_err=0.001517)
running M64_N4096_K4096_g128_b4_s0       ... PASS  median=  1.0417 ms    2061.43 GFLOPS      9.56 GB/s  (max_abs_err=0.125 max_rel_err=0.03594)
running M256_N8192_K8192_g128_b4_s0      ... PASS  median= 16.6542 ms    2063.12 GFLOPS      2.64 GB/s  (max_abs_err=0.25 max_rel_err=10.68)
running M16_N8192_K8192_g128_b4_s0       ... PASS  median=  1.2094 ms    1775.72 GFLOPS     29.91 GB/s  (max_abs_err=0.125 max_rel_err=0.1678)
```

- First implementation assumed that number of threads was always greater than number of elements. This failed as soon as that was not the case. **Fix:** add a strided loop that iterates $+\text{num\_threads}$. Adding runtime constant (`num_threads = blockDim.x * blockDim.y`) reduced flops by around 200 (can't perform loop unrolling at compile time). Replaced with compile time constant (`#define NUM_THREADS 256`) -- performance improved back to baseline.
- Also identified `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=568` -- 568 write conflicts to shared memory. This is caused by two consecutive 2-byte (`half`) activations writing to the 4-byte bank. The hardware is unable to service two concurrent writes to the same bank in one cycle, and so these writes are serialised. Because we observe L1 throughput at only 35%, this suggests that at any one point, only a fraction of blocks are performing writes. This means that although it may take an extra cycle to write per block, the SM is not sitting idle, and is instead able to issue work. Therefore, the latency is isolated to the block and does not materially propagate to the SM level. 
- Warp state statistics shows warps stall most on stall barrier (`__syncthreads`; ~2.9 cycles per instruction) and stall long scoreboard (load from global memory; ~1.8 cycles per instruction). Stall short scoreboard (load from local memory) lower at only ~0.5 cycles per instruction.
- For each FMA, we read one value from each of activations and weights shared memory to compute one element: 2 reads per FMA. Low arithmetic intensity. This motivates the next kernel -- if we instead compute a tile of outputs, like a $2\times 2$ block, we would load 2 activations and 2 weights (4 reads) to compute 4 outputs -- 1 read per FMA. At $4 \times 4$, we load 8 and compute 16 -- 0.5 reads per FMA, etc. 