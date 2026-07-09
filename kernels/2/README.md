# Register Tiling

```
running M16_N4096_K11008_g128_b4_s0      ... PASS  median=  1.8536 ms     778.38 GFLOPS     13.18 GB/s  (max_abs_err=0.25 max_rel_err=0.04158)
running M16_N16_K128_g128_b4_s0          ... PASS  median=  0.0260 ms       2.53 GFLOPS      0.22 GB/s  (max_abs_err=0 max_rel_err=0)
running M128_N256_K1024_g128_b4_s1       ... PASS  median=  0.1962 ms     342.11 GFLOPS      2.38 GB/s  (max_abs_err=0.0625 max_rel_err=0.01083)
running M64_N100_K256_g128_b4_s2         ... PASS  median=  0.0527 ms      62.14 GFLOPS      1.12 GB/s  (max_abs_err=0 max_rel_err=0)
running M64_N128_K192_g64_b4_s2          ... PASS  median=  0.0403 ms      78.14 GFLOPS      1.36 GB/s  (max_abs_err=0 max_rel_err=0)
running M1024_N4096_K4096_g128_b4_s0     ... PASS  median=  5.0972 ms    6740.93 GFLOPS      5.04 GB/s  (max_abs_err=0.125 max_rel_err=1.601)
running M256_N4096_K4096_g128_b4_s0      ... PASS  median=  1.3969 ms    6149.44 GFLOPS      9.38 GB/s  (max_abs_err=0.125 max_rel_err=0.2177)
running M16_N4096_K4096_g128_b4_s0       ... PASS  median=  0.7036 ms     763.03 GFLOPS     13.04 GB/s  (max_abs_err=0.125 max_rel_err=0.02526)
running M64_N128_K512_g128_b4_s0         ... PASS  median=  0.1017 ms      82.51 GFLOPS      1.15 GB/s  (max_abs_err=0.01562 max_rel_err=0.0009728)
running M17_N128_K256_g128_b4_s2         ... PASS  median=  0.0487 ms      22.89 GFLOPS      0.63 GB/s  (max_abs_err=0 max_rel_err=0)
running M256_N4096_K11008_g128_b4_s0     ... PASS  median=  3.8251 ms    6035.25 GFLOPS      8.28 GB/s  (max_abs_err=0.25 max_rel_err=10.98)
running M1_N4096_K4096_g128_b4_s0        ... PASS  median=  0.7025 ms      47.76 GFLOPS     12.71 GB/s  (max_abs_err=0.0625 max_rel_err=0.001517)
running M64_N4096_K4096_g128_b4_s0       ... PASS  median=  0.7836 ms    2740.65 GFLOPS     12.71 GB/s  (max_abs_err=0.125 max_rel_err=0.03594)
running M256_N8192_K8192_g128_b4_s0      ... PASS  median=  5.9229 ms    5801.14 GFLOPS      7.44 GB/s  (max_abs_err=0.25 max_rel_err=10.68)
running M16_N8192_K8192_g128_b4_s0       ... PASS  median=  1.9291 ms    1113.19 GFLOPS     18.75 GB/s  (max_abs_err=0.125 max_rel_err=0.1678)
```

- Blocks cover a $BM \times BN$ tile. If $N$ isn't a multiple of $BN$, the last block along the $N$ axis references garbage columns. On first implementation, there was no guard against this, and so indexes computed using their values, leading to certain shapes failing. Read guards prevent segfaults, and write guards prevent incorrectness.
- $TM=TN=4$ and $BM=BN=32$ had $96.50\%$ L1 throughput -- so we're bounded by shared memory reads. Increasing to $TM=TN=8$ and $BM=BN=64$ gave a significant uplift (~2000GLOPS for the better performing fixtures).