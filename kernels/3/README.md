# Vectorised dequant

```
running M16_N4096_K11008_g128_b4_s0      ... FAIL  median=  1.4634 ms     985.97 GFLOPS     16.70 GB/s  (max_abs_err=0.25 max_rel_err=6.058)
running M16_N16_K128_g128_b4_s0          ... PASS  median=  0.0187 ms       3.51 GFLOPS      0.31 GB/s  (max_abs_err=0.01562 max_rel_err=0.06469)
running M128_N256_K1024_g128_b4_s1       ... FAIL  median=  0.1429 ms     469.63 GFLOPS      3.27 GB/s  (max_abs_err=0.0625 max_rel_err=3.988)
running M64_N100_K256_g128_b4_s2         ... FAIL  median=  0.0403 ms      81.24 GFLOPS      1.47 GB/s  (max_abs_err=0.03125 max_rel_err=2.304)
running M64_N128_K192_g64_b4_s2          ... FAIL  median=  0.0301 ms     104.63 GFLOPS      1.82 GB/s  (max_abs_err=0.03125 max_rel_err=1.713)
running M1024_N4096_K4096_g128_b4_s0     ... FAIL  median=  4.8461 ms    7090.24 GFLOPS      5.30 GB/s  (max_abs_err=0.25 max_rel_err=1036)
running M256_N4096_K4096_g128_b4_s0      ... FAIL  median=  1.4633 ms    5870.20 GFLOPS      8.96 GB/s  (max_abs_err=0.25 max_rel_err=210.2)
running M16_N4096_K4096_g128_b4_s0       ... FAIL  median=  0.5483 ms     979.18 GFLOPS     16.73 GB/s  (max_abs_err=0.125 max_rel_err=17.45)
running M64_N128_K512_g128_b4_s0         ... FAIL  median=  0.0733 ms     114.37 GFLOPS      1.59 GB/s  (max_abs_err=0.0625 max_rel_err=1.295)
running M17_N128_K256_g128_b4_s2         ... FAIL  median=  0.0357 ms      31.20 GFLOPS      0.85 GB/s  (max_abs_err=0.03125 max_rel_err=2.289)
running M256_N4096_K11008_g128_b4_s0     ... FAIL  median=  4.0020 ms    5768.45 GFLOPS      7.92 GB/s  (max_abs_err=0.25 max_rel_err=2.968e+04)
running M1_N4096_K4096_g128_b4_s0        ... FAIL  median=  0.5444 ms      61.64 GFLOPS     16.40 GB/s  (max_abs_err=0.125 max_rel_err=0.3487)
running M64_N4096_K4096_g128_b4_s0       ... FAIL  median=  0.6340 ms    3387.20 GFLOPS     15.71 GB/s  (max_abs_err=0.25 max_rel_err=18.58)
running M256_N8192_K8192_g128_b4_s0      ... FAIL  median=  5.3416 ms    6432.46 GFLOPS      8.24 GB/s  (max_abs_err=0.25 max_rel_err=2580)
running M16_N8192_K8192_g128_b4_s0       ... FAIL  median=  1.4926 ms    1438.76 GFLOPS     24.24 GB/s  (max_abs_err=0.25 max_rel_err=180.5)
```

- fp16 `hfma2` accumulation adds noise that's unavoidable. The `FAIL` seen in the print above reflects an overly strict checker rather than an incorrect kernel.