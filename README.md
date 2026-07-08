# guppy

Implementing a Marlin-esque w4a16 (INT4 weight, FP16 activations) matmul kernel for Blackwell (sm_120). Same waters, smaller fish.

## Kernels

### Kernel 1: Naive
One CUDA thread computes one output element `output[m][n]`. Each thread independently loops over the `k` dimension and MACs.

```
running M16_N16_K128_g128_b4_s0          ... PASS  median=  0.0104 ms       6.27 GFLOPS      0.55 GB/s  (max_abs_err=0 max_rel_err=0)
running M128_N256_K1024_g128_b4_s1       ... PASS  median=  0.2060 ms     325.77 GFLOPS      2.27 GB/s  (max_abs_err=0.0625 max_rel_err=0.01083)
running M64_N128_K512_g128_b4_s0         ... PASS  median=  0.0321 ms     260.97 GFLOPS      3.63 GB/s  (max_abs_err=0.01562 max_rel_err=0.0009728)
```

### Kernel 2: Tiled
Threads in a block cooperatively load weights, activations, and scales into shared memory before computing. Threads still own one output element each, but now read shared memory instead of redundantly hitting global memory per MAC.

```
running M16_N16_K128_g128_b4_s0          ... PASS  median=  0.0064 ms      10.32 GFLOPS      0.90 GB/s  (max_abs_err=0 max_rel_err=0)
running M128_N256_K1024_g128_b4_s1       ... PASS  median=  0.0455 ms    1475.31 GFLOPS     10.27 GB/s  (max_abs_err=0.0625 max_rel_err=0.01083)
running M64_N128_K512_g128_b4_s0         ... PASS  median=  0.0157 ms     532.81 GFLOPS      7.41 GB/s  (max_abs_err=0.01562 max_rel_err=0.0009728)
```

## Resources
- [Marlin paper](https://arxiv.org/abs/2408.11743)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [Siboehm's matmul worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Aleksa Gordić's anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
- [Wafer's GPU perf engineering resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)
- [Pranjal Shankhdhar's Outperforming cuBLAS on H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [The Software Frontier's Mastering CUDA and HPC Series](https://www.thesoftwarefrontier.com/p/mastering-cuda-and-high-performance)