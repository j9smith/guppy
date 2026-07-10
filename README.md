# guppy

Implementing a Marlin-esque w4a16 (INT4 weight, FP16 activations) matmul kernel for Blackwell (sm_120). Same waters, smaller fish.

## Kernels

### [Kernel 1: Naive](./kernels/0)
One CUDA thread computes one output element `output[m][n]`. Each thread independently loops over the `k` dimension and MACs.

### [Kernel 2: Tiled](./kernels/1)
Threads in a block cooperatively load weights, activations, and scales into shared memory before computing. Threads still own one output element each, but now read shared memory instead of redundantly hitting global memory per MAC.

### [Kernel 3a: Register tiling](./kernels/2)
Each thread now computes a $TM \times TN$ subtile of output instead of a single element. Increases FMA per read. Fully unrolled accumulator array keeps everything in registers rather than local memory.

### [Kernel 3b: Vectorised Dequant](./kernels/3)
Same tiling as 3a, but the scalar unpack path is replaced with `lop3` (unpacking two weights per instruction) and the MAC loop with `hfma2` on packed `half2` register (two FMAs per instruction). Stores partials in fp16 which are then written to a fp32 accumulator to reduce aggregate rounding error.

## Resources
- [Marlin paper](https://arxiv.org/abs/2408.11743)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [Siboehm's matmul worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Aleksa Gordić's anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
- [Wafer's GPU perf engineering resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)
- [Pranjal Shankhdhar's Outperforming cuBLAS on H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [The Software Frontier's Mastering CUDA and HPC Series](https://www.thesoftwarefrontier.com/p/mastering-cuda-and-high-performance)
- [MLC's Modern GPU Programming For MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/)