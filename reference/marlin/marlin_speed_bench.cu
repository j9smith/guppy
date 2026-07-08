// Speed-only benchmark driver for Marlin, used purely as an external reference point on the
// GFLOPS-vs-stage chart. This intentionally does NOT verify correctness against your Python
// oracle and does NOT repack weights into Marlin's fragment layout: the kernel has no
// data-dependent control flow (every cp.async / ldsm4 / mma.sync / dequant() call runs the same
// number of times regardless of what bits are in B), so timing is identical whether B holds real
// repacked weights or garbage. Buffer *sizes* still have to be right, or you get an OOB crash.
//
// Build (adjust arch to your card):
//   nvcc -O3 -lineinfo -arch=sm_120 -std=c++17 -I ../../common \
//        -o marlin_speed_bench marlin_speed_bench.cu

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../../common/utils.cuh"  // CUDA_CHECK
#include "marlin.cuh"

namespace {

double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Shape {
    int M, N, K, group_size;  // group_size == -1 means per-column (ungrouped) scales
};

struct SpeedResult {
    Shape shape;
    bool ran = false;
    int marlin_status = 0;  // 0 == ok; ERR_PROB_SHAPE=1, ERR_KERN_SHAPE=2 from marlin.cuh
    double median_time_ms = 0.0;
    double gflops = 0.0;
};

// Marlin-only launch: opaque garbage-content buffers of the correct byte size, no fixture
// loading. `d_locks` is allocated once and grown/re-zeroed per shape rather than per call.
SpeedResult run_shape(const Shape& shape, int warmup_runs, int timed_runs) {
    SpeedResult result;
    result.shape = shape;

    const int M = shape.M, N = shape.N, K = shape.K, group_size = shape.group_size;

    if (N % 256 != 0 || K % 64 != 0 || (group_size != -1 && K % (group_size) != 0)) {
        std::fprintf(stderr, "  [skip] M=%d N=%d K=%d g=%d: shape not divisible by Marlin's tile sizes\n",
                    M, N, K, group_size);
        return result;
    }

    // A: MxK fp16
    const size_t a_bytes = static_cast<size_t>(M) * K * sizeof(half);
    // B: K*N 4-bit values, total bit count unchanged by Marlin's internal fragment permutation.
    const size_t b_bytes = static_cast<size_t>(K) * N / 2;
    // scales: fp16, (K/group_size) x N rows, or a single row of N when ungrouped.
    const size_t num_scale_rows = (group_size == -1) ? 1 : static_cast<size_t>(K) / group_size;
    const size_t s_bytes = num_scale_rows * N * sizeof(half);
    // C: MxN fp16
    const size_t c_bytes = static_cast<size_t>(M) * N * sizeof(half);
    // locks workspace: sized for max_par's default (16) even though we always pass max_par=16.
    const size_t locks_count = (static_cast<size_t>(N) / 128) * 16;

    void* d_A = nullptr;
    void* d_B = nullptr;
    void* d_s = nullptr;
    void* d_C = nullptr;
    int* d_locks = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_s, s_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_locks, locks_count * sizeof(int)));

    // Content is irrelevant for timing; zero it so ECC-scrubbing/NaN propagation can't skew
    // anything and so runs are deterministic byte-for-byte across shapes.
    CUDA_CHECK(cudaMemset(d_A, 0, a_bytes));
    CUDA_CHECK(cudaMemset(d_B, 0, b_bytes));
    CUDA_CHECK(cudaMemset(d_s, 0, s_bytes));
    CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));
    CUDA_CHECK(cudaMemset(d_locks, 0, locks_count * sizeof(int)));

    auto launch_once = [&]() {
        return marlin_cuda(d_A, d_B, d_C, d_s, M, N, K, d_locks, group_size);
    };

    int status = launch_once();
    if (status != 0) {
        std::fprintf(stderr, "  [skip] M=%d N=%d K=%d g=%d: marlin_cuda returned status %d "
                    "(1=ERR_PROB_SHAPE, 2=ERR_KERN_SHAPE)\n", M, N, K, group_size, status);
        result.marlin_status = status;
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_s));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_locks));
        return result;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmup_runs; ++i) {
        // Locks must be re-zeroed between launches: Marlin's barrier scheme leaves them
        // non-zero after a run (that's how cross-threadblock reduction is signaled).
        CUDA_CHECK(cudaMemset(d_locks, 0, locks_count * sizeof(int)));
        launch_once();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times_ms;
    times_ms.reserve(timed_runs);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < timed_runs; ++i) {
        CUDA_CHECK(cudaMemset(d_locks, 0, locks_count * sizeof(int)));
        CUDA_CHECK(cudaEventRecord(start));
        launch_once();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    result.ran = true;
    result.median_time_ms = median(times_ms);
    const double seconds = result.median_time_ms / 1000.0;
    const double flops = 2.0 * M * N * K;
    result.gflops = (seconds > 0.0) ? flops / seconds / 1e9 : 0.0;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_locks));

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    int warmup_runs = 5;
    int timed_runs = 30;

    // Shapes worth putting on the chart: a batch-size sweep at a representative hidden size,
    // plus a couple of larger square-ish shapes. Adjust freely -- N must be a multiple of 256,
    // K a multiple of 64 (and of group_size when grouped) for the tile configs Marlin ships.
    std::vector<Shape> shapes = {
        {1,    4096, 4096, 128},
        {16,   4096, 4096, 128},
        {64,   4096, 4096, 128},
        {256,  4096, 4096, 128},
        {1024, 4096, 4096, 128},
        {16,   4096, 4096, -1},
        {256,  4096, 4096, -1},
        {16,   4096, 11008, 128},
        {256,  4096, 11008, 128},
        {16,   8192, 8192, 128},
        {256,  8192, 8192, 128},
    };

    std::printf("%-6s %-6s %-6s %-6s %-12s %-10s %-8s\n",
               "M", "N", "K", "g", "time_ms", "GFLOPS", "status");
    std::printf("---------------------------------------------------------------\n");

    for (const auto& shape : shapes) {
        SpeedResult r = run_shape(shape, warmup_runs, timed_runs);
        if (!r.ran) {
            std::printf("%-6d %-6d %-6d %-6d %-12s %-10s %-8d\n",
                       shape.M, shape.N, shape.K, shape.group_size, "-", "-", r.marlin_status);
            continue;
        }
        std::printf("%-6d %-6d %-6d %-6d %-12.4f %-10.1f %-8s\n",
                   shape.M, shape.N, shape.K, shape.group_size,
                   r.median_time_ms, r.gflops, "ok");
    }

    return 0;
}
