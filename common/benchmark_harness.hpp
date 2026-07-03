#pragma once
#ifndef GUPPY_COMMON_BENCHMARK_HARNESS_HPP_
#define GUPPY_COMMON_BENCHMARK_HARNESS_HPP_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "fixture_loader.hpp"
#include "kernel_interface.hpp"
#include "utils.cuh"

struct BenchmarkResult {
    std::string fixture_name;
    int M = 0, N = 0, K = 0, group_size = 0;
    bool correctness_passed = false;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double median_time_ms = 0.0;
    double gflops = 0.0;
    double bandwidth_gb_s = 0.0;
};

namespace benchmark_detail {

inline double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace benchmark_detail

inline BenchmarkResult run_benchmark(const std::string& fixture_name, const Fixture& fx,
                                     int warmup_runs = 3, int timed_runs = 20,
                                     double abs_tol = 1e-2, double rel_tol = 5e-2) {
    BenchmarkResult result;
    result.fixture_name = fixture_name;
    result.M = fx.meta.M;
    result.N = fx.meta.N;
    result.K = fx.meta.K;
    result.group_size = fx.meta.group_size;

    const size_t M = fx.meta.M, N = fx.meta.N;

    const size_t activations_bytes = fx.activations.size() * sizeof(uint16_t);
    const size_t packed_qweight_bytes = fx.packed_qweight.size() * sizeof(uint32_t);
    const size_t scales_bytes = fx.scales.size() * sizeof(float);
    const size_t output_bytes = fx.output.size() * sizeof(uint16_t);

    half* d_activations = nullptr;
    uint32_t* d_packed_qweight = nullptr;
    float* d_scales = nullptr;
    half* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_activations, activations_bytes));
    CUDA_CHECK(cudaMalloc(&d_packed_qweight, packed_qweight_bytes));
    CUDA_CHECK(cudaMalloc(&d_scales, scales_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    CUDA_CHECK(cudaMemcpy(d_activations, fx.activations.data(), activations_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_packed_qweight, fx.packed_qweight.data(), packed_qweight_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, fx.scales.data(), scales_bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup_runs; ++i) {
        launch_w4a16_matmul(d_activations, d_packed_qweight, d_scales, d_output, fx.meta.M,
                            fx.meta.N, fx.meta.K, fx.meta.group_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times_ms;
    times_ms.reserve(timed_runs);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < timed_runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_w4a16_matmul(d_activations, d_packed_qweight, d_scales, d_output, fx.meta.M,
                            fx.meta.N, fx.meta.K, fx.meta.group_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    result.median_time_ms = benchmark_detail::median(times_ms);

    std::vector<uint16_t> host_output(M * N);
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    double max_abs = 0.0, max_rel = 0.0;
    bool passed = true;
    for (size_t i = 0; i < M * N; ++i) {
        const half got_h = *reinterpret_cast<const half*>(&host_output[i]);
        const half want_h = *reinterpret_cast<const half*>(&fx.output[i]);
        const double got = static_cast<double>(__half2float(got_h));
        const double want = static_cast<double>(__half2float(want_h));

        const double abs_err = std::abs(got - want);
        const double rel_err = abs_err / (std::abs(want) + 1e-6);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
        if (abs_err > abs_tol && rel_err > rel_tol) passed = false;
    }
    result.max_abs_error = max_abs;
    result.max_rel_error = max_rel;
    result.correctness_passed = passed;

    // Throughput metrics.
    const double seconds = result.median_time_ms / 1000.0;
    const double flops = 2.0 * M * N * fx.meta.K;  // one multiply + one add per MAC
    result.gflops = (seconds > 0.0) ? flops / seconds / 1e9 : 0.0;

    const double total_bytes =
        static_cast<double>(activations_bytes + packed_qweight_bytes + scales_bytes + output_bytes);
    result.bandwidth_gb_s = (seconds > 0.0) ? total_bytes / seconds / 1e9 : 0.0;

    CUDA_CHECK(cudaFree(d_activations));
    CUDA_CHECK(cudaFree(d_packed_qweight));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

#endif  // GUPPY_COMMON_BENCHMARK_HARNESS_HPP_