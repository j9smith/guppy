#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../common/benchmark_harness.hpp"
#include "../common/fixture_loader.hpp"
#include "../common/third_party/nlohmann/json.hpp"

#include "../kernels/0/kernel.cu"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    const std::string fixtures_dir = (argc > 1) ? argv[1] : "fixtures";

    if (!fs::exists(fixtures_dir) || !fs::is_directory(fixtures_dir)) {
        fprintf(stderr, "error: fixtures directory not found: %s\n", fixtures_dir.c_str());
        return 1;
    }

    std::vector<BenchmarkResult> results;

    for (const auto& entry : fs::directory_iterator(fixtures_dir)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        const std::string dir = entry.path().string();

        printf("running %-32s ... ", name.c_str());
        fflush(stdout);

        Fixture fx;
        try {
            fx = load_fixture(dir);
        } catch (const std::exception& e) {
            printf("SKIP (load error: %s)\n", e.what());
            continue;
        }

        const BenchmarkResult r = run_benchmark(name, fx);
        results.push_back(r);

        printf("%-4s  median=%8.4f ms  %9.2f GFLOPS  %8.2f GB/s  "
               "(max_abs_err=%.4g max_rel_err=%.4g)\n",
               r.correctness_passed ? "PASS" : "FAIL", r.median_time_ms, r.gflops,
               r.bandwidth_gb_s, r.max_abs_error, r.max_rel_error);
    }

    if (results.empty()) {
        fprintf(stderr, "warning: no fixtures found in %s\n", fixtures_dir.c_str());
    }

    nlohmann::json j = nlohmann::json::array();
    for (const auto& r : results) {
        j.push_back({
            {"fixture_name", r.fixture_name},
            {"M", r.M},
            {"N", r.N},
            {"K", r.K},
            {"group_size", r.group_size},
            {"correctness_passed", r.correctness_passed},
            {"max_abs_error", r.max_abs_error},
            {"max_rel_error", r.max_rel_error},
            {"median_time_ms", r.median_time_ms},
            {"gflops", r.gflops},
            {"bandwidth_gb_s", r.bandwidth_gb_s},
        });
    }

    std::ofstream out("results.json");
    out << j.dump(2);
    printf("\nwrote results.json (%zu results)\n", results.size());

    return 0;
}
