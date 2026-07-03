#pragma once
#ifndef GUPPY_COMMON_FIXTURE_LOADER_HPP_
#define GUPPY_COMMON_FIXTURE_LOADER_HPP_

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "third_party/nlohmann/json.hpp"

struct FixtureMeta {
    int M = 0;
    int N = 0;
    int K = 0;
    int group_size = 0;
    int num_bits = 0;
    int seed = 0;
    int num_words = 0;
    int num_groups = 0;
};

struct Fixture {
    FixtureMeta meta;
    std::vector<uint16_t> activations;
    std::vector<uint32_t> packed_qweight;
    std::vector<float> scales;
    std::vector<uint16_t> output;
};

namespace fixture_loader_detail {

inline std::string read_file_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <typename T>
std::vector<T> read_binary(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("failed to open: " + path);
    }
    const std::streamsize actual_bytes = f.tellg();
    const std::streamsize expected_bytes = static_cast<std::streamsize>(count * sizeof(T));
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error(path + ": size mismatch, file has " +
                                 std::to_string(actual_bytes) + " bytes, expected " +
                                 std::to_string(expected_bytes) + " (" + std::to_string(count) +
                                 " elements * " + std::to_string(sizeof(T)) + " bytes)");
    }
    f.seekg(0);
    std::vector<T> data(count);
    f.read(reinterpret_cast<char*>(data.data()), expected_bytes);
    if (!f) {
        throw std::runtime_error(path + ": read failed after size check passed");
    }
    return data;
}

}  // namespace fixture_loader_detail

inline Fixture load_fixture(const std::string& dir) {
    using nlohmann::json;
    using namespace fixture_loader_detail;

    const std::string sep = "/";
    json j = json::parse(read_file_text(dir + sep + "meta.json"));

    Fixture fx;
    fx.meta.M = j.at("M").get<int>();
    fx.meta.N = j.at("N").get<int>();
    fx.meta.K = j.at("K").get<int>();
    fx.meta.group_size = j.at("group_size").get<int>();
    fx.meta.num_bits = j.at("num_bits").get<int>();
    fx.meta.seed = j.at("seed").get<int>();
    fx.meta.num_words = j.at("num_words").get<int>();
    fx.meta.num_groups = j.at("num_groups").get<int>();

    const size_t M = fx.meta.M, N = fx.meta.N, K = fx.meta.K;
    const size_t num_words = fx.meta.num_words, num_groups = fx.meta.num_groups;

    fx.activations = read_binary<uint16_t>(dir + sep + "activations.bin", M * K);
    fx.packed_qweight = read_binary<uint32_t>(dir + sep + "packed_qweight.bin", num_words * N);
    fx.scales = read_binary<float>(dir + sep + "scales.bin", num_groups * N);
    fx.output = read_binary<uint16_t>(dir + sep + "output.bin", M * N);

    return fx;
}

#endif  // GUPPY_COMMON_FIXTURE_LOADER_HPP_