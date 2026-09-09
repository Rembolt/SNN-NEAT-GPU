#pragma once

#include "util/Log.h"

#include "HostMouseInput.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

class App;

namespace Util {

// Float slots for HostMouseInput / HostFeedbackInput packed contiguously into d_input

template <typename T>
constexpr std::size_t hostInputStructFloatSlotCount()
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "Host input structs must be trivially copyable for memcpy-style float packing.");
    static_assert(sizeof(T) % sizeof(float) == 0u,
                  "Host input structs must use only float fields (no padding gaps).");
    return sizeof(T) / sizeof(float);
}

template <typename T>
std::vector<float> packHostInputStructToFloatVector(const T& value)
{
    constexpr std::size_t float_count = hostInputStructFloatSlotCount<T>();
    const auto* begin = reinterpret_cast<const float*>(&value);
    return std::vector<float>(begin, begin + float_count);
}

inline void atomicFileReplace(const std::filesystem::path& temp_path, const std::filesystem::path& path)
{
    std::error_code ec;
    std::filesystem::rename(temp_path, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(temp_path, path, ec);
        if (ec) {
            Errors::stop("Util: failed to replace " + path.string() + ": " + ec.message());
        }
    }
}

inline nlohmann::json JsonAtomicOpen(
    const std::filesystem::path& path,
    std::filesystem::path* resolved_path = nullptr)
{
    std::filesystem::path resolved = path;
    std::ifstream in(resolved);
    if (!in.is_open() && resolved.is_relative()) {
        resolved = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / resolved;
        in.clear();
        in.open(resolved);
    }
    if (!in.is_open()) {
        Errors::stop("Util: failed to open " + resolved.string());
    }
    if (resolved_path != nullptr) {
        *resolved_path = resolved;
    }
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return nlohmann::json::parse(content);
}

inline bool isJsonFileTruncated(const std::filesystem::path& path, std::uintmax_t min_size)
{
    std::filesystem::path resolved = path;
    std::ifstream in(resolved, std::ios::binary);
#ifdef SNN_GPU_PROJECT_ROOT
    if (!in.is_open() && resolved.is_relative()) {
        resolved = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / resolved;
        in.clear();
        in.open(resolved, std::ios::binary);
    }
#endif
    if (!in.is_open()) {
        return true;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(resolved, ec) || ec) {
        return true;
    }

    const std::uintmax_t file_size = std::filesystem::file_size(resolved, ec);
    if (ec || file_size < min_size) {
        return true;
    }

    constexpr std::size_t k_head_bytes = 64u;
    constexpr std::size_t k_tail_bytes = 64u;

    std::string head(std::min(static_cast<std::size_t>(file_size), k_head_bytes), '\0');
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
    if (!in) {
        return true;
    }
    head.resize(static_cast<std::size_t>(in.gcount()));
    if (head.find('\0') != std::string::npos) {
        return true;
    }
    if (head.find('{') == std::string::npos) {
        return true;
    }

    std::string tail;
    if (file_size <= k_head_bytes) {
        tail = head;
    } else {
        const std::size_t tail_read = std::min(k_tail_bytes, static_cast<std::size_t>(file_size));
        in.clear();
        in.seekg(static_cast<std::streamoff>(file_size - tail_read), std::ios::beg);
        tail.resize(tail_read);
        in.read(tail.data(), static_cast<std::streamsize>(tail.size()));
        if (!in) {
            return true;
        }
        tail.resize(static_cast<std::size_t>(in.gcount()));
    }
    return tail.rfind('}') == std::string::npos;
}

inline void JsonAtomicWrite(const std::filesystem::path& path, const nlohmann::json& json, int indent = 2)
{
    const std::filesystem::path temp_path = path.string() + ".tmp";
    std::ofstream out(temp_path, std::ios::trunc);
    if (!out.is_open()) {
        Errors::stop("Util: failed to open temp " + temp_path.string());
    }
    out << json.dump(indent);
    if (!out) {
        Errors::stop("Util: failed to write temp " + temp_path.string());
    }
    out.close();
    atomicFileReplace(temp_path, path);
}

inline bool isClipCurrupted(const std::filesystem::path& path)
{
    std::filesystem::path resolved = path;
    std::ifstream in(resolved, std::ios::binary);
#ifdef SNN_GPU_PROJECT_ROOT
    if (!in.is_open() && resolved.is_relative()) {
        resolved = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / resolved;
        in.clear();
        in.open(resolved, std::ios::binary);
    }
#endif
    if (!in.is_open()) {
        return true;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(resolved, ec) || ec) {
        return true;
    }
    if (resolved.extension() != ".miclip") {
        return true;
    }

    const std::uintmax_t file_size = std::filesystem::file_size(resolved, ec);
    constexpr std::size_t k_header_size =
        sizeof(std::uint32_t) + sizeof(std::uint32_t) + sizeof(std::uint32_t) + sizeof(std::uint64_t);
    if (ec || file_size < k_header_size) {
        return true;
    }

    std::uint32_t magic = 0u;
    std::uint32_t version = 0u;
    std::uint32_t struct_size = 0u;
    std::uint64_t tick_count = 0u;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&struct_size), sizeof(struct_size));
    in.read(reinterpret_cast<char*>(&tick_count), sizeof(tick_count));
    if (!in) {
        return true;
    }

    if (magic != 0x504C434Du) {
        return true;
    }
    if (version != 1u) {
        return true;
    }
    if (struct_size != sizeof(HostMouseInput)) {
        return true;
    }
    if (tick_count == 0u) {
        return true;
    }

    const std::size_t payload_size = static_cast<std::size_t>(tick_count) * sizeof(HostMouseInput);
    return file_size != k_header_size + payload_size;
}

inline bool isDNACurrupted(const std::filesystem::path& path)
{
    return isJsonFileTruncated(path, 64u);
}

inline bool isWorldCurrupted(const std::filesystem::path& path)
{
    return isJsonFileTruncated(path, 32u);
}

inline bool areInnovationIdsCurrupted(const std::filesystem::path& path)
{
    return isJsonFileTruncated(path, 16u);
}

namespace RecordClips {

void populateMouseInputClipFolder(
    ::App& app,
    unsigned long tick_count,
    std::chrono::milliseconds tick_target,
    int clicks_required,
    int target_clip_count,
    int record_attempts_per_clip,
    const std::function<void(const std::vector<HostMouseInput>&)>& save_clip);

}  // namespace RecordClips

}  // namespace Util
