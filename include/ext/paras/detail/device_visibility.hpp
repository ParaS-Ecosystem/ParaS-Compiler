/**
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 *
 * This file is part of the ParaS Compiler, a component of the ParaS Ecosystem.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __SYCL_EXT_PARAS_DEVICE_VISIBILITY_HPP__
#define __SYCL_EXT_PARAS_DEVICE_VISIBILITY_HPP__

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "sycl/device.hpp"
#include "utilities/internal_utils.hpp"

namespace sycl {
namespace ext {
namespace paras {
namespace detail {

inline std::string trim(std::string s) {
    ::paras_extension::trim_inplace(s);
    return s;
}

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline bool is_uuid_token(const std::string& t) { return t.rfind("GPU-", 0) == 0 || t.rfind("MIG-", 0) == 0; }

inline std::vector<std::string> command_lines(const char* cmd) {
    std::vector<std::string> out;
    ::paras_extension::run_cmd_lines(cmd, out);
    return out;
}

inline std::vector<std::string> tokens_of(const char* value) {
    std::vector<std::string> tokens;
    std::string v(value);
    std::size_t start = 0;
    while (start <= v.size()) {
        const std::size_t end = v.find(',', start);
        std::string t = trim(v.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!t.empty()) tokens.push_back(t);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return tokens;
}

inline bool has_uuid_tokens(const char* env_name) {
    const char* env = std::getenv(env_name);
    if (env == nullptr) return false;
    const auto tokens = tokens_of(env);
    return !tokens.empty() && std::all_of(tokens.begin(), tokens.end(), is_uuid_token);
}

inline std::vector<std::size_t> visible_indices(const char* env_name, std::size_t count,
                                                const std::vector<std::string>& uuids) {
    std::vector<std::size_t> indices;
    const char* env = std::getenv(env_name);
    if (env == nullptr) {
        for (std::size_t i = 0; i < count; ++i) indices.push_back(i);
        return indices;
    }
    const auto tokens = tokens_of(env);
    const bool known = uuids.size() == count && count > 0;
    if (!tokens.empty() && !known && std::all_of(tokens.begin(), tokens.end(), is_uuid_token)) {
        for (std::size_t i = 0; i < tokens.size() && i < count; ++i) indices.push_back(i);
        return indices;
    }
    for (const std::string& token : tokens) {
        long id = -1;
        if (is_uuid_token(token)) {
            if (known) {
                const std::string want = lower(token);
                long found = -1;
                bool unique = true;
                for (std::size_t i = 0; i < uuids.size(); ++i) {
                    if (lower(uuids[i]).rfind(want, 0) != 0) continue;
                    if (found >= 0) unique = false;
                    found = static_cast<long>(i);
                }
                if (unique) id = found;
            }
        } else {
            char* end = nullptr;
            const long v = std::strtol(token.c_str(), &end, 10);
            if (end != token.c_str() && *end == '\0') id = v;
        }
        if (id >= 0 && static_cast<std::size_t>(id) < count) {
            const std::size_t index = static_cast<std::size_t>(id);
            if (std::find(indices.begin(), indices.end(), index) == indices.end()) indices.push_back(index);
        }
    }
    return indices;
}

inline std::vector<std::string> nvidia_uuids() {
    std::vector<std::string> uuids;
    for (const std::string& line : command_lines("nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null")) {
        const std::string t = trim(line);
        if (!t.empty()) uuids.push_back(t);
    }
    return uuids;
}

inline std::vector<std::size_t> visible_nvidia_indices(std::size_t count) {
    return visible_indices("CUDA_VISIBLE_DEVICES", count, nvidia_uuids());
}

inline std::vector<std::string> amd_uuids() {
    std::vector<std::string> uuids;
    for (const std::string& line : command_lines("rocm-smi --showuniqueid 2>/dev/null")) {
        const auto p = line.find("GPU[");
        const auto q = line.find("Unique ID:");
        if (p == std::string::npos || q == std::string::npos) continue;
        const std::size_t idx = static_cast<std::size_t>(std::atoi(line.c_str() + p + 4));
        std::string id = trim(line.substr(q + std::string("Unique ID:").size()));
        if (id.rfind("0x", 0) == 0 || id.rfind("0X", 0) == 0) id = id.substr(2);
        if (id.empty() || id.size() > 16 ||
            !std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isxdigit(c) != 0; }))
            continue;
        if (uuids.size() <= idx) uuids.resize(idx + 1);
        uuids[idx] = "GPU-" + std::string(16 - id.size(), '0') + lower(id);
    }
    if (std::any_of(uuids.begin(), uuids.end(), [](const std::string& u) { return u.empty(); })) uuids.clear();
    return uuids;
}

inline std::vector<sycl::device> visible_amd_gpus(const std::vector<sycl::device>& all) {
    const char* rocr_env = std::getenv("ROCR_VISIBLE_DEVICES");
    const char* hip_env = std::getenv("HIP_VISIBLE_DEVICES");
    const char* cuda_env = std::getenv("CUDA_VISIBLE_DEVICES");
    if (rocr_env == nullptr && hip_env == nullptr && cuda_env == nullptr) return all;
    const auto uuids = amd_uuids();
    const auto rocr = visible_indices("ROCR_VISIBLE_DEVICES", all.size(), uuids);
    std::vector<std::string> rocr_uuids;
    if (uuids.size() == all.size())
        for (const std::size_t i : rocr) rocr_uuids.push_back(uuids[i]);
    const auto hip = visible_indices(hip_env != nullptr ? "HIP_VISIBLE_DEVICES" : "CUDA_VISIBLE_DEVICES", rocr.size(), rocr_uuids);
    std::vector<sycl::device> out;
    for (std::size_t logical = 0; logical < hip.size(); ++logical) {
        sycl::device d = all[rocr[hip[logical]]];
        d.paras_set_native_id(static_cast<int>(logical));
        out.push_back(d);
    }
    return out;
}

}
}
}
}

#endif
