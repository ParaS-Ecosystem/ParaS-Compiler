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

#ifndef __PARAS_OFFLOAD_ARCH_CHECK_HPP__
#define __PARAS_OFFLOAD_ARCH_CHECK_HPP__

#include <cstdlib>
#include <string>
#include <vector>

#include "kem_gpu/gpu_utilities.hpp"
#include "sycl/exception.hpp"

namespace paras_detail {

inline std::string base_gfx_name(const std::string& s) {
    const auto p = s.find(':');
    return p == std::string::npos ? s : s.substr(0, p);
}

inline int sm_number(const std::string& s) {
    if (s.rfind("sm_", 0) != 0)
        return -1;
    int v = 0, digits = 0;
    for (std::size_t i = 3; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i, ++digits)
        v = v * 10 + (s[i] - '0');
    return digits ? v : -1;
}

inline std::vector<std::string> split_arch_list(const std::string& s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t end = s.find(',', start);
        const std::string item =
            s.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty())
            out.push_back(item);
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    return out;
}

inline bool arch_check_disabled() {
    const char* e = std::getenv("PARAS_SKIP_ARCH_CHECK");
    return e && e[0] == '1';
}

[[noreturn]] inline void throw_arch_mismatch(const std::string& built, const std::string& device,
                                             const std::string& name, const std::string& backend) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::kernel_not_supported),
                          "ParaS: kernels were compiled for " + built + " but the device '" + name +
                              "' is " + device + ". Rebuild with: -parasdevice " + backend + ":" +
                              device + "  (set PARAS_SKIP_ARCH_CHECK=1 to skip this check)");
}

#if (PARAS_HIP_BACKEND)
inline void check_offload_arch_hip(int device_id) {
#ifdef PARAS_OFFLOAD_ARCH
    if (arch_check_disabled())
        return;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_id) != hipSuccess)
        return;
    const std::string dev = base_gfx_name(prop.gcnArchName);
    if (dev.empty())
        return;
    for (const std::string& target : split_arch_list(PARAS_OFFLOAD_ARCH))
        if (base_gfx_name(target) == dev)
            return;
    throw_arch_mismatch(PARAS_OFFLOAD_ARCH, dev, prop.name, "hip");
#else
    (void)device_id;
#endif
}
#endif

#if (PARAS_CUDA_BACKEND)
inline void check_offload_arch_cuda(int device_id) {
#ifdef PARAS_OFFLOAD_ARCH
    if (arch_check_disabled())
        return;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
        return;
    const int dev = prop.major * 10 + prop.minor;
    int built = -1;
    for (const std::string& target : split_arch_list(PARAS_OFFLOAD_ARCH)) {
        const int sm = sm_number(target);
        if (sm > 0 && (built < 0 || sm < built))
            built = sm;
    }
    if (built > 0 && dev < built)
        throw_arch_mismatch(PARAS_OFFLOAD_ARCH, "sm_" + std::to_string(dev), prop.name, "cuda");
#else
    (void)device_id;
#endif
}
#endif

} // namespace paras_detail

#endif
