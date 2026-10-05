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

#ifndef __PARAS_SYCL_USM_ALLOC_HPP__
#define __PARAS_SYCL_USM_ALLOC_HPP__

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>

namespace sycl {
namespace usm {

enum class alloc { host, device, shared, unknown };

} // namespace usm

namespace paras_usm_detail {
struct host_alloc_record {
    std::size_t bytes;
    usm::alloc kind;
};
inline std::mutex& host_registry_mutex() {
    static std::mutex m;
    return m;
}
inline std::map<std::uintptr_t, host_alloc_record>& host_registry() {
    static std::map<std::uintptr_t, host_alloc_record> r;
    return r;
}
inline void register_host_alloc(void* p, std::size_t bytes, usm::alloc kind) {
    std::lock_guard<std::mutex> lock(host_registry_mutex());
    host_registry()[reinterpret_cast<std::uintptr_t>(p)] = {bytes, kind};
}
inline void unregister_host_alloc(void* p) {
    std::lock_guard<std::mutex> lock(host_registry_mutex());
    host_registry().erase(reinterpret_cast<std::uintptr_t>(p));
}
inline usm::alloc lookup_host_alloc(const void* p) {
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    std::lock_guard<std::mutex> lock(host_registry_mutex());
    auto& reg = host_registry();
    auto it = reg.upper_bound(addr);
    if (it == reg.begin())
        return usm::alloc::unknown;
    --it;
    return addr < it->first + it->second.bytes ? it->second.kind : usm::alloc::unknown;
}
} // namespace paras_usm_detail
} // namespace sycl

#endif // SYCL_USM_ALLOC_HPP
