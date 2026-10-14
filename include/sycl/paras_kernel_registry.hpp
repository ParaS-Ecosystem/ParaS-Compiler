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

#ifndef __PARAS_KERNEL_REGISTRY_HPP__
#define __PARAS_KERNEL_REGISTRY_HPP__

#include <cstddef>
#include <deque>
#include <algorithm>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "sycl/aspect.hpp"
#include "sycl/device.hpp"

namespace sycl {
namespace paras_kb {

template <sycl::aspect... As>
struct aspect_list {
    static std::vector<sycl::aspect> get() { return {As...}; }
};
template <std::size_t... Ns>
struct size_list {
    static std::vector<std::size_t> get() { return {Ns...}; }
};

struct kernel_record {
    const void* key;
    std::string name;
    std::vector<sycl::aspect> aspects;
    std::vector<std::size_t> reqd_work_group;
    std::vector<std::size_t> reqd_sub_group;
};

inline std::deque<kernel_record>& records() {
    static std::deque<kernel_record> all;
    return all;
}

template <typename T, typename = void>
struct is_complete : std::false_type {};
template <typename T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <typename T, typename = void>
struct device_has_of {
    static std::vector<sycl::aspect> get() { return {}; }
};
template <typename T>
struct device_has_of<T, std::void_t<typename T::paras_device_has>> {
    static std::vector<sycl::aspect> get() { return T::paras_device_has::get(); }
};
template <typename T, typename = void>
struct reqd_wg_of {
    static std::vector<std::size_t> get() { return {}; }
};
template <typename T>
struct reqd_wg_of<T, std::void_t<typename T::paras_reqd_work_group_size>> {
    static std::vector<std::size_t> get() { return T::paras_reqd_work_group_size::get(); }
};
template <typename T, typename = void>
struct reqd_sg_of {
    static std::vector<std::size_t> get() { return {}; }
};
template <typename T>
struct reqd_sg_of<T, std::void_t<typename T::paras_reqd_sub_group_size>> {
    static std::vector<std::size_t> get() { return T::paras_reqd_sub_group_size::get(); }
};

template <typename T>
inline void collect_attributes(kernel_record& rec) {
    if constexpr (!std::is_void_v<T> && std::is_class_v<T> && is_complete<T>::value) {
        auto add = [](auto& dst, const auto& src) {
            for (const auto& v : src)
                dst.push_back(v);
        };
        add(rec.aspects, device_has_of<T>::get());
        if (rec.reqd_work_group.empty())
            rec.reqd_work_group = reqd_wg_of<T>::get();
        if (rec.reqd_sub_group.empty())
            rec.reqd_sub_group = reqd_sg_of<T>::get();
    }
}

template <typename T>
struct type_key {
    static inline const char id = 0;
};

template <typename T>
inline std::string type_name() {
    std::string_view f = __PRETTY_FUNCTION__;
    const auto b = f.find("T = ");
    if (b == std::string_view::npos)
        return std::string(f);
    auto e = f.find_first_of(";]", b + 4);
    return std::string(f.substr(b + 4, e - (b + 4)));
}

template <typename KernelName, typename Func>
using key_type_t = std::conditional_t<std::is_void_v<KernelName>, Func, KernelName>;

template <typename KernelName, typename Func>
inline const kernel_record* add() {
    using K = key_type_t<KernelName, Func>;
    const void* key = &type_key<K>::id;
    for (auto& r : records())
        if (r.key == key) {
            kernel_record extra{key, {}, {}, {}, {}};
            collect_attributes<Func>(extra);
            for (auto a : extra.aspects)
                if (std::find(r.aspects.begin(), r.aspects.end(), a) == r.aspects.end())
                    r.aspects.push_back(a);
            if (r.reqd_work_group.empty())
                r.reqd_work_group = extra.reqd_work_group;
            if (r.reqd_sub_group.empty())
                r.reqd_sub_group = extra.reqd_sub_group;
            return &r;
        }
    kernel_record rec{key, type_name<K>(), {}, {}, {}};
    collect_attributes<KernelName>(rec);
    if constexpr (!std::is_same_v<KernelName, Func>)
        collect_attributes<Func>(rec);
    records().push_back(std::move(rec));
    return &records().back();
}

template <typename KernelName, typename Func>
struct registrar {
    static inline const kernel_record* const record = add<KernelName, Func>();
};

template <typename KernelName>
inline const kernel_record* find() {
    const void* key = &type_key<KernelName>::id;
    for (const auto& r : records())
        if (r.key == key)
            return &r;
    return nullptr;
}

inline bool meets_requirements(const kernel_record& rec, const sycl::device& dev) {
    for (sycl::aspect a : rec.aspects)
        if (!dev.has(a))
            return false;
    if (!rec.reqd_work_group.empty()) {
        std::size_t total = 1;
        for (std::size_t v : rec.reqd_work_group)
            total *= v;
        if (total > dev.get_info<sycl::info::device::max_work_group_size>())
            return false;
    }
    if (!rec.reqd_sub_group.empty()) {
        const auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        bool found = false;
        for (auto s : sizes)
            if (static_cast<std::size_t>(s) == rec.reqd_sub_group.front())
                found = true;
        if (!found)
            return false;
    }
    return true;
}

inline bool compatible(const kernel_record& rec, const sycl::device& dev) {
#if !PARASDEVICE
    if (dev.is_gpu())
        return false;
#endif
    for (sycl::aspect a : rec.aspects)
        if (!dev.has(a))
            return false;
    if (!rec.reqd_work_group.empty()) {
        std::size_t total = 1;
        for (std::size_t v : rec.reqd_work_group)
            total *= v;
        if (total > dev.get_info<sycl::info::device::max_work_group_size>())
            return false;
    }
    if (!rec.reqd_sub_group.empty()) {
        const auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        bool found = false;
        for (auto s : sizes)
            if (static_cast<std::size_t>(s) == rec.reqd_sub_group.front())
                found = true;
        if (!found)
            return false;
    }
    return true;
}

}
}

#define PARAS_REGISTER_KERNEL(KernelName, Func)                                                    \
    (void)::sycl::paras_kb::registrar<KernelName, Func>::record

#endif
