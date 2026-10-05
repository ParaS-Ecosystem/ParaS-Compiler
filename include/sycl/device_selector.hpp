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

#ifndef __PARAS_DEVICE_SELECTOR_HPP__
#define __PARAS_DEVICE_SELECTOR_HPP__

#include "device.hpp"
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>
#include <iostream>
#include <cstdlib>

#ifndef PARASDEVICE
#define PARASDEVICE 0
#endif

namespace sycl {

template <typename F>
class device_selector {
public:
    constexpr device_selector(F func) : select_(func) {}
    int operator()(const device& dev) const { return select_(dev); }

private:
    F select_;
};

inline constexpr auto cpu_selector_v =
    device_selector{[](const device& dev) -> int { return dev.is_cpu() ? 1 : -1; }};

inline constexpr auto gpu_selector_v =
    device_selector{[](const device& dev) -> int { return dev.is_gpu() ? 1 : -1; }};

inline constexpr auto accelerator_selector_v =
    device_selector{[](const device& dev) -> int { return dev.is_accelerator() ? 1 : -1; }};

inline constexpr auto default_selector_v = device_selector{[](const device& dev) -> int {
#if PARASDEVICE
    return dev.is_gpu()   ? 3
           : dev.is_cpu() ? 2
           :
#else
    return dev.is_cpu()   ? 3
           : dev.is_gpu() ? 2
           :
#endif
           dev.is_accelerator() ? 1
                                : -1;
}};

inline auto aspect_selector(const std::vector<aspect>& aspectList,
                            const std::vector<aspect>& denyList = {}) {
    return device_selector{[aspectList, denyList](const device& dev) -> int {
        for (const auto& a : aspectList) {
            if (!dev.has(a))
                return -1;
        }
        for (const auto& a : denyList) {
            if (dev.has(a))
                return -1;
        }
        return 1;
    }};
}

template <aspect... AspectList>
inline auto aspect_selector(const std::vector<aspect>& denyList = {},
                            const std::vector<aspect>& effortList = {}) {
    return device_selector{[denyList, effortList](const device& dev) -> int {
        bool has_all_required = (dev.has(AspectList) && ...);
        if (!has_all_required)
            return -1;
        for (const auto& a : denyList) {
            if (dev.has(a))
                return -1;
        }
        int score = 1;
        for (const auto& a : effortList) {
            if (dev.has(a))
                score++;
        }
        return score;
    }};
}

template <typename Aspect0, typename... Aspects>
inline auto aspect_selector(Aspect0 first, Aspects... rest) {
    static_assert(std::is_same_v<Aspect0, aspect> && (std::is_same_v<Aspects, aspect> && ...),
                  "aspect_selector(Aspects...) requires every argument to be a sycl::aspect value");
    return device_selector{[first, rest...](const device& dev) -> int {
        bool has_all_required = dev.has(first) && (dev.has(rest) && ...);
        return has_all_required ? 1 : -1;
    }};
}

} // namespace sycl

#endif
