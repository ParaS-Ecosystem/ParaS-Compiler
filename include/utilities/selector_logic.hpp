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

#ifndef __PARAS_SELECTOR_LOGIC_HPP__
#define __PARAS_SELECTOR_LOGIC_HPP__

#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "utilities/internal_utils.hpp"
#include <limits>
#include <type_traits>
#include <vector>

#ifndef PARASDEVICE
#define PARASDEVICE 0
#endif

namespace paras_extension {

template <typename SelectorV>
static inline std::vector<sycl::device>
selector_domain_devices(const SelectorV &) {
  return sycl::device::get_devices(sycl::info::device_type::all);
}

template <>
inline std::vector<sycl::device>
selector_domain_devices(const decltype(sycl::cpu_selector_v) &) {
  return sycl::device::get_devices(sycl::info::device_type::cpu);
}

template <>
inline std::vector<sycl::device>
selector_domain_devices(const decltype(sycl::gpu_selector_v) &) {
  return sycl::device::get_devices(sycl::info::device_type::gpu);
}

template <>
inline std::vector<sycl::device>
selector_domain_devices(const decltype(sycl::accelerator_selector_v) &) {
  return sycl::device::get_devices(sycl::info::device_type::accelerator);
}

template <typename SelectorV>
inline sycl::device select_device_with_selector(const SelectorV &selector_v) {
  constexpr bool has_flag = (PARASDEVICE != 0);

  auto candidates = selector_domain_devices(selector_v);

  int best_score = std::numeric_limits<int>::min();
  sycl::device *best = nullptr;

  for (auto &dev : candidates) {
    int score = selector_v(dev);
    if (score > best_score) {
      best_score = score;
      best = &dev;
    }
  }

  if constexpr (std::is_same_v<std::decay_t<SelectorV>,
                               std::decay_t<decltype(sycl::cpu_selector_v)>>) {
    if constexpr (has_flag) {
      paras_selector_error(
          "cpu_selector_v used but -parasdevice flag is present");
    }
    return best ? *best : sycl::device{};
  }

  if constexpr (std::is_same_v<std::decay_t<SelectorV>,
                               std::decay_t<decltype(sycl::gpu_selector_v)>>) {
    if constexpr (!has_flag) {
      paras_selector_error(
          "gpu_selector_v used but -parasdevice flag is NOT present");
    }
    if (!best) {
      return sycl::device(device_ctor_tag{}, "No GPU present", "", "", "", 0, 0,
                          0, sycl::info::local_mem_type::none, false, false,
                          false, -1, false);
    }
    return *best;
  }

  return best ? *best : sycl::device{};
}

inline sycl::device select_device_no_selector() {
#if PARASDEVICE
  auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (gpus.empty()) {
    return sycl::device(device_ctor_tag{}, "No GPU present", "", "", "", 0, 0,
                        0, sycl::info::local_mem_type::none, false, false,
                        false, -1, false);
  }
  return gpus.front();
#else
  return sycl::device{};
#endif
}

} // namespace paras_extension

#endif
