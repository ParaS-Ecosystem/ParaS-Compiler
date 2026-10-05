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

#ifndef __PARAS_ASPECT_HPP__
#define __PARAS_ASPECT_HPP__

namespace sycl {

enum class aspect : uint32_t {
    cpu,
    gpu,
    accelerator,
    custom,
    emulated,
    host_debuggable,
    fp16,
    fp64,
    atomic64,
    image,
    online_compiler,
    online_linker,
    queue_profiling,
    usm_device_allocations,
    usm_host_allocations,
    usm_atomic_host_allocations,
    usm_shared_allocations,
    usm_atomic_shared_allocations,
    usm_system_allocations
};

#ifndef PARASDEVICE
#define PARASDEVICE 0
#endif
template <aspect Aspect>
struct any_device_has
    : std::bool_constant<
          Aspect == aspect::cpu || Aspect == aspect::queue_profiling ||
          (PARASDEVICE != 0 && Aspect == aspect::gpu) || Aspect == aspect::usm_device_allocations ||
          Aspect == aspect::usm_host_allocations || Aspect == aspect::usm_shared_allocations> {};

template <aspect Aspect>
inline constexpr bool any_device_has_v = any_device_has<Aspect>::value;

template <aspect Aspect>
struct all_devices_have : std::false_type {};

template <aspect Aspect>
inline constexpr bool all_devices_have_v = all_devices_have<Aspect>::value;

} // namespace sycl

#endif
