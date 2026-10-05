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

#ifndef __PARAS_SPEC_CONSTANT_HPP__
#define __PARAS_SPEC_CONSTANT_HPP__

#include <type_traits>
#include <utility>
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename T, typename IDType = T>
class specialization_id {
public:
    using value_type = T;

    template <typename... Args>
    explicit constexpr specialization_id(Args&&... args) : value_(std::forward<Args>(args)...) {}

    specialization_id(const specialization_id&) = delete;
    specialization_id& operator=(const specialization_id&) = delete;

    T value_;
};

namespace paras_spec_detail {
template <auto& SpecName>
using value_t = typename std::remove_reference_t<decltype(SpecName)>::value_type;

template <auto& SpecName>
inline value_t<SpecName> host_value = SpecName.value_;

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
template <auto& SpecName>
__device__ value_t<SpecName> device_value = SpecName.value_;
#endif

template <auto& SpecName>
void set(const value_t<SpecName>& v) {
    host_value<SpecName> = v;
#if PARAS_CUDA_BACKEND
    (void)cudaMemcpyToSymbol(device_value<SpecName>, &v, sizeof(v));
#elif PARAS_HIP_BACKEND
    (void)hipMemcpyToSymbol(HIP_SYMBOL(device_value<SpecName>), &v, sizeof(v));
#endif
}
} // namespace paras_spec_detail

class kernel_handler {
public:
    template <auto& SpecName>
    PARAS_KERNEL_HD paras_spec_detail::value_t<SpecName> get_specialization_constant() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return paras_spec_detail::device_value<SpecName>;
#else
        return paras_spec_detail::host_value<SpecName>;
#endif
    }
};

} // namespace sycl

#endif
