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

#ifndef PARAS_GET_POINTER_TYPE_HPP
#define PARAS_GET_POINTER_TYPE_HPP

#include "usm_alloc.hpp"
#include "context.hpp"
#include "device.hpp"
#include "exception.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

namespace paras_usm_detail {
inline usm::alloc query(const void* ptr, int* native_device) {
    *native_device = -1;
    if (ptr == nullptr) {
        return usm::alloc::unknown;
    }
#if PARAS_CUDA_BACKEND
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        (void)cudaGetLastError();
        return lookup_host_alloc(ptr);
    }
    *native_device = attr.device;
    switch (attr.type) {
    case cudaMemoryTypeHost:
        return usm::alloc::host;
    case cudaMemoryTypeDevice:
        return usm::alloc::device;
    case cudaMemoryTypeManaged:
        return usm::alloc::shared;
    default:
        return lookup_host_alloc(ptr);
    }
#elif PARAS_HIP_BACKEND
    hipPointerAttribute_t attr{};
    if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
        (void)hipGetLastError();
        return lookup_host_alloc(ptr);
    }
    *native_device = attr.device;
#if HIP_VERSION >= 50000000
    const hipMemoryType type = attr.type;
#else
    const hipMemoryType type = attr.memoryType;
#endif
    switch (type) {
    case hipMemoryTypeHost:
        return usm::alloc::host;
    case hipMemoryTypeDevice:
        return usm::alloc::device;
    case hipMemoryTypeManaged:
    case hipMemoryTypeUnified:
        return usm::alloc::shared;
    default:
        return lookup_host_alloc(ptr);
    }
#else
    return lookup_host_alloc(ptr);
#endif
}
} // namespace paras_usm_detail

namespace usm {

inline alloc get_pointer_type(const void* ptr, const sycl::context& syclContext) {
    (void)syclContext;
    int native_device = -1;
    return paras_usm_detail::query(ptr, &native_device);
}

} // namespace usm

inline usm::alloc get_pointer_type(const void* ptr, const sycl::context& syclContext) {
    return usm::get_pointer_type(ptr, syclContext);
}

inline device get_pointer_device(const void* ptr, const sycl::context& syclContext) {
    int native_device = -1;
    const usm::alloc kind = paras_usm_detail::query(ptr, &native_device);
    const std::vector<device> devs = syclContext.get_devices();
    if (kind == usm::alloc::device || kind == usm::alloc::shared) {
        for (const device& d : devs) {
            if (d.is_gpu() && d.get_native_id() == native_device) {
                return d;
            }
        }
    }
    if (kind != usm::alloc::unknown && !devs.empty()) {
        return devs.front();
    }
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "get_pointer_device: pointer is not a USM allocation of this context");
}

} // namespace sycl

#endif
