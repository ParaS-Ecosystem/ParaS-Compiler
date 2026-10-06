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

#ifndef __PARAS_WORK_GROUP_ALGORITHMS_DETAIL_HPP__
#define __PARAS_WORK_GROUP_ALGORITHMS_DETAIL_HPP__

#include <cstddef>
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {
namespace detail {

constexpr std::size_t kMaxWorkGroupSize = 1024;

PARAS_KERNEL_HD inline std::size_t wg_local_linear_id() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
#else
    return paras_host_detail::tl_local_linear_id;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T* wg_scratch() {
    static_assert(sizeof(T) * kMaxWorkGroupSize <= 32768,
                  "group algorithm value type too large for work-group scratch");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    alignas(T) __shared__ unsigned char buf[sizeof(T) * kMaxWorkGroupSize];
    return reinterpret_cast<T*>(buf);
#else
    return reinterpret_cast<T*>(paras_host_detail::tl_group_scratch);
#endif
}

template <typename Group, typename T, typename Op>
PARAS_KERNEL_HD T wg_reduce(const Group& g, T x, Op op) {
    T* s = wg_scratch<T>();
    const std::size_t lid = wg_local_linear_id();
    const std::size_t n = g.get_local_linear_range();
    s[lid] = x;
    paras_syncthreads();
    for (std::size_t stride = 1; stride < n; stride <<= 1) {
        if (lid % (2 * stride) == 0 && lid + stride < n)
            s[lid] = op(s[lid], s[lid + stride]);
        paras_syncthreads();
    }
    const T r = s[0];
    paras_syncthreads();
    return r;
}

template <typename Group, typename T, typename Op>
PARAS_KERNEL_HD T wg_inclusive_scan(const Group& g, T x, Op op) {
    T* s = wg_scratch<T>();
    const std::size_t lid = wg_local_linear_id();
    const std::size_t n = g.get_local_linear_range();
    s[lid] = x;
    paras_syncthreads();
    for (std::size_t off = 1; off < n; off <<= 1) {
        const T v = (lid >= off) ? op(s[lid - off], s[lid]) : s[lid];
        paras_syncthreads();
        s[lid] = v;
        paras_syncthreads();
    }
    const T r = s[lid];
    paras_syncthreads();
    return r;
}

template <typename Group, typename T, typename Op>
PARAS_KERNEL_HD T wg_exclusive_scan(const Group& g, T x, T init, Op op) {
    T* s = wg_scratch<T>();
    const std::size_t lid = wg_local_linear_id();
    (void)wg_inclusive_scan(g, x, op);
    const T r = (lid == 0) ? init : op(init, s[lid - 1]);
    paras_syncthreads();
    return r;
}

template <typename Group, typename T>
PARAS_KERNEL_HD T wg_broadcast(const Group&, T x, std::size_t src) {
    T* s = wg_scratch<T>();
    if (wg_local_linear_id() == src)
        s[0] = x;
    paras_syncthreads();
    const T r = s[0];
    paras_syncthreads();
    return r;
}

template <typename Group, typename Id>
PARAS_KERNEL_HD std::size_t wg_linearize(const Group& g, const Id& id) {
    std::size_t lin = 0;
    for (int d = 0; d < Group::dimensions; ++d)
        lin = lin * g.get_local_range(d) + id[d];
    return lin;
}

struct wg_or {
    PARAS_KERNEL_HD int operator()(int a, int b) const { return a || b; }
};
struct wg_and {
    PARAS_KERNEL_HD int operator()(int a, int b) const { return a && b; }
};

template <typename Group>
PARAS_KERNEL_HD bool wg_any(const Group& g, bool p) {
    return wg_reduce(g, int(p), wg_or{}) != 0;
}
template <typename Group>
PARAS_KERNEL_HD bool wg_all(const Group& g, bool p) {
    return wg_reduce(g, int(p), wg_and{}) != 0;
}

} // namespace detail
} // namespace sycl

#endif
