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

static void paras_wait_all(const std::vector<sycl::event>& deps) {
    for (const sycl::event& e : deps) {
        e.wait();
    }
}

template <typename KernelName = void, typename Func>
sycl::event single_task(Func f) {
    return parallel_for<KernelName>(sycl::range<1>(1), [=](sycl::id<1>) { f(); });
}
template <typename KernelName = void, typename Func>
sycl::event single_task(sycl::event dep, Func f) {
    dep.wait();
    return single_task<KernelName>(f);
}
template <typename KernelName = void, typename Func>
sycl::event single_task(const std::vector<sycl::event>& deps, Func f) {
    paras_wait_all(deps);
    return single_task<KernelName>(f);
}

template <typename KernelName = void, typename Int, typename Func,
          std::enable_if_t<std::is_integral_v<Int>, int> = 0>
sycl::event parallel_for(Int numWorkItems, Func f) {
    return parallel_for<KernelName>(sycl::range<1>(static_cast<size_t>(numWorkItems)), f);
}

template <typename KernelName = void, typename Func, int dim>
sycl::event parallel_for(sycl::range<dim> r, sycl::event dep, Func f) {
    dep.wait();
    return parallel_for<KernelName>(r, f);
}
template <typename KernelName = void, typename Func, int dim>
sycl::event parallel_for(sycl::range<dim> r, const std::vector<sycl::event>& deps, Func f) {
    paras_wait_all(deps);
    return parallel_for<KernelName>(r, f);
}
template <typename KernelName = void, typename Func, int dim>
sycl::event parallel_for(const sycl::nd_range<dim>& r, sycl::event dep, Func f) {
    dep.wait();
    return parallel_for<KernelName>(r, f);
}
template <typename KernelName = void, typename Func, int dim>
sycl::event parallel_for(const sycl::nd_range<dim>& r, const std::vector<sycl::event>& deps,
                         Func f) {
    paras_wait_all(deps);
    return parallel_for<KernelName>(r, f);
}

sycl::event memcpy(void* dest, const void* src, size_t numBytes, sycl::event dep) {
    dep.wait();
    return memcpy(dest, src, numBytes);
}
sycl::event memcpy(void* dest, const void* src, size_t numBytes,
                   const std::vector<sycl::event>& deps) {
    paras_wait_all(deps);
    return memcpy(dest, src, numBytes);
}
sycl::event memset(void* ptr, int value, size_t numBytes, sycl::event dep) {
    dep.wait();
    return memset(ptr, value, numBytes);
}
sycl::event memset(void* ptr, int value, size_t numBytes, const std::vector<sycl::event>& deps) {
    paras_wait_all(deps);
    return memset(ptr, value, numBytes);
}

template <typename T>
sycl::event fill(void* ptr, const T& pattern, size_t count) {
    std::vector<T> staged(count, pattern);
    return memcpy(ptr, staged.data(), count * sizeof(T));
}
template <typename T>
sycl::event fill(void* ptr, const T& pattern, size_t count, sycl::event dep) {
    dep.wait();
    return fill(ptr, pattern, count);
}
template <typename T>
sycl::event fill(void* ptr, const T& pattern, size_t count, const std::vector<sycl::event>& deps) {
    paras_wait_all(deps);
    return fill(ptr, pattern, count);
}

template <typename T>
sycl::event copy(const T* src, T* dest, size_t count, sycl::event dep) {
    dep.wait();
    return memcpy(dest, src, count * sizeof(T));
}
template <typename T>
sycl::event copy(const T* src, T* dest, size_t count, const std::vector<sycl::event>& deps) {
    paras_wait_all(deps);
    return memcpy(dest, src, count * sizeof(T));
}

template <typename SrcT, int SrcDims, sycl::access_mode SrcMode, sycl::target SrcTgt,
          sycl::access::placeholder IsPlaceholder, typename DestT>
sycl::event copy(sycl::accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src,
                 std::shared_ptr<DestT> dest) {
    return submit([&](sycl::handler& h) { h.copy(src, dest); });
}
template <typename SrcT, typename DestT, int DestDims, sycl::access_mode DestMode,
          sycl::target DestTgt, sycl::access::placeholder IsPlaceholder>
sycl::event copy(std::shared_ptr<SrcT> src,
                 sycl::accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest) {
    return submit([&](sycl::handler& h) { h.copy(src, dest); });
}
template <typename SrcT, int SrcDims, sycl::access_mode SrcMode, sycl::target SrcTgt,
          sycl::access::placeholder IsPlaceholder, typename DestT>
sycl::event copy(sycl::accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src, DestT* dest) {
    return submit([&](sycl::handler& h) { h.copy(src, dest); });
}
template <typename SrcT, typename DestT, int DestDims, sycl::access_mode DestMode,
          sycl::target DestTgt, sycl::access::placeholder IsPlaceholder>
sycl::event copy(const SrcT* src,
                 sycl::accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest) {
    return submit([&](sycl::handler& h) { h.copy(src, dest); });
}
template <typename SrcT, int SrcDims, sycl::access_mode SrcMode, sycl::target SrcTgt,
          sycl::access::placeholder IsSrcPlaceholder, typename DestT, int DestDims,
          sycl::access_mode DestMode, sycl::target DestTgt,
          sycl::access::placeholder IsDestPlaceholder>
sycl::event copy(sycl::accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsSrcPlaceholder> src,
                 sycl::accessor<DestT, DestDims, DestMode, DestTgt, IsDestPlaceholder> dest) {
    return submit([&](sycl::handler& h) { h.copy(src, dest); });
}
template <typename T, int Dims, sycl::access_mode Mode, sycl::target Tgt,
          sycl::access::placeholder IsPlaceholder>
sycl::event fill(sycl::accessor<T, Dims, Mode, Tgt, IsPlaceholder> dest, const T& src) {
    return submit([&](sycl::handler& h) { h.fill(dest, src); });
}
template <typename T, int Dims, sycl::access_mode Mode, sycl::target Tgt,
          sycl::access::placeholder IsPlaceholder>
sycl::event update_host(sycl::accessor<T, Dims, Mode, Tgt, IsPlaceholder> acc) {
    return submit([&](sycl::handler& h) { h.update_host(acc); });
}
