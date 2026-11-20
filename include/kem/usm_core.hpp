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

#ifndef __PARAS_USM_CORE_HPP__
#define __PARAS_USM_CORE_HPP__

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>

#include "sycl/context.hpp"
#include "sycl/device.hpp"
#include "sycl/exception.hpp"
#include "sycl/usm_alloc.hpp"

namespace sycl {
namespace detail {

struct usm_record {
  void *base;
  std::size_t bytes;
  usm::alloc kind;
  device dev;
};

inline std::mutex &usm_mutex() {
  static std::mutex m;
  return m;
}
inline std::map<std::uintptr_t, usm_record> &usm_records() {
  static std::map<std::uintptr_t, usm_record> r;
  return r;
}

inline bool usm_find(const void *ptr, std::uintptr_t &start, usm_record &rec) {
  if (ptr == nullptr)
    return false;
  const auto p = reinterpret_cast<std::uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(usm_mutex());
  auto &recs = usm_records();
  auto it = recs.upper_bound(p);
  if (it == recs.begin())
    return false;
  --it;
  if (p >= it->first + (it->second.bytes ? it->second.bytes : 1))
    return false;
  start = it->first;
  rec = it->second;
  return true;
}

inline bool is_power_of_two(std::size_t v) { return v && !(v & (v - 1)); }

inline void *usm_backend_alloc(usm::alloc kind, std::size_t bytes,
                               const device &dev) {
#if PARAS_GPU_BACKEND
  if (dev.is_gpu()) {
    if (paras_cuda_detail::select_device(dev.get_native_id()) != cudaSuccess)
      return nullptr;
    void *p = nullptr;
    cudaError_t err = cudaSuccess;
    if (kind == usm::alloc::device)
      err = cudaMalloc(&p, bytes);
    else if (kind == usm::alloc::host)
      err = cudaMallocHost(&p, bytes);
    else {
      err = cudaMallocManaged(&p, bytes);
      if (err == cudaSuccess)
        paras_prepare_shared_allocation(p, bytes, dev.get_native_id());
    }
    if (err != cudaSuccess) {
      (void)cudaGetLastError();
      return nullptr;
    }
    return p;
  }
#endif
  (void)kind;
  (void)dev;
  return std::malloc(bytes);
}

inline void usm_backend_free(void *base, usm::alloc kind, const device &dev) {
#if PARAS_GPU_BACKEND
  if (dev.is_gpu()) {
    (void)cudaDeviceSynchronize();
    if (kind == usm::alloc::host)
      (void)cudaFreeHost(base);
    else
      (void)cudaFree(base);
    return;
  }
#endif
  (void)kind;
  (void)dev;
  std::free(base);
}

inline void *usm_raw_alloc(usm::alloc kind, std::size_t bytes,
                           std::size_t alignment, const device &dev) {
  if (bytes == 0 || kind == usm::alloc::unknown)
    return nullptr;
  if (alignment != 0 && !is_power_of_two(alignment))
    return nullptr;
  const std::size_t natural = alignof(std::max_align_t) > 256 ? alignof(std::max_align_t) : 256;
  const std::size_t extra = alignment > natural ? alignment : 0;
  void *base = usm_backend_alloc(kind, bytes + extra, dev);
  if (base == nullptr)
    return nullptr;
  std::uintptr_t user = reinterpret_cast<std::uintptr_t>(base);
  if (extra)
    user = (user + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
  {
    std::lock_guard<std::mutex> lock(usm_mutex());
    usm_records()[user] = usm_record{base, bytes, kind, dev};
  }
  return reinterpret_cast<void *>(user);
}

inline bool usm_raw_free(void *ptr) {
  if (ptr == nullptr)
    return true;
  usm_record rec;
  {
    std::lock_guard<std::mutex> lock(usm_mutex());
    auto &recs = usm_records();
    auto it = recs.find(reinterpret_cast<std::uintptr_t>(ptr));
    if (it == recs.end())
      return false;
    rec = it->second;
    recs.erase(it);
  }
  usm_backend_free(rec.base, rec.kind, rec.dev);
  return true;
}

inline device usm_host_device(const context &ctx) {
  const auto devs = ctx.get_devices();
  return devs.empty() ? device{} : devs.front();
}

}

inline usm::alloc get_pointer_type(const void *ptr, const context &) {
  std::uintptr_t start;
  detail::usm_record rec;
  return detail::usm_find(ptr, start, rec) ? rec.kind : usm::alloc::unknown;
}
inline device get_pointer_device(const void *ptr, const context &ctx) {
  std::uintptr_t start;
  detail::usm_record rec;
  if (!detail::usm_find(ptr, start, rec))
    throw exception(make_error_code(errc::invalid),
                    "get_pointer_device: not a USM pointer of this context");

  return rec.kind == usm::alloc::host ? detail::usm_host_device(ctx) : rec.dev;
}
namespace usm {

inline alloc get_pointer_type(const void *ptr, const context &ctx) {
  return ::sycl::get_pointer_type(ptr, ctx);
}
}

inline void *malloc_device(std::size_t bytes, const device &dev,
                           const context &, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::device, bytes, 0, dev);
}
inline void *malloc_host(std::size_t bytes, const context &ctx,
                         const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::host, bytes, 0, detail::usm_host_device(ctx));
}
inline void *malloc_shared(std::size_t bytes, const device &dev,
                           const context &, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::shared, bytes, 0, dev);
}
inline void *aligned_alloc_device(std::size_t alignment, std::size_t bytes,
                                  const device &dev, const context &,
                                  const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::device, bytes, alignment, dev);
}
inline void *aligned_alloc_host(std::size_t alignment, std::size_t bytes,
                                const context &ctx, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::host, bytes, alignment, detail::usm_host_device(ctx));
}
inline void *aligned_alloc_shared(std::size_t alignment, std::size_t bytes,
                                  const device &dev, const context &,
                                  const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::shared, bytes, alignment, dev);
}
inline void *malloc(std::size_t bytes, const device &dev, const context &,
                    usm::alloc kind, const property_list & = {}) {
  return detail::usm_raw_alloc(kind, bytes, 0, dev);
}
inline void *aligned_alloc(std::size_t alignment, std::size_t bytes,
                           const device &dev, const context &, usm::alloc kind,
                           const property_list & = {}) {
  return detail::usm_raw_alloc(kind, bytes, alignment, dev);
}

template <typename T>
T *malloc_device(std::size_t count, const device &dev, const context &ctx,
                 const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_device(alignof(T), count * sizeof(T), dev, ctx, p));
}
template <typename T>
T *malloc_host(std::size_t count, const context &ctx, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_host(alignof(T), count * sizeof(T), ctx, p));
}
template <typename T>
T *malloc_shared(std::size_t count, const device &dev, const context &ctx,
                 const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_shared(alignof(T), count * sizeof(T), dev, ctx, p));
}
template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count, const device &dev,
                        const context &ctx, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_device(alignment, count * sizeof(T), dev, ctx, p));
}
template <typename T>
T *aligned_alloc_host(std::size_t alignment, std::size_t count, const context &ctx,
                      const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_host(alignment, count * sizeof(T), ctx, p));
}
template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count, const device &dev,
                        const context &ctx, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_shared(alignment, count * sizeof(T), dev, ctx, p));
}
template <typename T>
T *malloc(std::size_t count, const device &dev, const context &ctx,
          usm::alloc kind, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc(alignof(T), count * sizeof(T), dev, ctx, kind, p));
}
template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count, const device &dev,
                 const context &ctx, usm::alloc kind, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc(alignment, count * sizeof(T), dev, ctx, kind, p));
}

inline void free(void *ptr, const context &) {
  if (!detail::usm_raw_free(ptr))
    std::free(ptr);
}

}

#endif
