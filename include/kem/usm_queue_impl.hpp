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

#ifndef __PARAS_USM_QUEUE_IMPL_HPP__
#define __PARAS_USM_QUEUE_IMPL_HPP__

#include <cstdlib>
#include <cstring>

#include "sycl/queue.hpp"
#include "sycl/usm.hpp"
#include "kem/usm_core.hpp"

namespace sycl {

template <typename T>
T *malloc_shared(size_t n, const queue &q, const property_list &) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::shared, n * sizeof(T), alignof(T), q.get_device()));
}
template <typename T>
T *malloc_host(size_t n, const queue &q, const property_list &) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::host, n * sizeof(T), alignof(T), q.get_device()));
}
template <typename T>
T *malloc_device(size_t n, const queue &q, const property_list &) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::device, n * sizeof(T), alignof(T), q.get_device()));
}
inline void *malloc_shared(size_t bytes, const queue &q, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::shared, bytes, 0, q.get_device());
}
inline void *malloc_host(size_t bytes, const queue &q, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::host, bytes, 0, q.get_device());
}
inline void *malloc_device(size_t bytes, const queue &q, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::device, bytes, 0, q.get_device());
}
inline void *aligned_alloc_shared(size_t al, size_t bytes, const queue &q, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::shared, bytes, al, q.get_device());
}
inline void *aligned_alloc_host(size_t al, size_t bytes, const queue &q, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::host, bytes, al, q.get_device());
}
inline void *aligned_alloc_device(size_t al, size_t bytes, const queue &q, const property_list & = {}) {
  return detail::usm_raw_alloc(usm::alloc::device, bytes, al, q.get_device());
}
inline void *malloc(size_t bytes, const queue &q, usm::alloc kind, const property_list & = {}) {
  return detail::usm_raw_alloc(kind, bytes, 0, q.get_device());
}
inline void *aligned_alloc(size_t al, size_t bytes, const queue &q, usm::alloc kind,
                           const property_list & = {}) {
  return detail::usm_raw_alloc(kind, bytes, al, q.get_device());
}
template <typename T>
T *aligned_alloc_shared(size_t al, size_t n, const queue &q, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_shared(al, n * sizeof(T), q, p));
}
template <typename T>
T *aligned_alloc_host(size_t al, size_t n, const queue &q, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_host(al, n * sizeof(T), q, p));
}
template <typename T>
T *aligned_alloc_device(size_t al, size_t n, const queue &q, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc_device(al, n * sizeof(T), q, p));
}
template <typename T>
T *malloc(size_t n, const queue &q, usm::alloc kind, const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc(alignof(T), n * sizeof(T), q, kind, p));
}
template <typename T>
T *aligned_alloc(size_t al, size_t n, const queue &q, usm::alloc kind,
                 const property_list &p = {}) {
  return static_cast<T *>(aligned_alloc(al, n * sizeof(T), q, kind, p));
}
inline void free(void *ptr, const queue &) {
  if (!detail::usm_raw_free(ptr))
    std::free(ptr);
}

#if PARAS_GPU_BACKEND
inline event queue::memcpy(void *dest, const void *src, size_t numBytes) {
  if (dev_.is_gpu())
    return get_or_create_gpu_pool().memcpy(dest, src, numBytes);
  std::memcpy(dest, src, numBytes);
  return event{};
}
inline event queue::memset(void *ptr, int value, size_t numBytes) {
  if (dev_.is_gpu())
    return get_or_create_gpu_pool().memset(ptr, value, numBytes);
  std::memset(ptr, value, numBytes);
  return event{};
}
#else
inline event queue::memcpy(void *dest, const void *src, size_t numBytes) {
  std::memcpy(dest, src, numBytes);
  return event{};
}
inline event queue::memset(void *ptr, int value, size_t numBytes) {
  std::memset(ptr, value, numBytes);
  return event{};
}
#endif

template <typename T>
event queue::copy(const T *src, T *dest, size_t count) {
  return memcpy(dest, src, count * sizeof(T));
}

template <typename T, usm::alloc AllocKind, std::size_t Alignment = 0>
class usm_allocator {
  static_assert(AllocKind != usm::alloc::device,
                "usm_allocator does not support device allocations");

public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  template <typename U> struct rebind {
    using other = usm_allocator<U, AllocKind, Alignment>;
  };

  usm_allocator() = delete;
  usm_allocator(const context &ctx, const device &dev,
                const property_list &props = {})
      : ctx_(ctx), dev_(dev), props_(props) {}
  usm_allocator(const queue &q, const property_list &props = {})
      : ctx_(q.get_context()), dev_(q.get_device()), props_(props) {}
  usm_allocator(const usm_allocator &) = default;
  usm_allocator(usm_allocator &&) noexcept = default;
  usm_allocator &operator=(const usm_allocator &) = default;
  usm_allocator &operator=(usm_allocator &&) = default;
  template <class U>
  usm_allocator(const usm_allocator<U, AllocKind, Alignment> &other) noexcept
      : ctx_(other.ctx_), dev_(other.dev_), props_(other.props_) {}

  T *allocate(std::size_t count) {
    const std::size_t al = Alignment > alignof(T) ? Alignment : alignof(T);
    T *p = static_cast<T *>(detail::usm_raw_alloc(AllocKind, count * sizeof(T), al, dev_));
    if (p == nullptr && count != 0)
      throw exception(make_error_code(errc::memory_allocation),
                      "usm_allocator: allocation failed");
    return p;
  }
  void deallocate(T *ptr, std::size_t) {
    if (ptr != nullptr)
      sycl::free(ptr, ctx_);
  }

  template <class U, usm::alloc K2, std::size_t A2>
  friend bool operator==(const usm_allocator &a, const usm_allocator<U, K2, A2> &b) {
    return AllocKind == K2 && Alignment == A2 && a.ctx_ == b.ctx_ && a.dev_ == b.dev_;
  }
  template <class U, usm::alloc K2, std::size_t A2>
  friend bool operator!=(const usm_allocator &a, const usm_allocator<U, K2, A2> &b) {
    return !(a == b);
  }

private:
  template <typename, usm::alloc, std::size_t> friend class usm_allocator;
  context ctx_;
  device dev_;
  property_list props_;
};

}

#endif
