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

namespace sycl {

#if PARAS_GPU_BACKEND

template <typename T>
T *malloc_shared(size_t n, const queue &q, const property_list &) {
  return q.get_device().is_gpu() ? malloc_shared<T>(n, q.paras_gpu_pool())
                                 : static_cast<T *>(std::malloc(sizeof(T) * n));
}
template <typename T>
T *malloc_host(size_t n, const queue &q, const property_list &) {
  return q.get_device().is_gpu() ? malloc_host<T>(n, q.paras_gpu_pool())
                                 : static_cast<T *>(std::malloc(sizeof(T) * n));
}
template <typename T>
T *malloc_device(size_t n, const queue &q, const property_list &) {
  return q.get_device().is_gpu() ? malloc_device<T>(n, q.paras_gpu_pool())
                                 : static_cast<T *>(std::malloc(sizeof(T) * n));
}
inline void free(void *ptr, const queue &q) {
  if (q.get_device().is_gpu())
    free(ptr, q.paras_gpu_pool());
  else
    std::free(ptr);
}
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

template <typename T>
T *malloc_shared(size_t n, const queue &, const property_list &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}
template <typename T>
T *malloc_host(size_t n, const queue &, const property_list &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}
template <typename T>
T *malloc_device(size_t n, const queue &, const property_list &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}
inline void free(void *ptr, const queue &) { std::free(ptr); }
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

}

#endif
