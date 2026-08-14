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

#ifndef __PARAS_HANDLER_HPP__
#define __PARAS_HANDLER_HPP__

#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../utilities/internal_utils.hpp"
#include "event.hpp"
#include "id.hpp"
#include "interop_handle.hpp"
#include "nd_item.hpp"
#include "nd_range.hpp"
#include "range.hpp"

class threadpool;
class cuda_threadpool;

namespace sycl {

class handler {
  threadpool *pool_;
  cuda_threadpool *gpu_pool_;
  std::vector<event> deps_;

public:
  handler() : pool_(nullptr), gpu_pool_(nullptr), local_memory_bytes_(0) {}

  handler(threadpool &p)
      : pool_(&p), gpu_pool_(nullptr), local_memory_bytes_(0) {}

  handler(cuda_threadpool &p)
      : pool_(nullptr), gpu_pool_(&p), local_memory_bytes_(0) {}

  template <typename KernelName, typename Func, int dim>
  void parallel_for(range<dim> r, Func f);

  template <typename KernelName, typename Func, int dim>
  void parallel_for(const nd_range<dim> &r, Func f);

  template <typename Func> void interop_task(Func f) {
    interop_handle ih(static_cast<void *>(pool_), sycl::backend::host);
    f(ih);
  }

  inline void memset(void *ptr, int value, size_t num_bytes);
  inline void memcpy(void *dest, const void *src, size_t num_bytes);

  bool isAsyncEnabled() const { return async_mode_; }

  void set_async_mode(bool mode) { async_mode_ = mode; }

  template <class T> std::size_t local_alloc_offset(std::size_t elements) {
    static_assert(!std::is_void<T>::value,
                  "local_accessor<void> is not supported");

    constexpr std::size_t alignment = alignof(T);
    static_assert((alignment & (alignment - 1)) == 0,
                  "local_accessor alignment must be a power of two");

    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("local_accessor allocation size overflow");
    }

    const std::size_t bytes = elements * sizeof(T);

    if (local_memory_bytes_ >
        std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
      throw std::overflow_error("local_accessor alignment overflow");
    }

    const std::size_t aligned_offset =
        (local_memory_bytes_ + alignment - 1) & ~(alignment - 1);

    if (aligned_offset > std::numeric_limits<std::size_t>::max() - bytes) {
      throw std::overflow_error("local_accessor total size overflow");
    }

    local_memory_bytes_ = aligned_offset + bytes;
    return aligned_offset;
  }

  std::size_t local_memory_size() const noexcept { return local_memory_bytes_; }

  void depends_on(const event &e) { deps_.push_back(e); }

  void depends_on(const std::vector<event> &events) {
    deps_.insert(deps_.end(), events.begin(), events.end());
  }

  const std::vector<event> &get_dependencies() const { return deps_; }

  void wait_for_dependencies() const { sycl::event::wait(deps_); }

  template <typename injectCustomFunc>
  void parasSYCL_enqueue_custom_operation(injectCustomFunc &&cgf) {
    if (gpu_pool_ == nullptr) {
      throw std::runtime_error(
          "custom CUDA operation has no cuda_threadpool backend");
    }

    wait_for_dependencies();
    async_mode_ = true;
    interop_handle ih(static_cast<void *>(gpu_pool_), sycl::backend::cuda);

    try {
      cgf(ih);
      async_mode_ = false;
    } catch (...) {
      async_mode_ = false;
      throw;
    }
  }

private:
  std::size_t local_memory_bytes_ = 0;
  bool async_mode_ = false;
};

} // namespace sycl

#endif
