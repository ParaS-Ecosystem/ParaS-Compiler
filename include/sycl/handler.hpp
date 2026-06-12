/**<
 * Copyright 2025 The ParaS-Compiler Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_HANDLER_HPP__
#define __PARAS_HANDLER_HPP__

#include "../utilities/internal_utils.hpp"
#include "event.hpp"
#include "id.hpp"
#include "interop_handle.hpp"
#include "nd_item.hpp"
#include "nd_range.hpp"
#include "range.hpp"
#include <functional>
#include <vector>

class threadpool;
class cuda_threadpool;

namespace sycl {

class handler {
  threadpool *pool_;
  cuda_threadpool *gpu_pool_;

  std::vector<event> deps_;

public:
  handler() : pool_(nullptr), gpu_pool_(nullptr) {}
  handler(threadpool &p) : pool_(&p), gpu_pool_(nullptr) {}
  handler(cuda_threadpool &p) : pool_(nullptr), gpu_pool_(&p) {}

  template <typename KernelName, typename Func, int dim>
  void parallel_for(range<dim> r, Func f);

  template <typename KernelName, typename Func, int dim>
  void parallel_for(const nd_range<dim> &r, Func f);

  template <typename Func> void interop_task(Func f) {
    interop_handle ih(static_cast<void *>(pool_));
    f(ih);
  }

  inline void memset(void *ptr, int value, size_t num_bytes);
  inline void memcpy(void *dest, const void *src, size_t num_bytes);

  bool isAsyncEnabled() const { return async_mode_; }

  void set_async_mode(bool mode) { async_mode_ = mode; }

  template <class T>
  paras_extension::local_memory::address local_alloc(size_t elements) {
    return _alloc_loc_mem.alloc<T>(elements);
  }

  void depends_on(const event &e) { deps_.push_back(e); }

  void depends_on(const std::vector<event> &events) {
    deps_.insert(deps_.end(), events.begin(), events.end());
  }

  const std::vector<event> &get_dependencies() const { return deps_; }

  template <typename injectCustomFunc>
  void parasSYCL_enqueue_custom_operation(injectCustomFunc &&cgf) {
    async_mode_ = true;
    interop_handle ih(static_cast<void *>(gpu_pool_));
    cgf(ih);
    async_mode_ = false;
  }

private:
  paras_extension::local_memory_allocator _alloc_loc_mem;
  bool async_mode_ = false;
};

} // namespace sycl

#endif
