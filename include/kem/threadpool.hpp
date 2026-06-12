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

#ifndef __PARAS_THREADPOOL_HPP__
#define __PARAS_THREADPOOL_HPP__

#include "sycl/context.hpp"
#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "sycl/event.hpp"
#include "sycl/id.hpp"
#include "sycl/item.hpp"
#include "sycl/range.hpp"
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace sycl {
class handler;
}
#include "sycl/handler.hpp"

class threadpool {
public:
  threadpool() {};
  sycl::device get_device() const { return sycl::device(); }
  sycl::context get_context() const { return sycl::context(); }

  sycl::backend get_backend() const { return sycl::backend::host; }

  template <typename Selector> threadpool(const Selector &) {}
  static unsigned get_num_threads();

  template <typename Func> void spawn_1D(Func f);

  template <typename Func> sycl::event spawn_1D_event(Func f);

  template <typename Func> void spawn_ND(Func f);

  void wait() {}

  template <typename Func> void execute_1D(const sycl::range<1> &r, Func f);

  template <typename Func> void execute_2D(const sycl::range<2> &r, Func f);

  template <typename Func>
  void execute_nd_range_1D(const sycl::nd_range<1> &r, Func f);

  template <typename Func>
  void execute_nd_range_2D(const sycl::nd_range<2> &r, Func f);

  template <typename KernelName, typename Func, int dim>
  void parallel_for(sycl::range<dim> r, Func f) {
    if constexpr (dim == 1) {
      execute_1D(r, f);
    } else if (dim == 2) {
      execute_2D(r, f);
    } else {
      static_assert(dim <= 2, "Only 1D/2D supported");
    }
  }

  template <typename KernelName, typename Func, int dim>
  void parallel_for(const sycl::nd_range<dim> &r, Func f) {
    if constexpr (dim == 1) {
      execute_nd_range_1D(r, f);
    } else if (dim == 2) {
      execute_nd_range_2D(r, f);
    } else {
      static_assert(dim <= 2, "Only 1D/2D supported");
    }
  }

  sycl::event memset(void *ptr, int value, size_t numBytes) {
    std::memset(ptr, value, numBytes);
    return sycl::event{};
  }

  sycl::event memcpy(void *dest, const void *src, size_t numBytes) {
    std::memcpy(dest, src, numBytes);
    return sycl::event{};
  }
};

#include "threadpool_execute_1D.hpp"
#include "threadpool_execute_ND.hpp"
#include "threadpool_execute_common.hpp"
#include "threadpool_execute_nd_range_1D.hpp"
#include "threadpool_execute_nd_range_2D.hpp"
#include "threadpool_spawn.hpp"

namespace sycl {

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(range<dim> r, Func f) {

  if constexpr (dim == 1) {
    pool_->execute_1D(r, f);
  } else if (dim == 2) {
    pool_->execute_2D(r, f);
  } else {
    static_assert(dim <= 2, "Only 1D and 2D supported");
  }
}

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(const nd_range<dim> &r, Func f) {

  if constexpr (dim == 1) {
    pool_->execute_nd_range_1D(r, f);
  } else if (dim == 2) {
    pool_->execute_nd_range_2D(r, f);
  } else {
    static_assert(dim <= 2, "Only 1D and 2D supported");
  }
}

void handler::memset(void *ptr, int value, size_t num_bytes) {
  std::memset(ptr, value, num_bytes);
}

void handler::memcpy(void *dest, const void *src, size_t num_bytes) {
  std::memcpy(dest, src, num_bytes);
}

template <typename T> T *malloc_shared(size_t n, const threadpool &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T> T *malloc_host(size_t n, const threadpool &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T> T *malloc_device(size_t n, const threadpool &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

inline void free(void *ptr, const threadpool &) { std::free(ptr); }

} // namespace sycl
#endif
