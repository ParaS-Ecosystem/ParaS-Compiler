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

#ifndef __PARAS_THREADPOOL_HPP__
#define __PARAS_THREADPOOL_HPP__

#include "sycl/context.hpp"
#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "sycl/event.hpp"
#include "sycl/id.hpp"
#include "sycl/item.hpp"
#include "sycl/range.hpp"
#include <algorithm>
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
  threadpool() {}
  sycl::device get_device() const { return dev_; }
  sycl::context get_context() const { return ctx_; }

  sycl::backend get_backend() const { return sycl::backend::host; }

  template <typename Selector> threadpool(const Selector &) {}

  explicit threadpool(const sycl::context &ctx, const sycl::device &dev,
                      const sycl::property_list &props = {})
      : ctx_(ctx), dev_(dev) {
    (void)props;
  }

  explicit threadpool(const sycl::device &dev,
                      const sycl::property_list &props = {})
      : dev_(dev) {
    (void)props;
  }

  static unsigned get_num_threads();

  template <typename Func> sycl::event spawn_1D(Func f);

  template <typename Func> sycl::event spawn_1D_event(Func f);

  template <typename Func> void spawn_ND(Func f);

  template <typename CGF> sycl::event submit(CGF &&cgf) {
    sycl::handler cgh(*this);
    std::forward<CGF>(cgf)(cgh);
    return sycl::event{};
  }

  void wait() {}

  void wait_and_throw() { wait(); }

  template <typename Func> void execute_1D(const sycl::range<1> &r, Func f);

  template <typename Func> void execute_2D(const sycl::range<2> &r, Func f);

  template <typename Func>
  void execute_nd_range_1D(const sycl::nd_range<1> &r, Func f);

  template <typename Func>
  void execute_nd_range_2D(const sycl::nd_range<2> &r, Func f);

  template <typename Func>
  void execute_nd_range_3D(const sycl::nd_range<3> &r, Func f);

  template <typename KernelName, typename Func, int dim>
  void parallel_for(sycl::range<dim> r, Func f) {
    if constexpr (dim == 1) {
      execute_1D(r, f);
    } else if constexpr (dim == 2) {
      execute_2D(r, f);
    } else {
      static_assert(dim <= 2, "Only 1D/2D supported");
    }
  }

  template <typename KernelName, typename Func, int dim>
  void parallel_for(const sycl::nd_range<dim> &r, Func f) {
    if constexpr (dim == 1) {
      execute_nd_range_1D(r, f);
    } else if constexpr (dim == 2) {
      execute_nd_range_2D(r, f);
    } else if constexpr (dim == 3) {
      execute_nd_range_3D(r, f);
    } else {
      static_assert(dim <= 3, "Only 1D, 2D and 3D supported");
    }
  }

  sycl::event memcpy(void *dest, const void *src, size_t numBytes) {
    std::memcpy(dest, src, numBytes);
    return sycl::event{};
  }

  template <typename T> sycl::event copy(const T *src, T *dest, size_t count) {
    std::copy(src, src + count, dest);
    return sycl::event{};
  }

  sycl::event memset(void *ptr, int value, size_t numBytes) {
    std::memset(ptr, value, numBytes);
    return sycl::event{};
  }

private:
  sycl::context ctx_{};
  sycl::device dev_{};
};

#include "threadpool_execute_1D.hpp"
#include "threadpool_execute_ND.hpp"
#include "threadpool_execute_common.hpp"
#include "threadpool_execute_nd_range_1D.hpp"
#include "threadpool_execute_nd_range_2D.hpp"
#include "threadpool_execute_nd_range_3D.hpp"
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
  } else if constexpr (dim == 2) {
    pool_->execute_nd_range_2D(r, f);
  } else if constexpr (dim == 3) {
    pool_->execute_nd_range_3D(r, f);
  } else {
    static_assert(dim <= 3, "Only 1D, 2D and 3D supported");
  }
}

void handler::memcpy(void *dest, const void *src, size_t num_bytes) {
  std::memcpy(dest, src, num_bytes);
}

void handler::memset(void *ptr, int value, size_t num_bytes) {
  std::memset(ptr, value, num_bytes);
}

template <typename T> T *malloc_shared(size_t n, const threadpool &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T>
T *malloc_shared(size_t n, const device &dev, const context &ctx,
                 const property_list &propList = {}) {

  (void)dev;
  (void)ctx;
  (void)propList;

  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T> T *malloc_host(size_t n, const threadpool &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T> T *malloc_device(size_t n, const threadpool &) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T>
T *malloc_device(size_t n, const device &dev, const context &ctx,
                 const property_list &propList = {}) {
  (void)dev;
  (void)ctx;
  (void)propList;

  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

inline void free(void *ptr, const threadpool &) { std::free(ptr); }

} // namespace sycl
#endif
