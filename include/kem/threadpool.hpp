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
#include "sycl/property_list.hpp"
#include "sycl/range.hpp"
#include "utilities/selector_logic.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace sycl {
class handler;
}
#include "sycl/handler.hpp"

class threadpool {
public:
  using paras_pool_tag = void;

  sycl::event prefetch(const void *, size_t) { return sycl::event{}; }
  sycl::event mem_advise(const void *, size_t, int) { return sycl::event{}; }

  template <typename PropT> bool has_property() const noexcept {
    return props_.template has_property<PropT>();
  }
  template <typename PropT> PropT get_property() const {
    if (!has_property<PropT>())
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "queue does not have the requested property");
    return props_.template get_property<PropT>();
  }

  threadpool() {}

  explicit threadpool(const sycl::property_list &props)
      : threadpool(sycl::default_selector_v, props) {}
  sycl::device get_device() const { return dev_; }
  sycl::context get_context() const { return ctx_; }

  sycl::backend get_backend() const { return sycl::backend::host; }

  template <typename Selector,
            typename = std::enable_if_t<std::is_invocable_r_v<
                int, const Selector &, const sycl::device &>>>
  explicit threadpool(const Selector &selector,
                      const sycl::property_list &props = {})
      : dev_(::paras_extension::select_device_with_selector(selector)),
        props_(props) {}

  explicit threadpool(const sycl::context &ctx, const sycl::device &dev,
                      const sycl::property_list &props = {})
      : ctx_(ctx), dev_(dev) {
    props_ = props;
  }

  explicit threadpool(const sycl::async_handler &,
                 const sycl::property_list &props = {})
      : threadpool(sycl::default_selector_v, props) {}
  explicit threadpool(const sycl::device &dev, const sycl::async_handler &,
                 const sycl::property_list &props = {})
      : threadpool(dev, props) {}
  explicit threadpool(const sycl::context &ctx, const sycl::device &dev,
                 const sycl::async_handler &,
                 const sycl::property_list &props = {})
      : threadpool(ctx, dev, props) {}
  template <typename Selector,
              typename = std::enable_if_t<std::is_invocable_r_v<
                  int, const Selector &, const sycl::device &>>>
  explicit threadpool(const sycl::context &ctx, const Selector &selector,
                 const sycl::property_list &props = {})
      : threadpool(ctx, ::paras_extension::select_device_with_selector(selector),
              props) {}
  template <typename Selector,
              typename = std::enable_if_t<std::is_invocable_r_v<
                  int, const Selector &, const sycl::device &>>>
  explicit threadpool(const sycl::context &ctx, const Selector &selector,
                 const sycl::async_handler &,
                 const sycl::property_list &props = {})
      : threadpool(ctx, selector, props) {}

  explicit threadpool(const sycl::device &dev,
                      const sycl::property_list &props = {})
      : dev_(dev), props_(props) {}

  template <typename Selector,
            typename = std::enable_if_t<std::is_invocable_r_v<
                int, const Selector &, const sycl::device &>>>
  threadpool(const Selector &selector, const sycl::async_handler &,
             const sycl::property_list &props = {})
      : threadpool(selector, props) {}

  template <typename Q,
            typename = std::enable_if_t<std::is_same_v<Q, sycl::queue>>>
  threadpool(const Q &q) : ctx_(q.get_context()), dev_(q.get_device()) {}

  static unsigned get_num_threads();

  template <typename Func> sycl::event spawn_1D(Func f);

  template <typename Func> sycl::event spawn_1D_event(Func f);

  template <typename Func> sycl::event spawn_ND(Func f);

  template <typename CGF> sycl::event submit(CGF &&cgf) {
    std::lock_guard<std::mutex> lock(*submit_mutex_);
    sycl::handler cgh(*this);
    std::forward<CGF>(cgf)(cgh);
    return sycl::event{};
  }

  void wait() {}

  void wait_and_throw() { wait(); }

  template <typename Func> void execute_1D(const sycl::range<1> &r, Func f);

  template <typename Func> void execute_2D(const sycl::range<2> &r, Func f);

  template <typename Func> void execute_3D(const sycl::range<3> &r, Func f);

  template <typename Func>
  void execute_nd_range_1D(const sycl::nd_range<1> &r, Func f);

  template <typename Func>
  void execute_nd_range_2D(const sycl::nd_range<2> &r, Func f);

  template <typename Func>
  void execute_nd_range_3D(const sycl::nd_range<3> &r, Func f);

  template <typename KernelName = void, typename Func>
  sycl::event single_task(Func f) {
    execute_1D(sycl::range<1>(1), sycl::detail::single_task_kernel<Func>{f});
    return sycl::event{};
  }
  template <typename KernelName = void, typename Func>
  sycl::event single_task(sycl::event, Func f) {
    return single_task<KernelName>(f);
  }
  template <typename KernelName = void, typename Func>
  sycl::event single_task(const std::vector<sycl::event> &, Func f) {
    return single_task<KernelName>(f);
  }

  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(sycl::range<dim> r, Func f) {
    if constexpr (dim == 1) {
      execute_1D(r, f);
    } else if constexpr (dim == 2) {
      execute_2D(r, f);
    } else if constexpr (dim == 3) {
      execute_3D(r, f);
    } else {
      static_assert(dim <= 3, "Only 1D/2D/3D supported");
    }
    return sycl::event{};
  }

  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(const sycl::nd_range<dim> &r, Func f) {
    if constexpr (dim == 1) {
      execute_nd_range_1D(r, f);
    } else if constexpr (dim == 2) {
      execute_nd_range_2D(r, f);
    } else if constexpr (dim == 3) {
      execute_nd_range_3D(r, f);
    } else {
      static_assert(dim <= 3, "Only 1D, 2D and 3D supported");
    }
    return sycl::event{};
  }

  template <typename T>
  sycl::event fill(void *ptr, const T &pattern, size_t count) {
    T *p = static_cast<T *>(ptr);
    for (size_t i = 0; i < count; ++i)
      p[i] = pattern;
    return sycl::event{};
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
  sycl::property_list props_{};

  std::shared_ptr<std::mutex> submit_mutex_ = std::make_shared<std::mutex>();
};

#include "threadpool_execute_1D.hpp"
#include "threadpool_execute_ND.hpp"
#include "threadpool_execute_3D.hpp"
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
  } else if constexpr (dim == 2) {
    pool_->execute_2D(r, f);
  } else if constexpr (dim == 3) {
    pool_->execute_3D(r, f);
  } else {
    static_assert(dim <= 3, "Only 1D, 2D and 3D supported");
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

}
#endif
