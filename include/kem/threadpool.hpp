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
#include "sycl/kernel.hpp"
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

  void paras_set_async_handler(const sycl::async_handler &h) {
    if (h)
      async_ = std::make_shared<sycl::async_handler>(h);
  }
  void paras_attach_reporter(sycl::event &e) const {
    auto h = async_ ? async_ : ctx_.paras_async_handler();
    if (h)
      e.paras_set_reporter([h](std::vector<std::exception_ptr> errs) {
        (*h)(sycl::exception_list(std::move(errs)));
      });
  }

  void throw_asynchronous() { pending_->report_all(); }

  const void *paras_identity() const { return submit_mutex_.get(); }
  std::shared_ptr<const void> paras_identity_ptr() const { return submit_mutex_; }
  friend bool operator==(const threadpool &a, const threadpool &b) {
    return a.paras_identity() == b.paras_identity();
  }
  friend bool operator!=(const threadpool &a, const threadpool &b) { return !(a == b); }
  bool is_in_order() const { return props_.has_in_order(); }
  template <typename AccT> sycl::event update_host(AccT) { return sycl::event{}; }

  template <typename SrcT, typename DestT>
  auto copy(SrcT src, DestT dest)
      -> decltype(std::declval<sycl::handler &>().copy(src, dest), sycl::event{}) {
    sycl::handler cgh(*this);
    cgh.copy(src, dest);
    return sycl::event{};
  }
  template <typename AccT, typename T,
            typename = decltype(std::declval<AccT &>().get_pointer())>
  sycl::event fill(AccT dest, const T &pattern) {
    sycl::handler cgh(*this);
    cgh.fill(dest, pattern);
    return sycl::event{};
  }

  sycl::event prefetch(const void *, size_t) { return sycl::event{}; }
  sycl::event mem_advise(const void *, size_t, int) { return sycl::event{}; }

  sycl::event prefetch(const void *p, size_t n, sycl::event dep) { dep.wait(); return prefetch(p, n); }
  sycl::event prefetch(const void *p, size_t n, const std::vector<sycl::event> &deps) {
    for (auto &e : deps) e.wait();
    return prefetch(p, n);
  }
  sycl::event mem_advise(const void *p, size_t n, int a, sycl::event dep) { dep.wait(); return mem_advise(p, n, a); }
  sycl::event mem_advise(const void *p, size_t n, int a, const std::vector<sycl::event> &deps) {
    for (auto &e : deps) e.wait();
    return mem_advise(p, n, a);
  }

  template <typename CGF, typename SecondaryQ> sycl::event submit(CGF cgf, SecondaryQ &) { return submit(cgf); }

  template <typename Param> typename Param::return_type get_info() const {
    if constexpr (std::is_same_v<typename Param::return_type, sycl::context>)
      return get_context();
    else
      return get_device();
  }

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

  sycl::context get_context() const {
    return own_ctx_ ? ctx_
                    : sycl::detail::paras_default_context(dev_.get_platform());
  }

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
      : ctx_(ctx), own_ctx_(true), dev_(dev) {
    props_ = props;
  }

  explicit threadpool(const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : threadpool(sycl::default_selector_v, props) { paras_set_async_handler(paras_ah); }
  explicit threadpool(const sycl::device &dev, const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : threadpool(dev, props) { paras_set_async_handler(paras_ah); }
  explicit threadpool(const sycl::context &ctx, const sycl::device &dev,
                 const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : threadpool(ctx, dev, props) { paras_set_async_handler(paras_ah); }
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
                 const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : threadpool(ctx, selector, props) { paras_set_async_handler(paras_ah); }

  explicit threadpool(const sycl::device &dev,
                      const sycl::property_list &props = {})
      : dev_(dev), props_(props) {}

  template <typename Selector,
            typename = std::enable_if_t<std::is_invocable_r_v<
                int, const Selector &, const sycl::device &>>>
  threadpool(const Selector &selector, const sycl::async_handler &paras_ah,
             const sycl::property_list &props = {})
      : threadpool(selector, props) { paras_set_async_handler(paras_ah); }

  template <typename Q,
            typename = std::enable_if_t<std::is_same_v<Q, sycl::queue>>>
  threadpool(const Q &q)
      : ctx_(q.get_context()), own_ctx_(true), dev_(q.get_device()) {}

  static unsigned get_num_threads();

  template <typename Func> sycl::event spawn_1D(Func f);

  template <typename Func> sycl::event spawn_1D_event(Func f);

  template <typename Func> sycl::event spawn_ND(Func f);

  template <typename CGF> sycl::event submit(CGF &&cgf) {
    std::lock_guard<std::mutex> lock(*submit_mutex_);
    const std::uint64_t t_submit = sycl::event::paras_now_ns();
    sycl::handler cgh(*this);
    std::forward<CGF>(cgf)(cgh);
    sycl::event e = cgh.paras_take_event();
    if (props_.template has_property<sycl::property::queue::enable_profiling>())
      e.paras_start_profiling(t_submit);
    pending_->add(e);
    return e;
  }

  void wait() { pending_->wait_all(); }
  void paras_add_pending(sycl::event e) {
    paras_attach_reporter(e);
    pending_->add(e);
  }

  void wait_and_throw() { pending_->report_all(); }

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
    sycl::detail::note_and_check_kernel<
        KernelName, sycl::detail::single_task_kernel<Func>>(get_device());
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
    sycl::detail::note_and_check_kernel<KernelName, Func>(get_device());
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
    sycl::detail::note_and_check_kernel<KernelName, Func>(get_device());
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

  template <typename KernelName = void, int dim, typename First, typename... Rest,
            typename = std::enable_if_t<sycl::detail::is_reduction_v<First>>>
  sycl::event parallel_for(sycl::range<dim> r, First &&first, Rest &&...rest) {
    return submit([&](sycl::handler &cgh) {
      cgh.template parallel_for<KernelName>(r, first, rest...);
    });
  }
  template <typename KernelName = void, int dim, typename First, typename... Rest,
            typename = std::enable_if_t<sycl::detail::is_reduction_v<First>>>
  sycl::event parallel_for(const sycl::nd_range<dim> &r, First &&first, Rest &&...rest) {
    return submit([&](sycl::handler &cgh) {
      cgh.template parallel_for<KernelName>(r, first, rest...);
    });
  }
  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(sycl::range<dim> r, sycl::event dep, Func f) {
    dep.wait();
    return parallel_for<KernelName>(r, f);
  }
  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(sycl::range<dim> r,
                           const std::vector<sycl::event> &deps, Func f) {
    for (auto &e : deps) e.wait();
    return parallel_for<KernelName>(r, f);
  }
  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(const sycl::nd_range<dim> &r, sycl::event dep,
                           Func f) {
    dep.wait();
    return parallel_for<KernelName>(r, f);
  }
  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(const sycl::nd_range<dim> &r,
                           const std::vector<sycl::event> &deps, Func f) {
    for (auto &e : deps) e.wait();
    return parallel_for<KernelName>(r, f);
  }

  template <typename KernelName = void, typename Func>
  sycl::event parallel_for(sycl::range<2> r, Func f) {
    return parallel_for<KernelName, Func, 2>(r, f);
  }
  template <typename KernelName = void, typename Func>
  sycl::event parallel_for(sycl::range<3> r, Func f) {
    return parallel_for<KernelName, Func, 3>(r, f);
  }

  template <typename KernelName = void, typename Func>
  sycl::event parallel_for(std::size_t n, Func f) {
    return parallel_for<KernelName>(sycl::range<1>(n), f);
  }
  template <typename KernelName = void, typename Func>
  sycl::event parallel_for(std::size_t n, sycl::event dep, Func f) {
    return parallel_for<KernelName>(sycl::range<1>(n), dep, f);
  }
  template <typename KernelName = void, typename Func>
  sycl::event parallel_for(std::size_t n, const std::vector<sycl::event> &deps,
                           Func f) {
    return parallel_for<KernelName>(sycl::range<1>(n), deps, f);
  }
  sycl::event memcpy(void *dest, const void *src, size_t numBytes,
                     sycl::event dep) {
    dep.wait();
    return memcpy(dest, src, numBytes);
  }
  sycl::event memcpy(void *dest, const void *src, size_t numBytes,
                     const std::vector<sycl::event> &deps) {
    for (auto &e : deps) e.wait();
    return memcpy(dest, src, numBytes);
  }
  sycl::event memset(void *ptr, int value, size_t numBytes, sycl::event dep) {
    dep.wait();
    return memset(ptr, value, numBytes);
  }
  sycl::event memset(void *ptr, int value, size_t numBytes,
                     const std::vector<sycl::event> &deps) {
    for (auto &e : deps) e.wait();
    return memset(ptr, value, numBytes);
  }
  template <typename T>
  sycl::event copy(const T *src, T *dest, size_t count, sycl::event dep) {
    dep.wait();
    return copy(src, dest, count);
  }
  template <typename T>
  sycl::event copy(const T *src, T *dest, size_t count,
                   const std::vector<sycl::event> &deps) {
    for (auto &e : deps) e.wait();
    return copy(src, dest, count);
  }
  template <typename T>
  sycl::event fill(void *ptr, const T &pattern, size_t count, sycl::event dep) {
    dep.wait();
    return fill(ptr, pattern, count);
  }
  template <typename T>
  sycl::event fill(void *ptr, const T &pattern, size_t count,
                   const std::vector<sycl::event> &deps) {
    for (auto &e : deps) e.wait();
    return fill(ptr, pattern, count);
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
  bool own_ctx_ = false;
  sycl::device dev_{};
  sycl::property_list props_{};

  std::shared_ptr<std::mutex> submit_mutex_ = std::make_shared<std::mutex>();
  std::shared_ptr<sycl::async_handler> async_;
  std::shared_ptr<sycl::detail::pending_events> pending_ =
      std::make_shared<sycl::detail::pending_events>();
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

inline void handler::paras_enqueue_dependencies() { wait_for_dependencies(); }
inline void handler::paras_gpu_command_enqueued(bool) {}

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(range<dim> r, Func f) {
  paras_launch_kernel<KernelName, Func>();
  paras_check_range_kernel<Func>();
  wait_for_dependencies();

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
  paras_launch_kernel<KernelName, Func>();
  paras_check_reqd_work_group_size<Func>(r);
  wait_for_dependencies();

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

inline void handler::memcpy(void *dest, const void *src, size_t num_bytes) {
  std::memcpy(dest, src, num_bytes);
}

inline void handler::memset(void *ptr, int value, size_t num_bytes) {
  std::memset(ptr, value, num_bytes);
}

}
#include "kem/usm_core.hpp"
namespace sycl {

template <typename T> T *malloc_shared(size_t n, const threadpool &q) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::shared, n * sizeof(T), alignof(T), q.get_device()));
}
template <typename T> T *malloc_host(size_t n, const threadpool &q) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::host, n * sizeof(T), alignof(T), q.get_device()));
}
template <typename T> T *malloc_device(size_t n, const threadpool &q) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::device, n * sizeof(T), alignof(T), q.get_device()));
}
inline void free(void *ptr, const threadpool &) {
  if (!detail::usm_raw_free(ptr))
    std::free(ptr);
}

}
namespace std {
template <> struct hash<threadpool> {
  size_t operator()(const threadpool &q) const {
    return std::hash<const void *>()(q.paras_identity());
  }
};
}

#include "kem/usm_queue_impl.hpp"
#include "kem/reduction_impl.hpp"
#if !PARAS_GPU_BACKEND

inline sycl::device sycl::handler::paras_reduction_device() const {
  return pool_ ? pool_->get_device() : sycl::device{};
}
inline threadpool &sycl::queue::get_or_create_host_pool() const {
  if (!host_pool_)
    host_pool_ = std::make_shared<threadpool>(get_context(), dev_, props_);
  return *host_pool_;
}
inline void sycl::handler::paras_reduction_wait() {}
#endif

#endif
