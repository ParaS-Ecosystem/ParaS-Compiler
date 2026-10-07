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

#ifndef __PARAS_QUEUE_HPP__
#define __PARAS_QUEUE_HPP__

#include "context.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "property_list.hpp"
#include "utilities/selector_logic.hpp"
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

class cuda_threadpool;

namespace sycl {

namespace detail {

template <typename Pool>
auto adopt_gpu_state(std::shared_ptr<queue_gpu_state> &state, const Pool &pool,
                     int) -> decltype(pool.paras_gpu_state(), void()) {
  state = pool.paras_gpu_state();
}
template <typename Pool>
void adopt_gpu_state(std::shared_ptr<queue_gpu_state> &, const Pool &, long) {}

}

class queue {
public:
  queue() : dev_(::paras_extension::select_device_no_selector()) {}

  explicit queue(const property_list &props)
      : dev_(::paras_extension::select_device_no_selector()), props_(props) {}

  explicit queue(const decltype(cpu_selector_v) &paras_sel,
                 const property_list &props = {})
      : dev_(::paras_extension::select_device_with_selector(paras_sel)),
        props_(props) {}
  explicit queue(const decltype(gpu_selector_v) &paras_sel,
                 const property_list &props = {})
      : dev_(::paras_extension::select_device_with_selector(paras_sel)),
        props_(props) {}
  explicit queue(const decltype(accelerator_selector_v) &paras_sel,
                 const property_list &props = {})
      : dev_(::paras_extension::select_device_with_selector(paras_sel)),
        props_(props) {}
  explicit queue(const decltype(default_selector_v) &paras_sel,
                 const property_list &props = {})
      : dev_(::paras_extension::select_device_with_selector(paras_sel)),
        props_(props) {}

  explicit queue(const context &ctx, const device &dev,
                 const property_list &props = {});
  explicit queue(const device &dev, const property_list &props = {});

  explicit queue(const async_handler &paras_ah, const property_list &props = {})
      : queue(props) { paras_set_async_handler(paras_ah); }
  explicit queue(const device &dev, const async_handler &paras_ah,
                 const property_list &props = {})
      : queue(dev, props) { paras_set_async_handler(paras_ah); }
  explicit queue(const context &ctx, const device &dev, const async_handler &paras_ah,
                 const property_list &props = {})
      : queue(ctx, dev, props) { paras_set_async_handler(paras_ah); }

  template <typename Selector,
            typename = std::enable_if_t<
                std::is_invocable_r_v<int, const Selector &, const device &>>>
  explicit queue(const Selector &selector, const property_list &props = {})
      : dev_(::paras_extension::select_device_with_selector(selector)),
        props_(props) {}
  template <typename Selector,
            typename = std::enable_if_t<
                std::is_invocable_r_v<int, const Selector &, const device &>>>
  explicit queue(const Selector &selector, const async_handler &paras_ah,
                 const property_list &props = {})
      : queue(selector, props) { paras_set_async_handler(paras_ah); }
  template <typename Selector,
            typename = std::enable_if_t<
                std::is_invocable_r_v<int, const Selector &, const device &>>>
  explicit queue(const context &ctx, const Selector &selector,
                 const property_list &props = {})
      : queue(ctx, ::paras_extension::select_device_with_selector(selector),
              props) {}
  template <typename Selector,
            typename = std::enable_if_t<
                std::is_invocable_r_v<int, const Selector &, const device &>>>
  explicit queue(const context &ctx, const Selector &selector,
                 const async_handler &paras_ah, const property_list &props = {})
      : queue(ctx, selector, props) { paras_set_async_handler(paras_ah); }

  template <typename Pool, typename = typename Pool::paras_pool_tag>
  queue(const Pool &pool) : queue(pool.get_context(), pool.get_device()) {
    identity_ = pool.paras_identity_ptr();
    detail::adopt_gpu_state(gpu_state_, pool, 0);
  }

  template <typename CGF> event submit(CGF cgf) {
    event e;
#if PARAS_GPU_BACKEND
    if (dev_.is_gpu()) {
      handler h(get_or_create_gpu_pool());
      cgf(h);
      e = h.paras_take_event();
    } else
#endif
    {
#if PARAS_GPU_BACKEND
      handler h;
#else

      handler h(get_or_create_host_pool());
#endif
      cgf(h);
      e = h.paras_take_event();
    }
    paras_attach_reporter(e);
    pending_->add(e);
    return e;
  }

  void wait();

  void wait_and_throw();

  device get_device() const { return dev_; }
  context get_context() const { return ctx_; }
  bool is_in_order() const { return props_.has_in_order(); }

  backend get_backend() const {
#if PARAS_HIP_BACKEND
    return dev_.is_gpu() ? backend::hip : backend::host;
#else
    return dev_.is_gpu() ? backend::cuda : backend::host;
#endif
  }

  template <typename KernelName = void, typename Func, int dim>
  event parallel_for(range<dim> r, Func f) {
    return submit(
        [&](handler &cgh) { cgh.template parallel_for<KernelName>(r, f); });
  }

  template <typename KernelName = void, int dim, typename First, typename... Rest,
            typename = std::enable_if_t<detail::is_reduction_v<First>>>
  event parallel_for(range<dim> r, First &&first, Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName>(r, first, rest...);
    });
  }
  template <typename KernelName = void, int dim, typename First, typename... Rest,
            typename = std::enable_if_t<detail::is_reduction_v<First>>>
  event parallel_for(const nd_range<dim> &r, First &&first, Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName>(r, first, rest...);
    });
  }
  template <typename KernelName = void, typename Func, int dim>
  event parallel_for(range<dim> r, event dep, Func f) {
    return submit([&](handler &cgh) {
      cgh.depends_on(dep);
      cgh.template parallel_for<KernelName>(r, f);
    });
  }

  template <typename KernelName = void, typename Func, int dim>
  event parallel_for(range<dim> r, const std::vector<event> &deps, Func f) {
    return submit([&](handler &cgh) {
      cgh.depends_on(deps);
      cgh.template parallel_for<KernelName>(r, f);
    });
  }

  template <typename KernelName = void, typename Func, int dim>
  event parallel_for(const nd_range<dim> &r, Func f) {
    return submit(
        [&](handler &cgh) { cgh.template parallel_for<KernelName>(r, f); });
  }

  template <typename KernelName = void, typename Func, int dim>
  event parallel_for(const nd_range<dim> &r, event dep, Func f) {
    return submit([&](handler &cgh) {
      cgh.depends_on(dep);
      cgh.template parallel_for<KernelName>(r, f);
    });
  }

  template <typename KernelName = void, typename Func, int dim>
  event parallel_for(const nd_range<dim> &r, const std::vector<event> &deps,
                     Func f) {
    return submit([&](handler &cgh) {
      cgh.depends_on(deps);
      cgh.template parallel_for<KernelName>(r, f);
    });
  }

  template <typename KernelName = void, typename Func>
  event single_task(Func f) {
    return submit([&](handler &cgh) { cgh.template single_task<KernelName>(f); });
  }
  template <typename KernelName = void, typename Func>
  event single_task(event dep, Func f) {
    return submit([&](handler &cgh) {
      cgh.depends_on(dep);
      cgh.template single_task<KernelName>(f);
    });
  }
  template <typename KernelName = void, typename Func>
  event single_task(const std::vector<event> &deps, Func f) {
    return submit([&](handler &cgh) {
      cgh.depends_on(deps);
      cgh.template single_task<KernelName>(f);
    });
  }

  template <typename injectCustomFunc>
  event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.paras_request_native_event();
      cgh.parasSYCL_enqueue_custom_operation(cgf);
    });
  }

  event memset(void *ptr, int value, size_t numBytes);
  template <typename T> event copy(const T *src, T *dest, size_t count);
  event memcpy(void *dest, const void *src, size_t numBytes);

  event memcpy(void *dest, const void *src, size_t numBytes,
                     event dep) {
    dep.wait();
    return memcpy(dest, src, numBytes);
  }
  event memcpy(void *dest, const void *src, size_t numBytes,
                     const std::vector<event> &deps) {
    for (auto &e : deps) e.wait();
    return memcpy(dest, src, numBytes);
  }
  event memset(void *ptr, int value, size_t numBytes, event dep) {
    dep.wait();
    return memset(ptr, value, numBytes);
  }
  event memset(void *ptr, int value, size_t numBytes,
                     const std::vector<event> &deps) {
    for (auto &e : deps) e.wait();
    return memset(ptr, value, numBytes);
  }
  template <typename T>
  event copy(const T *src, T *dest, size_t count, event dep) {
    dep.wait();
    return copy(src, dest, count);
  }
  template <typename T>
  event copy(const T *src, T *dest, size_t count,
                   const std::vector<event> &deps) {
    for (auto &e : deps) e.wait();
    return copy(src, dest, count);
  }
  template <typename T>
  event fill(void *ptr, const T &pattern, size_t count, event dep) {
    dep.wait();
    return fill(ptr, pattern, count);
  }
  template <typename T>
  event fill(void *ptr, const T &pattern, size_t count,
                   const std::vector<event> &deps) {
    for (auto &e : deps) e.wait();
    return fill(ptr, pattern, count);
  }

  template <typename PropT> bool has_property() const noexcept {
    return props_.template has_property<PropT>();
  }

  friend bool operator==(const queue &a, const queue &b) {
    return a.identity_ == b.identity_;
  }
  friend bool operator!=(const queue &a, const queue &b) { return !(a == b); }
  const void *paras_identity() const { return identity_.get(); }

  std::shared_ptr<const void> paras_identity_ptr() const { return identity_; }
  std::shared_ptr<detail::queue_gpu_state> paras_gpu_state() const {
    return gpu_state_;
  }

  template <typename AccT> event update_host(AccT) { return event{}; }

  void paras_set_async_handler(const async_handler &h) {
    if (h)
      async_ = std::make_shared<async_handler>(h);
  }
  void paras_attach_reporter(event &e) const {
    auto h = async_ ? async_ : ctx_.paras_async_handler();
    if (h)
      e.paras_set_reporter([h](std::vector<std::exception_ptr> errs) {
        (*h)(exception_list(std::move(errs)));
      });
  }

  void throw_asynchronous() { pending_->report_all(); }

  template <typename SrcT, typename DestT>
  auto copy(SrcT src, DestT dest)
      -> decltype(std::declval<handler &>().copy(src, dest), event{}) {
    return submit([&](handler &cgh) { cgh.copy(src, dest); });
  }
  template <typename AccT, typename T,
            typename = decltype(std::declval<AccT &>().get_pointer())>
  event fill(AccT dest, const T &pattern) {
    return submit([&](handler &cgh) { cgh.fill(dest, pattern); });
  }

  event prefetch(const void *, size_t) { return event{}; }
  event mem_advise(const void *, size_t, int) { return event{}; }

  event prefetch(const void *p, size_t n, event dep) { dep.wait(); return prefetch(p, n); }
  event prefetch(const void *p, size_t n, const std::vector<event> &deps) {
    for (auto &e : deps) e.wait();
    return prefetch(p, n);
  }
  event mem_advise(const void *p, size_t n, int a, event dep) { dep.wait(); return mem_advise(p, n, a); }
  event mem_advise(const void *p, size_t n, int a, const std::vector<event> &deps) {
    for (auto &e : deps) e.wait();
    return mem_advise(p, n, a);
  }

  template <typename CGF, typename SecondaryQ> event submit(CGF cgf, SecondaryQ &) { return submit(cgf); }

  template <typename Param> typename Param::return_type get_info() const {
    if constexpr (std::is_same_v<typename Param::return_type, context>)
      return get_context();
    else
      return get_device();
  }

  template <typename PropT> PropT get_property() const {
    if (!has_property<PropT>())
      throw exception(make_error_code(errc::invalid),
                      "queue does not have the requested property");
    return props_.template get_property<PropT>();
  }
  template <typename T>
  event fill(void *ptr, const T &pattern, size_t count) {
    return submit([&](handler &cgh) {
      cgh.fill(static_cast<T *>(ptr), pattern, count);
    });
  }

private:

  std::shared_ptr<const void> identity_ = std::make_shared<int>(0);

  std::shared_ptr<detail::queue_gpu_state> gpu_state_ =
      std::make_shared<detail::queue_gpu_state>();
  std::shared_ptr<async_handler> async_;
  std::shared_ptr<detail::pending_events> pending_ =
      std::make_shared<detail::pending_events>();

public:
  void paras_wait_pending() { pending_->wait_all(); }

  cuda_threadpool &paras_gpu_pool() const { return get_or_create_gpu_pool(); }
  void pending_report_all() { pending_->report_all(); }

private:
  mutable std::shared_ptr<cuda_threadpool> gpu_pool_{};

  cuda_threadpool &get_or_create_gpu_pool() const;

  mutable std::shared_ptr<threadpool> host_pool_{};
  threadpool &get_or_create_host_pool() const;

  context ctx_{};
  device dev_{};
  property_list props_{};

  struct async_state {
    std::mutex mtx;
    std::vector<std::exception_ptr> exceptions;
  };

  inline void rec_async_exceptn(std::exception_ptr excp);
  inline void throw_async_exceptns();

  std::shared_ptr<async_state> async_state_ = std::make_shared<async_state>();

};

inline sycl::queue::queue(const sycl::context &ctx, const sycl::device &dev,
                          const sycl::property_list &props)
    : ctx_(ctx), dev_(dev), props_(props) {}

inline sycl::queue::queue(const sycl::device &dev,
                          const sycl::property_list &props)
    : dev_(dev), props_(props) {}

inline void sycl::queue::rec_async_exceptn(std::exception_ptr ex) {
  std::lock_guard<std::mutex> lock(async_state_->mtx);
  async_state_->exceptions.push_back(std::move(ex));
}

inline void sycl::queue::throw_async_exceptns() {
  std::vector<std::exception_ptr> excps;

  {
    std::lock_guard<std::mutex> lock(async_state_->mtx);
    excps.swap(async_state_->exceptions);
  }

  for (const auto &ex : excps) {
    std::rethrow_exception(ex);
  }
}

}

#if !PARAS_GPU_BACKEND

inline void sycl::queue::wait() { paras_wait_pending(); }
inline void sycl::queue::wait_and_throw() { pending_report_all(); }
#endif

namespace std {
template <> struct hash<sycl::queue> {
  size_t operator()(const sycl::queue &q) const {
    return std::hash<const void *>()(q.paras_identity());
  }
};
template <> struct hash<sycl::event> {
  size_t operator()(const sycl::event &e) const {
    return std::hash<const void *>()(e.paras_identity());
  }
};
}

#endif
