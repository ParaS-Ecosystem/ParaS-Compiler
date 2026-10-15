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

#ifndef __PARAS_GPU_THREADPOOL_HPP__
#define __PARAS_GPU_THREADPOOL_HPP__

#include "kem_gpu/hd_region.hpp"
#ifndef PARAS_HD_REGION_ACTIVE
#define PARAS_HD_REGION_ACTIVE
#define PARAS_HD_REGION_OWNER_GPU_THREADPOOL
PARAS_HD_REGION_OPEN
#endif

#include "sycl/context.hpp"
#include "sycl/kernel.hpp"
#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "sycl/event.hpp"
#include "sycl/exception.hpp"
#include "kem_gpu/offload_arch_check.hpp"
#include "sycl/id.hpp"
#include "sycl/interop_handle.hpp"
#include "sycl/item.hpp"
#include "sycl/queue.hpp"
#include "sycl/range.hpp"
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kem_gpu/gpu_runtime_api.hpp"
#include <iostream>

#include "sycl/handler.hpp"
#include "sycl/property_list.hpp"
#include "sycl/queue.hpp"
#include "utilities/selector_logic.hpp"

namespace paras_cuda_detail {

inline const cudaDeviceProp &device_props(int device) {
  static std::mutex mtx;
  static std::map<int, cudaDeviceProp> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(device);
  if (it == cache.end()) {
    cudaDeviceProp props{};
    const cudaError_t err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaGetDeviceProperties: ") +
                               cudaGetErrorString(err));
    }
    it = cache.emplace(device, props).first;
  }
  return it->second;
}

inline cudaError_t select_device(int device) noexcept {
  static const bool always = [] {
    const char *e = std::getenv("PARAS_GPU_ALWAYS_SET_DEVICE");
    return e != nullptr && e[0] == '1';
  }();
  thread_local int current = -1;
  if (!always && current == device) {
    return cudaSuccess;
  }
  const cudaError_t err = cudaSetDevice(device);
  current = (err == cudaSuccess) ? device : -1;
  return err;
}

inline void use_device(int device, const char *where) {
  const cudaError_t err = select_device(device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(err));
  }
}

inline bool sync_mode() {
  static const bool on = [] {
    const char *e = std::getenv("PARAS_GPU_SYNC");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

inline void throw_on_error(cudaError_t err, const char *where) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(err));
  }
}

inline void state_wait_until(sycl::detail::queue_gpu_state &st,
                             unsigned long long seq, void *native) {
  if (st.stream == nullptr) {
    return;
  }
  use_device(st.native_device, "cudaSetDevice before waiting for a queue");
  if (native != nullptr) {
    throw_on_error(cudaEventSynchronize(static_cast<cudaEvent_t>(native)),
                   "waiting for a command of a GPU queue");
    st.note_completed(seq);
  } else {
    const unsigned long long target = st.submitted.load();
    throw_on_error(cudaStreamSynchronize(static_cast<cudaStream_t>(st.stream)),
                   "waiting for a GPU queue");
    st.note_completed(target);
  }
}

inline bool state_is_done(sycl::detail::queue_gpu_state &st,
                          unsigned long long seq, void *native) {
  if (st.stream == nullptr) {
    return true;
  }
  use_device(st.native_device, "cudaSetDevice before querying a queue");
  cudaError_t err;
  unsigned long long target = seq;
  if (native != nullptr) {
    err = cudaEventQuery(static_cast<cudaEvent_t>(native));
  } else {
    target = st.submitted.load();
    err = cudaStreamQuery(static_cast<cudaStream_t>(st.stream));
  }
  if (err == cudaErrorNotReady) {
    (void)cudaGetLastError();
    return false;
  }
  throw_on_error(err, "querying a GPU queue");
  st.note_completed(target);
  return true;
}

inline void state_release_event(sycl::detail::queue_gpu_state &st,
                                void *native) noexcept {
  std::lock_guard<std::mutex> lock(st.mtx);
  st.free_events.push_back(native);
}

inline void *state_record_event(sycl::detail::queue_gpu_state &st,
                                unsigned long long &seq) {
  use_device(st.native_device, "cudaSetDevice before recording an event");
  std::lock_guard<std::mutex> lock(st.mtx);
  cudaEvent_t ev = nullptr;
  if (!st.free_events.empty()) {
    ev = static_cast<cudaEvent_t>(st.free_events.back());
    st.free_events.pop_back();
  } else {
    throw_on_error(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
                   "cudaEventCreateWithFlags");
  }
  const cudaError_t err =
      cudaEventRecord(ev, static_cast<cudaStream_t>(st.stream));
  if (err != cudaSuccess) {
    st.free_events.push_back(ev);
    throw_on_error(err, "cudaEventRecord");
  }
  seq = st.submitted.fetch_add(1) + 1;
  return ev;
}

inline void state_destroy(sycl::detail::queue_gpu_state &st) noexcept {
  cudaStream_t s = static_cast<cudaStream_t>(st.stream);
  st.stream = nullptr;

  if (st.native_device >= 0) {
    cudaError_t setErr = select_device(st.native_device);
    if (setErr != cudaSuccess) {
      std::cerr << "cudaSetDevice before cudaStreamDestroy failed: "
                << cudaGetErrorString(setErr) << '\n';
    }
  }

  cudaError_t syncErr = cudaStreamSynchronize(s);
  if (syncErr != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize before destroy failed: "
              << cudaGetErrorString(syncErr) << '\n';
  }

  for (void *ev : st.free_events) {
    (void)cudaEventDestroy(static_cast<cudaEvent_t>(ev));
  }
  st.free_events.clear();

  cudaError_t err = cudaStreamDestroy(s);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(err)
              << '\n';
  }
}

}

class cuda_threadpool {
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

  const void *paras_identity() const { return identity_.get(); }
  std::shared_ptr<const void> paras_identity_ptr() const { return identity_; }

  std::shared_ptr<sycl::detail::queue_gpu_state> paras_gpu_state() const {
    return st_;
  }
  friend bool operator==(const cuda_threadpool &a, const cuda_threadpool &b) {
    return a.paras_identity() == b.paras_identity();
  }
  friend bool operator!=(const cuda_threadpool &a, const cuda_threadpool &b) { return !(a == b); }
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

public:
  sycl::backend get_backend() const {
#if PARAS_HIP_BACKEND
    return sycl::backend::hip;
#elif PARAS_GPU_BACKEND
    return sycl::backend::cuda;
#else
    return sycl::backend::host;
#endif
  }

  sycl::device get_device() const { return dev_; }

  sycl::context get_context() const {
    return own_ctx_ ? ctx_
                    : sycl::detail::paras_default_context(dev_.get_platform());
  }

  cudaStream_t stream = nullptr;

  explicit cuda_threadpool(const sycl::property_list &props)
      : cuda_threadpool(sycl::default_selector_v, props) {}

  cuda_threadpool() : dev_(::paras_extension::select_device_no_selector()) {
    initialize_stream_for_device();
  }

  template <typename Selector,
            typename = std::enable_if_t<std::is_invocable_r_v<
                int, const Selector &, const sycl::device &>>>
  explicit cuda_threadpool(const Selector &sel,
                           const sycl::property_list &props = {})
      : dev_(::paras_extension::select_device_with_selector(sel)),
        props_(props) {
    initialize_stream_for_device();
  }

  template <typename Selector,
            typename = std::enable_if_t<std::is_invocable_r_v<
                int, const Selector &, const sycl::device &>>>
  cuda_threadpool(const Selector &sel, const sycl::async_handler &paras_ah,
                  const sycl::property_list &props = {})
      : cuda_threadpool(sel, props) { paras_set_async_handler(paras_ah); }

  cuda_threadpool(const sycl::queue &q)
      : dev_(q.get_device()), ctx_(q.get_context()), own_ctx_(true),
        identity_(q.paras_identity_ptr()), queue_(q),
        st_(q.paras_gpu_state()) {
    initialize_stream_for_device();
  }

  explicit cuda_threadpool(const sycl::context &ctx, const sycl::device &dev,
                           const sycl::property_list &props = {})
      : dev_(dev), ctx_(ctx), own_ctx_(true), props_(props) {
    initialize_stream_for_device();
  }

  explicit cuda_threadpool(const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : cuda_threadpool(sycl::default_selector_v, props) { paras_set_async_handler(paras_ah); }
  explicit cuda_threadpool(const sycl::device &dev, const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : cuda_threadpool(dev, props) { paras_set_async_handler(paras_ah); }
  explicit cuda_threadpool(const sycl::context &ctx, const sycl::device &dev,
                 const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : cuda_threadpool(ctx, dev, props) { paras_set_async_handler(paras_ah); }
  template <typename Selector,
              typename = std::enable_if_t<std::is_invocable_r_v<
                  int, const Selector &, const sycl::device &>>>
  explicit cuda_threadpool(const sycl::context &ctx, const Selector &selector,
                 const sycl::property_list &props = {})
      : cuda_threadpool(ctx, ::paras_extension::select_device_with_selector(selector),
              props) {}
  template <typename Selector,
              typename = std::enable_if_t<std::is_invocable_r_v<
                  int, const Selector &, const sycl::device &>>>
  explicit cuda_threadpool(const sycl::context &ctx, const Selector &selector,
                 const sycl::async_handler &paras_ah,
                 const sycl::property_list &props = {})
      : cuda_threadpool(ctx, selector, props) { paras_set_async_handler(paras_ah); }

  explicit cuda_threadpool(const sycl::device &dev,
                           const sycl::property_list &props = {})
      : dev_(dev), props_(props) {
    initialize_stream_for_device();
  }

  cuda_threadpool(const cuda_threadpool &other)
      : stream(other.stream), dev_(other.dev_), ctx_(other.ctx_),
        own_ctx_(other.own_ctx_), props_(other.props_), identity_(other.identity_),
        async_(other.async_), pending_(other.pending_), queue_(other.queue_),
        st_(other.st_) {}

  cuda_threadpool &operator=(const cuda_threadpool &other) {
    if (this == &other) {
      return *this;
    }

    stream = other.stream;
    dev_ = other.dev_;
    ctx_ = other.ctx_;
    own_ctx_ = other.own_ctx_;
    props_ = other.props_;
    identity_ = other.identity_;
    async_ = other.async_;
    pending_ = other.pending_;
    queue_ = other.queue_;
    st_ = other.st_;
    return *this;
  }

  cuda_threadpool(cuda_threadpool &&other) noexcept
      : stream(other.stream), dev_(std::move(other.dev_)),
        ctx_(std::move(other.ctx_)), own_ctx_(other.own_ctx_), props_(std::move(other.props_)),
        identity_(std::move(other.identity_)), async_(std::move(other.async_)),
        pending_(std::move(other.pending_)), queue_(std::move(other.queue_)),
        st_(std::move(other.st_)) {
    other.stream = nullptr;
  }

  cuda_threadpool &operator=(cuda_threadpool &&other) noexcept {
    if (this == &other) {
      return *this;
    }

    stream = other.stream;
    dev_ = std::move(other.dev_);
    ctx_ = std::move(other.ctx_);
    own_ctx_ = other.own_ctx_;
    props_ = std::move(other.props_);
    identity_ = std::move(other.identity_);
    async_ = std::move(other.async_);
    pending_ = std::move(other.pending_);
    queue_ = std::move(other.queue_);
    st_ = std::move(other.st_);

    other.stream = nullptr;

    return *this;
  }

  template <typename Func> sycl::event submit(Func cgf) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    const std::uint64_t t_submit = sycl::event::paras_now_ns();
    sycl::handler cgh(*this);
    cgf(cgh);
    sycl::event e = cgh.paras_take_event();
    if (props_.template has_property<sycl::property::queue::enable_profiling>())
      e.paras_start_profiling(t_submit);
    paras_attach_reporter(e);
    pending_->add(e);
    return e;
  }
  void paras_add_pending(sycl::event e) {
    paras_attach_reporter(e);
    pending_->add(e);
  }

  static unsigned gpu_get_num_threads();

  template <typename Func> sycl::event spawn_1D(Func f);

  template <typename Func> sycl::event spawn_1D_event(Func f);

  template <typename Func> sycl::event spawn_ND(Func f);

  void wait() {
    pending_->wait_all();
    ensure_stream();
    try {
      paras_cuda_detail::state_wait_until(*st_, 0, nullptr);
    } catch (const std::runtime_error &e) {
      throw sycl::exception(
          std::string("cudaStreamSynchronize in queue::wait failed: ") +
          e.what());
    }
  }

  template <typename injectCustomFunc>
  sycl::event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
    return submit([&](sycl::handler &cgh) {
      cgh.paras_request_native_event();
      cgh.parasSYCL_enqueue_custom_operation(cgf);
    });
  }

  void paras_command_enqueued(const char *where) {
    const unsigned long long seq = st_->submitted.fetch_add(1) + 1;
    if (paras_cuda_detail::sync_mode()) {
      paras_cuda_detail::throw_on_error(cudaStreamSynchronize(stream), where);
      st_->note_completed(seq);
    }
  }

  sycl::event paras_last_command_event(bool native = false) {
    sycl::event e;
    if (stream == nullptr) {
      return e;
    }
    unsigned long long seq = st_->submitted.load();
    void *ev = nullptr;
    if (native && st_->completed.load() < seq) {
      ev = paras_cuda_detail::state_record_event(*st_, seq);
    }
    e.paras_attach_gpu(st_, seq, ev);
    return e;
  }

  void paras_enqueue_dependency(const sycl::event &dep) {
    const auto ds = dep.paras_state();
    if (ds->done.valid()) {
      ds->done.wait();
    }
    const std::shared_ptr<sycl::detail::queue_gpu_state> other = ds->gpu;
    if (!other || other == st_ || other->stream == nullptr) {
      return;
    }
    if (other->completed.load() >= ds->gpu_seq.load()) {
      return;
    }
    ensure_stream();
    void *ev = ds->gpu_event.load();
    if (ev == nullptr) {
      unsigned long long seq = 0;
      void *recorded = paras_cuda_detail::state_record_event(*other, seq);
      void *expected = nullptr;
      if (ds->gpu_event.compare_exchange_strong(expected, recorded)) {

        ds->gpu_seq.store(seq);
        ev = recorded;
      } else {
        paras_cuda_detail::state_release_event(*other, recorded);
        ev = expected;
      }
    }
    paras_cuda_detail::use_device(dev_.get_native_id(),
                                  "cudaSetDevice before cudaStreamWaitEvent");
    paras_cuda_detail::throw_on_error(
        cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(ev), 0),
        "cudaStreamWaitEvent");
  }
  void paras_enqueue_dependencies(const std::vector<sycl::event> &deps) {
    for (const auto &e : deps) {
      paras_enqueue_dependency(e);
    }
  }

  void wait_and_throw() {
    wait();
    pending_->report_all();
  }

  template <typename Func> void gpu_execute_1D(const sycl::range<1> &r, Func f);

  template <typename Func> void gpu_execute_1D_async(Func f);

  template <typename Func> void gpu_execute_2D(const sycl::range<2> &r, Func f);

  template <typename Func>

  void gpu_execute_3D(const sycl::range<3> &r, Func f);

  template <typename Func>
  void gpu_execute_nd_range_1D(const sycl::nd_range<1> &r, Func f,
                               std::size_t sharedMemoryBytes = 0);

  template <typename Func>
  void gpu_execute_nd_range_2D(const sycl::nd_range<2> &r, Func f,
                               std::size_t sharedMemoryBytes = 0);

  template <typename Func>
  void gpu_execute_nd_range_3D(const sycl::nd_range<3> &r, Func f,
                               std::size_t sharedMemoryBytes = 0);

  template <typename KernelName = void, typename Func>
  sycl::event single_task(Func f) {
    sycl::detail::note_and_check_kernel<
        KernelName, sycl::detail::single_task_kernel<Func>>(get_device());
    gpu_execute_1D(sycl::range<1>(1), sycl::detail::single_task_kernel<Func>{f});
    return paras_last_command_event();
  }
  template <typename KernelName = void, typename Func>
  sycl::event single_task(sycl::event dep, Func f) {
    paras_enqueue_dependency(dep);
    return single_task<KernelName>(f);
  }
  template <typename KernelName = void, typename Func>
  sycl::event single_task(const std::vector<sycl::event> &deps, Func f) {
    paras_enqueue_dependencies(deps);
    return single_task<KernelName>(f);
  }

  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(sycl::range<dim> r, Func f) {
    sycl::detail::note_and_check_kernel<KernelName, Func>(get_device());
    if constexpr (dim == 1) {
      gpu_execute_1D(r, f);
    } else if constexpr (dim == 2) {
      gpu_execute_2D(r, f);
    } else if constexpr (dim == 3) {
      gpu_execute_3D(r, f);
    } else {
      static_assert(dim <= 3, "Only 1D/2D/3D supported");
    }
    return paras_last_command_event();
  }

  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(const sycl::nd_range<dim> &r, Func f) {
    sycl::detail::note_and_check_nd_kernel<KernelName, Func>(
        get_device(), r.get_local_range());
    if constexpr (dim == 1) {
      gpu_execute_nd_range_1D(r, f);
    } else if constexpr (dim == 2) {
      gpu_execute_nd_range_2D(r, f);
    } else if constexpr (dim == 3) {
      gpu_execute_nd_range_3D(r, f);
    } else {
      static_assert(dim <= 3, "Only 1D/2D/3D supported");
    }
    return paras_last_command_event();
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
    paras_enqueue_dependency(dep);
    return parallel_for<KernelName>(r, f);
  }
  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(sycl::range<dim> r,
                           const std::vector<sycl::event> &deps, Func f) {
    paras_enqueue_dependencies(deps);
    return parallel_for<KernelName>(r, f);
  }
  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(const sycl::nd_range<dim> &r, sycl::event dep,
                           Func f) {
    paras_enqueue_dependency(dep);
    return parallel_for<KernelName>(r, f);
  }
  template <typename KernelName = void, typename Func, int dim>
  sycl::event parallel_for(const sycl::nd_range<dim> &r,
                           const std::vector<sycl::event> &deps, Func f) {
    paras_enqueue_dependencies(deps);
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
    paras_enqueue_dependency(dep);
    return memcpy(dest, src, numBytes);
  }
  sycl::event memcpy(void *dest, const void *src, size_t numBytes,
                     const std::vector<sycl::event> &deps) {
    paras_enqueue_dependencies(deps);
    return memcpy(dest, src, numBytes);
  }
  sycl::event memset(void *ptr, int value, size_t numBytes, sycl::event dep) {
    paras_enqueue_dependency(dep);
    return memset(ptr, value, numBytes);
  }
  sycl::event memset(void *ptr, int value, size_t numBytes,
                     const std::vector<sycl::event> &deps) {
    paras_enqueue_dependencies(deps);
    return memset(ptr, value, numBytes);
  }
  template <typename T>
  sycl::event copy(const T *src, T *dest, size_t count, sycl::event dep) {
    paras_enqueue_dependency(dep);
    return copy(src, dest, count);
  }
  template <typename T>
  sycl::event copy(const T *src, T *dest, size_t count,
                   const std::vector<sycl::event> &deps) {
    paras_enqueue_dependencies(deps);
    return copy(src, dest, count);
  }
  template <typename T>
  sycl::event fill(void *ptr, const T &pattern, size_t count, sycl::event dep) {
    paras_enqueue_dependency(dep);
    return fill(ptr, pattern, count);
  }
  template <typename T>
  sycl::event fill(void *ptr, const T &pattern, size_t count,
                   const std::vector<sycl::event> &deps) {
    paras_enqueue_dependencies(deps);
    return fill(ptr, pattern, count);
  }

  template <typename T>
  sycl::event fill(void *ptr, const T &pattern, size_t count) {
    T *p = static_cast<T *>(ptr);
    const T value = pattern;
    return parallel_for<sycl::detail::paras_internal_kernel>(
        sycl::range<1>(count), [=](sycl::id<1> i) { p[i[0]] = value; });
  }

  sycl::event memcpy(void *dest, const void *src, size_t numBytes) {
    ensure_stream();
    paras_cuda_detail::use_device(dev_.get_native_id(),
                                  "cudaSetDevice before memcpy");

    cudaError_t err =
        cudaMemcpyAsync(dest, src, numBytes, cudaMemcpyDefault, stream);

    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                               cudaGetErrorString(err));
    }

    paras_command_enqueued("cudaStreamSynchronize after memcpy failed");
    return paras_last_command_event();
  }

  template <typename T> sycl::event copy(const T *src, T *dest, size_t count) {
    ensure_stream();
    paras_cuda_detail::use_device(dev_.get_native_id(),
                                  "cudaSetDevice before copy");

    cudaError_t err = cudaMemcpyAsync(
        static_cast<void *>(dest), static_cast<const void *>(src),
        count * sizeof(T), cudaMemcpyDefault, stream);

    if (err != cudaSuccess) {
      throw sycl::exception(cudaGetErrorString(err));
    }

    try {
      paras_command_enqueued("cudaStreamSynchronize after copy failed");
    } catch (const std::runtime_error &e) {
      throw sycl::exception(e.what());
    }
    return paras_last_command_event();
  }

  sycl::event memset(void *ptr, int value, size_t numBytes) {
    ensure_stream();
    paras_cuda_detail::use_device(dev_.get_native_id(),
                                  "cudaSetDevice before memset");

    cudaError_t err = cudaMemsetAsync(ptr, value, numBytes, stream);

    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemsetAsync failed: ") +
                               cudaGetErrorString(err));
    }

    paras_command_enqueued("cudaStreamSynchronize after memset failed");
    return paras_last_command_event();
  }

  ~cuda_threadpool() noexcept = default;

private:
  sycl::device dev_;
  sycl::context ctx_{};
  bool own_ctx_ = false;
  sycl::property_list props_;
  std::shared_ptr<const void> identity_ = std::make_shared<int>(0);
  std::shared_ptr<sycl::async_handler> async_;
  std::shared_ptr<sycl::detail::pending_events> pending_ =
      std::make_shared<sycl::detail::pending_events>();
  sycl::queue queue_;
  std::mutex submit_mutex_;
  std::shared_ptr<sycl::detail::queue_gpu_state> st_ =
      std::make_shared<sycl::detail::queue_gpu_state>();

  void initialize_stream_for_device() {
    stream = nullptr;

    if (!dev_.is_gpu()) {
      return;
    }

    if (!st_) {
      st_ = std::make_shared<sycl::detail::queue_gpu_state>();
    }

    std::lock_guard<std::mutex> lock(st_->mtx);
    if (st_->stream == nullptr) {
      init_cuda_for(dev_);

      cudaStream_t created = nullptr;
      cudaError_t err =
          cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
      if (err != cudaSuccess) {
        throw sycl::exception(cudaGetErrorString(err));
      }
      st_->stream = created;
      st_->native_device = dev_.get_native_id();
      st_->destroy = &paras_cuda_detail::state_destroy;
      st_->wait_until = &paras_cuda_detail::state_wait_until;
      st_->is_done = &paras_cuda_detail::state_is_done;
      st_->release_event = &paras_cuda_detail::state_release_event;
    }
    stream = static_cast<cudaStream_t>(st_->stream);
  }

  void ensure_stream() {
    if (stream == nullptr) {
      initialize_stream_for_device();
    }

    if (stream == nullptr) {
      throw std::runtime_error(
          "CUDA operation requested without a valid CUDA stream");
    }

    if (!st_->arch_checked) {
      paras_detail::check_offload_arch(dev_.get_native_id());
      st_->arch_checked = true;
    }
  }

  static void init_cuda_for(const sycl::device &d) {
    if (d.is_gpu()) {
      cudaError_t setErr =
          paras_cuda_detail::select_device(d.get_native_id());
      if (setErr != cudaSuccess) {
        throw sycl::exception(cudaGetErrorString(setErr));
      }
    }

    cudaError_t err = cudaFree(nullptr);
    if (err != cudaSuccess) {
      throw sycl::exception(cudaGetErrorString(err));
    }
  }
};

#if PARAS_HIP_BACKEND
using rocm_threadpool = cuda_threadpool;
#endif

inline cuda_threadpool &sycl::queue::get_or_create_gpu_pool() const {
  if (!gpu_pool_) {
    gpu_pool_ = std::make_shared<cuda_threadpool>(*this);
  }
  return *gpu_pool_;
}

#include "kem_gpu/gpu_threadpool_execute_1D.hpp"
#include "kem_gpu/gpu_threadpool_execute_1D_async.hpp"
#include "kem_gpu/gpu_threadpool_execute_ND.hpp"
#include "kem_gpu/gpu_threadpool_execute_common.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_1D.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_2D.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_3D.hpp"

namespace sycl {

inline void handler::memset(void *ptr, int value, size_t num_bytes) {
  if (gpu_pool_ == nullptr) {
    throw std::runtime_error("handler::memset has no CUDA backend");
  }
  paras_enqueue_dependencies();
  (void)gpu_pool_->memset(ptr, value, num_bytes);
  paras_gpu_command_enqueued();
}

inline void handler::memcpy(void *dest, const void *src, size_t num_bytes) {
  if (gpu_pool_ == nullptr) {
    throw std::runtime_error("handler::memcpy has no CUDA backend");
  }
  paras_enqueue_dependencies();
  (void)gpu_pool_->memcpy(dest, src, num_bytes);
  paras_gpu_command_enqueued();
}

inline void paras_prepare_shared_allocation(void *ptr, size_t bytes, int device) {
#if PARAS_HIP_BACKEND
  static const bool coarse = [] {
    const char *e = std::getenv("PARAS_HIP_COARSE_GRAIN_SHARED");
    return e != nullptr && e[0] == '1';
  }();
  if (ptr != nullptr && bytes != 0 && coarse) {
    if (device < 0)
      (void)hipGetDevice(&device);
    (void)hipMemAdvise(ptr, bytes, hipMemAdviseSetCoarseGrain, device);
  }
#else
  (void)ptr;
  (void)bytes;
  (void)device;
#endif
}

}
#include "kem/usm_core.hpp"
namespace sycl {

template <typename T> T *malloc_shared(size_t n, const cuda_threadpool &q) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::shared, n * sizeof(T), alignof(T), q.get_device()));
}
template <typename T> T *malloc_host(size_t n, const cuda_threadpool &q) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::host, n * sizeof(T), alignof(T), q.get_device()));
}
template <typename T> T *malloc_device(size_t n, const cuda_threadpool &q) {
  return static_cast<T *>(detail::usm_raw_alloc(usm::alloc::device, n * sizeof(T), alignof(T), q.get_device()));
}

inline void freeCudaUsmPointer(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA synchronization before SYCL USM free failed: ") +
        cudaGetErrorString(syncErr));
  }

  cudaPointerAttributes attributes{};
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    std::free(ptr);
    return;
  }

#if CUDART_VERSION >= 10000
  const cudaMemoryType memoryType = attributes.type;
#else
  const cudaMemoryType memoryType = attributes.memoryType;
#endif

  if (memoryType == cudaMemoryTypeHost) {
    err = cudaFreeHost(ptr);
  } else if (memoryType == cudaMemoryTypeDevice ||
             memoryType == cudaMemoryTypeManaged) {
    err = cudaFree(ptr);
  } else {
    throw std::runtime_error("Unsupported CUDA pointer type in sycl::free");
  }

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("SYCL USM deallocation failed: ") +
                             cudaGetErrorString(err));
  }
}

inline void free(void *ptr, const cuda_threadpool &) {
  if (!detail::usm_raw_free(ptr))
    freeCudaUsmPointer(ptr);
}

inline void *sycl::interop_handle::get_native_queue() { return backend_ptr_; }

template <>
inline sycl::backend_return_t<sycl::backend::cuda, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::cuda>() const {
  if (backend_ != sycl::backend::cuda) {
    return nullptr;
  }

#if PARAS_HIP_BACKEND
  return nullptr;
#else
  auto *pool = static_cast<cuda_threadpool *>(backend_ptr_);
  return pool ? pool->stream : nullptr;
#endif
}

template <>
inline sycl::backend_return_t<sycl::backend::host, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::host>() const {
  if (backend_ != sycl::backend::host) {
    return nullptr;
  }

  return backend_ptr_;
}

template <>
inline sycl::backend_return_t<sycl::backend::hip, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::hip>() const {
  if (backend_ != sycl::backend::hip) {
    return nullptr;
  }

#if PARAS_HIP_BACKEND
  auto *pool = static_cast<cuda_threadpool *>(backend_ptr_);
  return pool ? pool->stream : nullptr;
#else
  return nullptr;
#endif
}
}

template <typename Func> sycl::event cuda_threadpool::spawn_1D(Func f) {
  sycl::handler cgh(*this);
  f(cgh);
  sycl::event e = cgh.paras_take_event();
  paras_add_pending(e);
  return e;
}

template <typename Func> sycl::event cuda_threadpool::spawn_1D_event(Func f) {
  return spawn_1D(f);
}

template <typename Func> sycl::event cuda_threadpool::spawn_ND(Func f) {
  sycl::handler cgh(*this);
  f(cgh);
  sycl::event e = cgh.paras_take_event();
  paras_add_pending(e);
  return e;
}

#include "kem_gpu/handler_impl.hpp"

inline void sycl::queue::wait() {
  paras_wait_pending();
  if (gpu_pool_) {
    gpu_pool_->wait();
  } else if (gpu_state_->stream != nullptr) {
    paras_cuda_detail::state_wait_until(*gpu_state_, 0, nullptr);
  }
}

inline void sycl::queue::wait_and_throw() {
  if (gpu_pool_) {
    gpu_pool_->wait_and_throw();
  } else if (gpu_state_->stream != nullptr) {
    paras_cuda_detail::state_wait_until(*gpu_state_, 0, nullptr);
  }

  throw_async_exceptns();
  pending_report_all();
}
namespace std {
template <> struct hash<cuda_threadpool> {
  size_t operator()(const cuda_threadpool &q) const {
    return std::hash<const void *>()(q.paras_identity());
  }
};
}

#include "kem/usm_queue_impl.hpp"
#include "kem/reduction_impl.hpp"

inline sycl::device sycl::handler::paras_reduction_device() const {

  return gpu_pool_ ? gpu_pool_->get_device() : sycl::device{};
}
inline void sycl::handler::paras_reduction_wait() {
  if (gpu_pool_)
    gpu_pool_->wait();
}

#ifdef PARAS_HD_REGION_OWNER_GPU_THREADPOOL
PARAS_HD_REGION_CLOSE
#undef PARAS_HD_REGION_OWNER_GPU_THREADPOOL
#undef PARAS_HD_REGION_ACTIVE
#endif

#endif
