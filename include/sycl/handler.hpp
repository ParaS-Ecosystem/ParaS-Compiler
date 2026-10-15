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
#include <memory>
#include <thread>
#include <future>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../utilities/internal_utils.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "kernel_registry.hpp"
#include "id.hpp"
#include "interop_handle.hpp"
#include "nd_item.hpp"
#include "nd_range.hpp"
#include "range.hpp"
#include "reduction.hpp"

class threadpool;
class cuda_threadpool;

namespace sycl {

namespace detail {

template <int D, typename Func> struct hierarchical_kernel;
template <typename T> struct is_hierarchical_kernel : std::false_type {};
template <int D, typename Func>
struct is_hierarchical_kernel<hierarchical_kernel<D, Func>> : std::true_type {};
template <int D, typename Func> struct hierarchical_kernel {
  using paras_inner = Func;
  Func f;
  range<D> groups;
  range<D> local;
  PARAS_KERNEL_HD void operator()(id<D> g) const {
#if !PARAS_DEVICE_PASS

    if (!paras_host_detail::tl_local_mem) {
      thread_local std::unique_ptr<unsigned char[]> mem(
          new unsigned char[paras_host_detail::kHostLocalMemBytes]);
      paras_host_detail::tl_local_mem = mem.get();
      struct reset {
        ~reset() { paras_host_detail::tl_local_mem = nullptr; }
      } r;
      invoke_with_kh(f, group<D>(g, groups, local));
      return;
    }
#endif
    invoke_with_kh(f, group<D>(g, groups, local));
  }
};

template <typename Func> struct single_task_kernel {
  using paras_inner = Func;
  Func f;
  PARAS_KERNEL_HD void operator()(id<1>) const {
    if constexpr (std::is_invocable_v<const Func &, kernel_handler>)
      f(kernel_handler{});
    else
      f();
  }
};
}

class handler {
  threadpool *pool_;
  cuda_threadpool *gpu_pool_;
  std::vector<event> deps_;
  event result_;

public:
  handler() : pool_(nullptr), gpu_pool_(nullptr), local_memory_bytes_(0) {}

  handler(threadpool &p)
      : pool_(&p), gpu_pool_(nullptr), local_memory_bytes_(0) {}

  handler(cuda_threadpool &p)
      : pool_(nullptr), gpu_pool_(&p), local_memory_bytes_(0) {}

  template <typename KernelName = void, typename Func, int dim>
  void parallel_for(range<dim> r, Func f);
  template <typename KernelName = void, typename Func, int dim>
  void parallel_for(const nd_range<dim> &r, Func f);

  template <typename KernelName = void, typename Func>
  void parallel_for(range<2> r, Func f) {
    parallel_for<KernelName, Func, 2>(r, f);
  }
  template <typename KernelName = void, typename Func>
  void parallel_for(range<3> r, Func f) {
    parallel_for<KernelName, Func, 3>(r, f);
  }

  template <typename KernelName = void, int dim, typename First, typename... Rest,
            typename = std::enable_if_t<detail::is_reduction_v<First>>>
  void parallel_for(range<dim> r, First &&first, Rest &&...rest);
  template <typename KernelName = void, int dim, typename First, typename... Rest,
            typename = std::enable_if_t<detail::is_reduction_v<First>>>
  void parallel_for(const nd_range<dim> &r, First &&first, Rest &&...rest);

  device paras_reduction_device() const;
  void paras_reduction_wait();

  template <typename KernelName = void, typename Func>
  void single_task(Func f) {
    parallel_for<KernelName>(range<1>(1), detail::single_task_kernel<Func>{f});
  }

  template <typename KernelName = void, int Dim, typename Func>
  void parallel_for_work_group(range<Dim> numWorkGroups, Func f) {
    range<Dim> one;
    for (int d = 0; d < Dim; ++d)
      one[d] = 1;
    parallel_for_work_group<KernelName>(numWorkGroups, one, f);
  }
  template <typename KernelName = void, int Dim, typename Func>
  void parallel_for_work_group(range<Dim> numWorkGroups,
                               range<Dim> workGroupSize, Func f) {
    parallel_for<KernelName>(
        numWorkGroups,
        detail::hierarchical_kernel<Dim, Func>{f, numWorkGroups, workGroupSize});
  }

  template <typename KernelName = void, typename Func>
  void parallel_for(std::size_t numWorkItems, Func f) {
    parallel_for(range<1>(numWorkItems), f);
  }

  template <typename FunctorT> void host_task(FunctorT &&hostFunction) {
    auto promise = std::make_shared<std::promise<void>>();
    result_ = event::paras_async(promise->get_future().share(), deps_);
    auto state = result_.paras_state();
    detail::all_host_tasks().add(result_);
    std::thread([deps = deps_, fn = std::decay_t<FunctorT>(
                                   std::forward<FunctorT>(hostFunction)),
                 promise, state, pool = static_cast<void *>(pool_)]() mutable {
      detail::tl_in_host_task = true;
      sycl::event::wait(deps);
      try {
        if constexpr (std::is_invocable_v<std::decay_t<FunctorT> &,
                                          interop_handle>) {
          interop_handle ih(pool, sycl::backend::host);
          fn(ih);
        } else {
          fn();
        }
      } catch (...) {
        state->error = std::current_exception();
      }
      promise->set_value();
    }).detach();
  }

  event paras_take_event() {
    if (result_.paras_host_complete() && !result_.paras_has_wait_list())
      result_.set_dependencies(deps_);
    return result_;
  }

  void paras_enqueue_dependencies();
  void paras_gpu_command_enqueued(bool by_interop = false);
  void paras_request_native_event() { want_native_event_ = true; }

  template <auto &SpecName>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type value) {
    paras_check_no_bundle();
    spec_const_set_ = true;
    spec_restore_.remember(&SpecName);
    detail::spec_const_set_host<SpecName>(value);
  }
  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const {
    paras_check_no_bundle();
    return detail::spec_const_value_host<SpecName>();
  }

  void prefetch(const void *, std::size_t) {}
  void mem_advise(const void *, std::size_t, int) {}

  template <typename AccessorT> void require(AccessorT &&) {}
  template <typename AccessorT> void update_host(AccessorT &&) {}

  void use_kernel_bundle(const kernel_bundle<bundle_state::executable> &kb) {
    if (spec_const_set_)
      throw exception(make_error_code(errc::invalid),
                      "use_kernel_bundle after set_specialization_constant");
    kernel_bundle_used_ = true;
    bundle_ = kb.paras_impl();
    for (const auto &sv : bundle_->spec) {
      spec_restore_.remember(sv.first);
      detail::spec_const_set_raw(sv.first, sv.second);
    }
  }

  template <typename KernelName, typename Func, int D>
  void paras_launch_nd_kernel(const nd_range<D> &r) {
    paras_launch_kernel<KernelName, Func>();
#if !PARAS_DEVICE_PASS
    detail::check_nd_launch(detail::note_kernel<KernelName, Func>(),
                            r.get_local_range(), paras_reduction_device());
#else
    (void)r;
#endif
  }
  template <typename KernelName, typename Func> void paras_launch_kernel() {
#if !PARAS_DEVICE_PASS
    const detail::kernel_info *ki = detail::note_kernel<KernelName, Func>();
    if (ki != nullptr)
      detail::kernel_launch_check(ki, bundle_.get(), paras_reduction_device());
#endif
  }

  template <typename Func, int dim>
  void paras_check_reqd_work_group_size(const nd_range<dim> &r) const {
#if !PARAS_DEVICE_PASS
    const detail::kernel_props p = detail::kernel_props_of<Func>::get();
    if (p.wg_dims == 0)
      return;
    bool match = p.wg_dims == dim;
    for (int i = 0; match && i < dim; ++i)
      match = r.get_local_range()[i] == p.wg[i];
    if (!match)
      throw exception(make_error_code(errc::nd_range),
                      "the nd_range work-group size does not match the "
                      "kernel's [[sycl::reqd_work_group_size]]");
#else
    (void)r;
#endif
  }

  template <typename Func> void paras_check_range_kernel() const {
    if constexpr (!detail::is_hierarchical_kernel<Func>::value)
      if (has_local_acc_)
        throw exception(make_error_code(errc::kernel_argument),
                        "local_accessor used in a single_task or a range "
                        "parallel_for");
  }
  void paras_check_no_bundle() const {
    if (kernel_bundle_used_)
      throw exception(make_error_code(errc::invalid),
                      "specialization constant access after use_kernel_bundle");
  }
  template <typename T> void set_arg(int, T &&) {}
  template <typename... Ts> void set_args(Ts &&...) {}

  template <typename T> void copy(const T *src, T *dest, std::size_t count) {
    memcpy(dest, src, count * sizeof(T));
  }
  template <typename T>
  void fill(T *ptr, const T &pattern, std::size_t count) {
    const T value = pattern;
    parallel_for<detail::paras_internal_kernel>(
        range<1>(count), [=](id<1> i) { ptr[i[0]] = value; });
  }

  template <typename AccessorT, typename T,
            typename = decltype(std::declval<AccessorT &>().get_pointer())>
  void copy(const AccessorT &src, T *dest) {
    const auto n = src.size();
    for (std::size_t i = 0; i < n; ++i)
      dest[i] = src.begin()[i];
  }
  template <typename T, typename AccessorT,
            typename = decltype(std::declval<AccessorT &>().get_pointer())>
  void copy(const T *src, const AccessorT &dest) {
    const auto n = dest.size();
    for (std::size_t i = 0; i < n; ++i)
      dest.begin()[i] = src[i];
  }

  template <typename AccessorT, typename T,
            typename = decltype(std::declval<AccessorT &>().get_pointer())>
  void copy(const AccessorT &src, std::shared_ptr<T> dest) {
    copy(src, dest.get());
  }
  template <typename T, typename AccessorT,
            typename = decltype(std::declval<AccessorT &>().get_pointer())>
  void copy(std::shared_ptr<T> src, const AccessorT &dest) {
    copy(static_cast<const T *>(src.get()), dest);
  }

  template <typename SrcAccT, typename DestAccT,
            typename = decltype(std::declval<SrcAccT &>().get_pointer()),
            typename = decltype(std::declval<DestAccT &>().get_pointer())>
  void copy(const SrcAccT &src, const DestAccT &dest) {

    if (dest.size() < src.size())
      throw exception(make_error_code(errc::invalid),
                      "handler::copy: destination accessor is smaller than "
                      "the source accessor");
    const auto n = src.size();
    for (std::size_t i = 0; i < n; ++i)
      dest.begin()[i] = src.begin()[i];
  }

  template <typename AccessorT, typename T,
            typename = decltype(std::declval<AccessorT &>().get_pointer())>
  void fill(const AccessorT &dest, const T &pattern) {
    const auto n = dest.size();
    for (std::size_t i = 0; i < n; ++i)
      dest.begin()[i] = pattern;
  }

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
    has_local_acc_ = true;
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

    paras_enqueue_dependencies();
    async_mode_ = true;
    interop_handle ih(static_cast<void *>(gpu_pool_),
                      PARAS_HIP_BACKEND ? sycl::backend::hip
                                        : sycl::backend::cuda);

    try {
      cgf(ih);
      async_mode_ = false;
    } catch (...) {
      async_mode_ = false;
      throw;
    }

    paras_gpu_command_enqueued( true);
  }

private:
  std::size_t local_memory_bytes_ = 0;
  bool async_mode_ = false;
  bool want_native_event_ = false;
  bool kernel_bundle_used_ = false;
  std::shared_ptr<detail::bundle_impl> bundle_;
  bool spec_const_set_ = false;
  bool has_local_acc_ = false;

  detail::spec_const_restorer spec_restore_;
};

}

#endif
