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
#include <string>
#include <type_traits>
#include <vector>

#include "range.hpp"
#include "id.hpp"
#include "nd_item.hpp"
#include "nd_range.hpp"
#include "item.hpp"
#include "interop_handle.hpp"
#include "event.hpp"
#include "group.hpp"
#include "reduction.hpp"
#include "spec_constant.hpp"
#include "../utilities/internal_utils.hpp"

class threadpool;
class cuda_threadpool;
class rocm_threadpool;
namespace sycl {

class handler {
    threadpool* pool_;
    [[maybe_unused]] cuda_threadpool* gpu_pool_;
    [[maybe_unused]] rocm_threadpool* rocm_pool_;
    std::vector<event> deps_;

public:
    handler() : pool_(nullptr), gpu_pool_(nullptr), rocm_pool_(nullptr), local_memory_bytes_(0) {}

    handler(threadpool& p)
        : pool_(&p), gpu_pool_(nullptr), rocm_pool_(nullptr), local_memory_bytes_(0) {}

    handler(cuda_threadpool& p)
        : pool_(nullptr), gpu_pool_(&p), rocm_pool_(nullptr), local_memory_bytes_(0) {}

    handler(rocm_threadpool& p)
        : pool_(nullptr), gpu_pool_(nullptr), rocm_pool_(&p), local_memory_bytes_(0) {}

    template <typename KernelName = void, typename Func, int dim>
    void parallel_for(range<dim> r, Func f);

    template <typename KernelName = void, typename Func, int dim>
    void parallel_for(const nd_range<dim>& r, Func f);

    template <typename KernelName = void, typename Int, typename Func,
              std::enable_if_t<std::is_integral_v<Int>, int> = 0>
    void parallel_for(Int numWorkItems, Func f) {
        this->template parallel_for<KernelName>(range<1>(static_cast<size_t>(numWorkItems)), f);
    }

    template <typename KernelName = void, typename Func>
    void single_task(Func f) {
        if constexpr (std::is_invocable_v<const Func&, kernel_handler>) {
            this->template parallel_for<KernelName>(range<1>(1),
                                                    [f](id<1>) { f(kernel_handler{}); });
        } else {
            this->template parallel_for<KernelName>(range<1>(1), [f](id<1>) { f(); });
        }
    }

    template <typename Func>
    void interop_task(Func f) {
        interop_handle ih(static_cast<void*>(pool_), sycl::backend::host);
        f(ih);
    }

    /** SYCL-CTS integration: host_task(f), required by tests/accessor_basic
     *  (`cgh.host_task([=] { ... });`). ParaS's simplified command-group
     *  model has no deferred/async task scheduling represented at the
     *  header level, so host_task runs its functor immediately —
     *  synchronously, right here — rather than queueing it for later.
     *  By the time queue::wait()/wait_and_throw() returns the task has
     *  already completed either way, which is all the CTS's host-side
     *  observability checks depend on.
    >*/
    template <typename FunctorT>
    void host_task(FunctorT&& hostFunction) {
        hostFunction();
    }

    /** SYCL-CTS integration: require(accessor), required by
     *  tests/accessor_basic (`cgh.require(acc);`). Real SYCL uses this to
     *  register an accessor's buffer dependency with the command group
     *  when the accessor wasn't already built from the (buffer, handler,
     *  ...) constructor. ParaS's accessors already hold a direct pointer
     *  into their buffer's data at construction time, so there is no
     *  separate dependency-registration bookkeeping to perform — this is
     *  a no-op that exists purely so the call compiles.
    >*/
    template <typename AccessorT>
    void require(AccessorT&) {}

    /** SYCL-CTS integration: set_specialization_constant<SpecName>(value)
     *  (SYCL-2020 4.9.5, tests/spec_constants). Writes directly into the
     *  specialization_id object's own storage -- see spec_constant.hpp's
     *  changelog entry for why (no kernel-bundle build stage to route
     *  this through at the header level). Paired with kernel_handler::
     *  get_specialization_constant<SpecName>(), which reads the same
     *  storage back from inside the kernel functor.
    >*/
    template <auto& SpecName>
    void set_specialization_constant(
        typename std::remove_reference_t<decltype(SpecName)>::value_type value) {
        paras_spec_detail::set<SpecName>(value);
        spec_resets_.push_back([] { paras_spec_detail::set<SpecName>(SpecName.value_); });
    }

    template <auto& SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type
    get_specialization_constant() const {
        return paras_spec_detail::host_value<SpecName>;
    }

    ~handler() {
        for (auto& reset : spec_resets_)
            reset();
    }
    handler(const handler&) = delete;
    handler& operator=(const handler&) = delete;

    template <typename KernelName = void, int Dim, typename Func>
    void parallel_for_work_group(range<Dim> numWorkGroups, Func f) {
        range<Dim> localRange;
        for (int d = 0; d < Dim; ++d)
            localRange[d] = 1;
        parallel_for_work_group<KernelName>(numWorkGroups, localRange, f);
    }

    template <typename KernelName = void, int Dim, typename Func>
    void parallel_for_work_group(range<Dim> numWorkGroups, range<Dim> workGroupSize, Func f) {
        this->template parallel_for<KernelName>(numWorkGroups, [=](item<Dim> it) {
            group<Dim> g(it.get_id(), numWorkGroups, workGroupSize);
            f(g);
        });
    }

    /** parallel_for with a sycl::reduction (SYCL 2020 4.9.4.2.2). Runs as an
     *  ordinary parallel_for on this handler's backend; every work-item
     *  owns a reducer and merges it into the reduction variable when done
     *  (see reduction.hpp). */
    template <typename KernelName = void, int Dim, typename T, typename BinaryOperation,
              typename KernelFunc>
    void parallel_for(range<Dim> r, reduction_impl<T, BinaryOperation> red, KernelFunc f) {
        paras_reduction_prepare(red);
        T* var = red.var_;
        const BinaryOperation op = red.op_;
        const T identity = red.identity_;
        const bool known = red.identity_known_;
        this->template parallel_for<KernelName>(r, [=](item<Dim> it) {
            reducer<T, BinaryOperation> rd(identity, known, op);
            if constexpr (std::is_invocable_v<const KernelFunc&, id<Dim>,
                                              reducer<T, BinaryOperation>&>) {
                f(it.get_id(), rd);
            } else {
                f(it, rd);
            }
            if (rd.paras_has_value())
                paras_reduction_merge(var, rd.paras_value(), op);
        });
    }

    template <typename KernelName = void, int Dim, typename T, typename BinaryOperation,
              typename KernelFunc>
    void parallel_for(const nd_range<Dim>& r, reduction_impl<T, BinaryOperation> red,
                      KernelFunc f) {
        paras_reduction_prepare(red);
        T* var = red.var_;
        const BinaryOperation op = red.op_;
        const T identity = red.identity_;
        const bool known = red.identity_known_;
        this->template parallel_for<KernelName>(r, [=](nd_item<Dim> it) {
            reducer<T, BinaryOperation> rd(identity, known, op);
            f(it, rd);
            if (rd.paras_has_value())
                paras_reduction_merge(var, rd.paras_value(), op);
        });
    }

    inline void memset(void* ptr, int value, size_t num_bytes);
    inline void memcpy(void* dest, const void* src, size_t num_bytes);

    bool isAsyncEnabled() const { return async_mode_; }

    void set_async_mode(bool mode) { async_mode_ = mode; }

    /*
     * Register one SYCL local-memory allocation.
     *
     * local_accessor objects are copied into the CUDA kernel closure. They
     * therefore store only this aligned byte offset, never a host pointer or
     * a cudaMallocManaged pointer.
     */
    template <class T>
    std::size_t local_alloc_offset(std::size_t elements) {
        static_assert(!std::is_void<T>::value, "local_accessor<void> is not supported");

        constexpr std::size_t alignment = alignof(T);
        static_assert((alignment & (alignment - 1)) == 0,
                      "local_accessor alignment must be a power of two");

        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("ParaS local_accessor allocation size overflow");
        }

        const std::size_t bytes = elements * sizeof(T);

        if (local_memory_bytes_ > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
            throw std::overflow_error("ParaS local_accessor alignment overflow");
        }

        const std::size_t aligned_offset = (local_memory_bytes_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset > std::numeric_limits<std::size_t>::max() - bytes) {
            throw std::overflow_error("ParaS local_accessor total size overflow");
        }

        local_memory_bytes_ = aligned_offset + bytes;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
        if (pool_ != nullptr && local_memory_bytes_ > paras_host_detail::kHostLocalMemBytes) {
            throw sycl::exception(
                sycl::make_error_code(sycl::errc::memory_allocation),
                "local_accessor allocations exceed the host device's local memory (" +
                    std::to_string(paras_host_detail::kHostLocalMemBytes) + " bytes)");
        }
#endif
        return aligned_offset;
    }

    std::size_t local_memory_size() const noexcept { return local_memory_bytes_; }

    void depends_on(const event& e) { deps_.push_back(e); }

    void depends_on(const std::vector<event>& events) {
        deps_.insert(deps_.end(), events.begin(), events.end());
    }

    const std::vector<event>& get_dependencies() const { return deps_; }

    void wait_for_dependencies() const { sycl::event::wait(deps_); }

    template <typename injectCustomFunc>
    void parasSYCL_enqueue_custom_operation(injectCustomFunc&& cgf) {
#if (PARAS_CUDA_BACKEND)
        if (gpu_pool_ == nullptr) {
            throw std::runtime_error("custom CUDA operation has no cuda_threadpool backend");
        }

        wait_for_dependencies();
        async_mode_ = true;
        interop_handle ih(static_cast<void*>(gpu_pool_), sycl::backend::cuda);

        try {
            cgf(ih);
            async_mode_ = false;
        } catch (...) {
            async_mode_ = false;
            throw;
        }

#elif (PARAS_HIP_BACKEND)
        if (rocm_pool_ == nullptr) {
            throw std::runtime_error("custom HIP operation has no rocm_threadpool backend");
        }

        wait_for_dependencies();
        async_mode_ = true;
        interop_handle ih(static_cast<void*>(rocm_pool_), sycl::backend::hip);

        try {
            cgf(ih);
            async_mode_ = false;
        } catch (...) {
            async_mode_ = false;
            throw;
        }
#endif
    }

    /** SYCL-CTS integration (error_log_98): fill<T>() (SYCL-2020
     *  4.9.4.2.7). Immediate host-side loop, matching every other
     *  synchronous-host command in this class.
    >*/
    template <typename T>
    void fill(T* ptr, const T& pattern, std::size_t count) {
        std::vector<T> staged(count, pattern);
        memcpy(ptr, staged.data(), count * sizeof(T));
    }

    /** SYCL-CTS integration (error_log_108): handler::copy<T>()
     *  (SYCL-2020 4.9.4.1) -- tests/usm calls parent.copy(src, dest,
     *  count, events...) directly on a handler, not through queue's
     *  own shortcut. handler had no copy at all. The trailing pack
     *  accepts (and ignores) any extra event arguments the call site
     *  forwards.
    >*/
    template <typename T, typename... Ignored>
    void copy(const T* src, T* dest, std::size_t count, Ignored&&...) {
        memcpy(dest, src, count * sizeof(T));
    }

    /** SYCL-CTS integration (error_log_112): handler::mem_advise()
     *  and handler::prefetch() -- both performance hints only in
     *  real SYCL, matching queue's own versions' no-op reasoning.
     *  The trailing pack accepts and ignores any advice/event
     *  arguments the call site forwards.
    >*/
    template <typename... Ignored>
    void mem_advise(const void*, std::size_t, Ignored&&...) {}

    template <typename... Ignored>
    void prefetch(const void*, std::size_t, Ignored&&...) {}

private:
    template <typename T, typename BinaryOperation>
    void paras_reduction_prepare(const reduction_impl<T, BinaryOperation>& red) {
        if (red.initialize_to_identity_) {
            const T identity = red.identity_;
            memcpy(red.var_, &identity, sizeof(T));
        }
    }

    std::size_t local_memory_bytes_ = 0;
    std::vector<std::function<void()>> spec_resets_;
    bool async_mode_ = false;
};

} // namespace sycl

#endif
