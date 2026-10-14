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

#include <thread>
#include <vector>
#include <functional>
#include <cstring>
#include "sycl/id.hpp"
#include "sycl/range.hpp"
#include "sycl/item.hpp"
#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "sycl/context.hpp"
#include "sycl/event.hpp"
#include "sycl/paras_kernel_registry.hpp"

#include "sycl/queue.hpp"

namespace sycl {
class handler;
}
#include "sycl/handler.hpp"
#include "sycl/kernel.hpp" 

class threadpool {
public:
    threadpool() {};
    sycl::device get_device() const { return dev_; }
    sycl::context get_context() const {
        for (const auto& d : ctx_.get_devices())
            if (d == dev_)
                return ctx_;
        return sycl::paras_default_context(dev_.get_platform());
    }

    sycl::backend get_backend() const { return sycl::backend::host; }

    bool operator==(const threadpool& rhs) const noexcept { return id_ == rhs.id_; }

    bool operator!=(const threadpool& rhs) const noexcept { return !(*this == rhs); }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    explicit threadpool(const Selector& selector, const sycl::property_list& props = {})
        : dev_(selector), props_(props) {}

    explicit threadpool(const sycl::property_list& props) : props_(props) {}

    threadpool(const sycl::queue& q)
        : ctx_(q.get_context()), dev_(q.get_device()), props_(q.paras_props()),
          paras_async_handler_(q.paras_async_handler()) {}

    explicit threadpool(const sycl::async_handler& ah, const sycl::property_list& props = {})
        : props_(props), paras_async_handler_(paras_make_async_handler(ah)) {}

    explicit threadpool(const sycl::device& dev, const sycl::property_list& props = {})
        : dev_(dev), props_(props) {}

    threadpool(const sycl::device& dev, const sycl::async_handler& ah,
               const sycl::property_list& props = {})
        : dev_(dev), props_(props), paras_async_handler_(paras_make_async_handler(ah)) {}

    std::uint64_t paras_profiling_start() const {
        return props_.has_profiling() ? sycl::paras_now_ns() : 0;
    }
    sycl::event paras_finish_event(std::uint64_t t0) const {
        if (!props_.has_profiling()) {
            return sycl::event{};
        }
        return sycl::event::paras_profiled(t0, t0, sycl::paras_now_ns());
    }
    bool is_in_order() const { return props_.has_in_order(); }
    const void* paras_identity() const noexcept { return id_.get(); }
    template <typename Property>
    bool has_property() const noexcept {
        if constexpr (std::is_same_v<Property, sycl::property::queue::in_order>) {
            return props_.has_in_order();
        } else if constexpr (std::is_same_v<Property, sycl::property::queue::enable_profiling>) {
            return props_.has_profiling();
        } else {
            return props_.template paras_has_property<Property>();
        }
    }
   
    template <typename CGF>
    sycl::event submit(CGF cgf, const threadpool& secondaryQueue) {
        try {
            return submit(cgf);
        } catch (...) {
            threadpool secondary = secondaryQueue; // queues have reference semantics
            return secondary.submit(cgf);
        }
    }
    void throw_asynchronous() {
        std::vector<std::shared_ptr<sycl::paras_event_state>> mine;
        for (auto& st : sycl::paras_host_tasks().snapshot())
            if (st->paras_handler == paras_async_handler())
                mine.push_back(st);
        sycl::paras_report_async_errors(mine);
    }

    template <typename Param>
    typename Param::return_type get_info() const {
        if constexpr (std::is_same_v<Param, sycl::info::queue::context>)
            return get_context();
        else
            return get_device();
    }
    
    template <typename Property>
    Property get_property() const {
        return props_.template paras_get_property<Property>();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    threadpool(const Selector& selector, const sycl::async_handler& ah,
               const sycl::property_list& props = {})
        : dev_(selector), props_(props), paras_async_handler_(paras_make_async_handler(ah)) {}

    explicit threadpool(const sycl::context& ctx, const sycl::device& dev,
                        const sycl::property_list& props = {})
        : ctx_(ctx), dev_(sycl::paras_device_in_context(ctx, dev)), props_(props) {}

    threadpool(const sycl::context& ctx, const sycl::device& dev, const sycl::async_handler& ah,
               const sycl::property_list& props = {})
        : ctx_(ctx), dev_(sycl::paras_device_in_context(ctx, dev)), props_(props), paras_async_handler_(paras_make_async_handler(ah)) {}
    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    threadpool(const sycl::context& ctx, const Selector& selector,
               const sycl::property_list& props = {})
        : ctx_(ctx), dev_(sycl::paras_device_in_context(ctx, sycl::device(selector))), props_(props) {}
    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    threadpool(const sycl::context& ctx, const Selector& selector, const sycl::async_handler& ah,
               const sycl::property_list& props = {})
        : ctx_(ctx), dev_(sycl::paras_device_in_context(ctx, sycl::device(selector))), props_(props),
          paras_async_handler_(paras_make_async_handler(ah)) {}

    static unsigned get_num_threads();

    template <typename Func>
    sycl::event spawn_1D(Func f);

    template <typename Func>
    sycl::event spawn_1D_event(Func f);

    template <typename Func>
    sycl::event spawn_ND(Func f);

    void wait() { sycl::paras_drain_host_tasks(); }

    void wait_and_throw() {
        wait();
        std::vector<std::shared_ptr<sycl::paras_event_state>> mine;
        for (auto& s : sycl::paras_host_tasks().snapshot())
            if (s->paras_handler == paras_async_handler())
                mine.push_back(s);
        sycl::paras_report_async_errors(mine);
    }

    template <typename CGF>
    sycl::event submit(CGF cgf) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        sycl::handler h(*this);
        h.paras_enable_deferred_host_task();
        cgf(h);
        return h.paras_finish_host_task(paras_finish_event(paras_t0), paras_async_handler());
    }

    const std::shared_ptr<sycl::paras_async_handler_fn>& paras_async_handler() const {
        if (!(paras_async_handler_ && *paras_async_handler_) && ctx_.paras_async_handler())
            return ctx_.paras_async_handler();
        return paras_async_handler_;
    }

    template <typename Func>
    void execute_1D(const sycl::range<1>& r, Func f);

    template <typename Func>
    void execute_2D(const sycl::range<2>& r, Func f);

    template <typename Func>
    void execute_3D(const sycl::range<3>& r, Func f);

    template <typename Func>
    void execute_nd_range_1D(const sycl::nd_range<1>& r, Func f);

    template <typename Func>
    void execute_nd_range_2D(const sycl::nd_range<2>& r, Func f);

    template <typename Func>
    void execute_nd_range_3D(const sycl::nd_range<3>& r, Func f);

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(sycl::range<dim> r, Func f) {
        sycl::paras_drain_host_tasks(); 
        PARAS_REGISTER_KERNEL(KernelName, Func); 
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            execute_1D(r, f);
        } else if constexpr (dim == 2) {
            execute_2D(r, f);
        } else if constexpr (dim == 3) {
            execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(const sycl::nd_range<dim>& r, Func f) {
        sycl::paras_drain_host_tasks(); 
        PARAS_REGISTER_KERNEL(KernelName, Func); 
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            execute_nd_range_1D(r, f);
        } else if constexpr (dim == 2) {
            execute_nd_range_2D(r, f);
        } else if constexpr (dim == 3) {
            execute_nd_range_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

    sycl::event memset(void* ptr, int value, size_t numBytes) {
        sycl::paras_drain_host_tasks(); 
        const std::uint64_t paras_t0 = paras_profiling_start();
        std::memset(ptr, value, numBytes);
        return paras_finish_event(paras_t0);
    }

    sycl::event memcpy(void* dest, const void* src, size_t numBytes) {
        sycl::paras_drain_host_tasks(); 
        const std::uint64_t paras_t0 = paras_profiling_start();
        std::memcpy(dest, src, numBytes);
        return paras_finish_event(paras_t0);
    }

    template <typename T>
    sycl::event copy(const T* src, T* dest, size_t count) {
        return memcpy(dest, src, count * sizeof(T));
    }

#include "kem/queue_shortcuts.hpp"

private:
    sycl::context ctx_{sycl::paras_default_context(sycl::platform{})}; 
    sycl::device dev_{};
    sycl::property_list props_{};
    std::shared_ptr<int> id_{std::make_shared<int>()};

    std::shared_ptr<sycl::paras_async_handler_fn> paras_async_handler_{};
    static std::shared_ptr<sycl::paras_async_handler_fn>
    paras_make_async_handler(const sycl::async_handler& ah) {
        return ah ? std::make_shared<sycl::paras_async_handler_fn>(ah) : nullptr;
    }
};

#include "threadpool_spawn.hpp"
#include "threadpool_execute_common.hpp"
#include "threadpool_execute_1D.hpp"
#include "threadpool_execute_ND.hpp"
#include "threadpool_execute_3D.hpp"
#include "threadpool_execute_nd_range_1D.hpp"
#include "threadpool_execute_nd_range_2D.hpp"
#include "threadpool_execute_nd_range_3D.hpp"
namespace sycl {

template <typename KernelName>
void handler::paras_check_bound_bundle() {
    if constexpr (!std::is_void_v<KernelName>) {
        if (paras_bundle_has_) {
            const paras_kb::kernel_record* rec = paras_kb::find<KernelName>();
            if (!rec || !paras_bundle_has_(rec))
                throw sycl::exception(sycl::make_error_code(sycl::errc::kernel_not_supported),
                                      "kernel is not contained in the bound kernel bundle");
        }
    }
}

template <typename KernelName, typename Func>
void handler::paras_check_kernel_features() {
    if (!pool_ && !paras_queue_device_)
        return;
    const paras_kb::kernel_record* rec = paras_kb::registrar<KernelName, Func>::record;
    if (rec && !paras_kb::meets_requirements(*rec, pool_ ? pool_->get_device()
                                                          : paras_queue_device_()))
        throw sycl::exception(sycl::make_error_code(sycl::errc::kernel_not_supported),
                              "kernel uses optional features the device does not support");
}

inline bool handler::paras_device_has(aspect a) const {
    if (pool_)
        return pool_->get_device().has(a);
    return paras_queue_device_ ? paras_queue_device_().has(a) : false;
}

inline void handler::use_kernel_bundle(const kernel_bundle<bundle_state::executable>& execBundle) {
    if (paras_handler_spec_set_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "use_kernel_bundle after set_specialization_constant");
    if ((pool_ && pool_->get_context() != execBundle.get_context()) ||
        (!pool_ && paras_queue_context_ && paras_queue_context_() != execBundle.get_context()))
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "kernel bundle context differs from the queue context");
    paras_bundle_has_ = [execBundle](const void* rec) {
        return execBundle.has_kernel(
            paras_kb::access::make_id(static_cast<const paras_kb::kernel_record*>(rec)));
    };
    for (const auto& kv : paras_kb::access::spec(execBundle).values) {
        kv.second.apply();
        spec_resets_.push_back(kv.second.reset);
    }
}

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(range<dim> r, Func f) {
    paras_drain_host_tasks(); 
    paras_check_bound_bundle<KernelName>(); 
    paras_check_kernel_features<KernelName, Func>(); 
    if (local_memory_bytes_ > 0 && !paras_hierarchical_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::kernel_argument),
                              "local_accessor used with single_task or range parallel_for");
    if constexpr (std::is_invocable_v<const Func&, id<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](id<dim> i) { f(i, kernel_handler{}); });
        return;
    } else if constexpr (std::is_invocable_v<const Func&, item<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](item<dim> i) { f(i, kernel_handler{}); });
        return;
    } else {
        PARAS_REGISTER_KERNEL(KernelName, Func); 

        if constexpr (dim == 1) {
            pool_->execute_1D(r, f);
        } else if constexpr (dim == 2) {
            pool_->execute_2D(r, f);
        } else if constexpr (dim == 3) {
            pool_->execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
    }
}

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(const nd_range<dim>& r, Func f) {
    paras_drain_host_tasks(); 
    paras_check_bound_bundle<KernelName>(); 
    paras_check_kernel_features<KernelName, Func>();
    if constexpr (std::is_invocable_v<const Func&, nd_item<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](nd_item<dim> i) { f(i, kernel_handler{}); });
        return;
    } else {
        PARAS_REGISTER_KERNEL(KernelName, Func); 

        if constexpr (dim == 1) {
            pool_->execute_nd_range_1D(r, f);
        } else if constexpr (dim == 2) {
            pool_->execute_nd_range_2D(r, f);
        } else if constexpr (dim == 3) {
            pool_->execute_nd_range_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
    }
}

inline void handler::memset(void* ptr, int value, size_t num_bytes) {
    paras_drain_host_tasks(); 
    std::memset(ptr, value, num_bytes);
}

inline void handler::memcpy(void* dest, const void* src, size_t num_bytes) {
    paras_drain_host_tasks(); 
    std::memcpy(dest, src, num_bytes);
}

template <>
struct is_property_of<property::queue::in_order, ::threadpool> : std::true_type {};
template <>
struct is_property_of<property::queue::enable_profiling, ::threadpool> : std::true_type {};
} // namespace sycl

template <>
struct std::hash<threadpool> {
    std::size_t operator()(const threadpool& q) const noexcept {
        return std::hash<const void*>{}(q.paras_identity());
    }
};
#endif
