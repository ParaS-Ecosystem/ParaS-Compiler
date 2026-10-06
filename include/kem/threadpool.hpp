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

#include "sycl/queue.hpp"

namespace sycl {
class handler;
}
#include "sycl/handler.hpp"

class threadpool {
public:
    threadpool() {};
    sycl::device get_device() const { return dev_; }
    sycl::context get_context() const { return ctx_; }

    sycl::backend get_backend() const { return sycl::backend::host; }

    bool operator==(const threadpool& rhs) const noexcept { return id_ == rhs.id_; }

    bool operator!=(const threadpool& rhs) const noexcept { return !(*this == rhs); }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    explicit threadpool(const Selector&, const sycl::property_list& props = {}) : props_(props) {}

    explicit threadpool(const sycl::property_list& props) : props_(props) {}

    threadpool(const sycl::queue& q) : ctx_(q.get_context()), dev_(q.get_device()) {}

    explicit threadpool(const sycl::async_handler&, const sycl::property_list& props = {})
        : props_(props) {}

    explicit threadpool(const sycl::device& dev, const sycl::property_list& props = {})
        : dev_(dev), props_(props) {}

    threadpool(const sycl::device& dev, const sycl::async_handler&,
               const sycl::property_list& props = {})
        : dev_(dev), props_(props) {}

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
            return false;
        }
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    threadpool(const Selector&, const sycl::async_handler&, const sycl::property_list& props = {})
        : props_(props) {}

    explicit threadpool(const sycl::context& ctx, const sycl::device& dev,
                        const sycl::property_list& props = {})
        : ctx_(ctx), dev_(dev), props_(props) {}

    static unsigned get_num_threads();

    template <typename Func>
    sycl::event spawn_1D(Func f);

    template <typename Func>
    sycl::event spawn_1D_event(Func f);

    template <typename Func>
    sycl::event spawn_ND(Func f);

    void wait() {}

    void wait_and_throw() { wait(); }

    template <typename CGF>
    sycl::event submit(CGF cgf) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        sycl::handler h(*this);
        cgf(h);
        return paras_finish_event(paras_t0);
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
        const std::uint64_t paras_t0 = paras_profiling_start();
        std::memset(ptr, value, numBytes);
        return paras_finish_event(paras_t0);
    }

    sycl::event memcpy(void* dest, const void* src, size_t numBytes) {
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
    sycl::context ctx_{};
    sycl::device dev_{};
    sycl::property_list props_{};
    std::shared_ptr<int> id_{std::make_shared<int>()};
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

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(range<dim> r, Func f) {
    if constexpr (std::is_invocable_v<const Func&, id<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](id<dim> i) { f(i, kernel_handler{}); });
        return;
    } else if constexpr (std::is_invocable_v<const Func&, item<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](item<dim> i) { f(i, kernel_handler{}); });
        return;
    } else {

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
    if constexpr (std::is_invocable_v<const Func&, nd_item<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](nd_item<dim> i) { f(i, kernel_handler{}); });
        return;
    } else {

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
    std::memset(ptr, value, num_bytes);
}

inline void handler::memcpy(void* dest, const void* src, size_t num_bytes) {
    std::memcpy(dest, src, num_bytes);
}

} // namespace sycl
#endif
