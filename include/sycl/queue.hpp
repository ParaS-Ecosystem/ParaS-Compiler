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

#include <iostream>
#include <exception>
#include <stdexcept>
#include <mutex>
#include <memory>
#include <vector>
#include "../kem_gpu/gpu_utilities.hpp"
#include "context.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "property_list.hpp"
#include "utilities/selector_logic.hpp"

class cuda_threadpool;
class rocm_threadpool;

namespace sycl {

class queue {
public:
    queue() : dev_(::paras_extension::select_device_no_selector()) {}

    explicit queue(const decltype(cpu_selector_v)& selector, const property_list& = {})
        : dev_(selector) {}
    explicit queue(const decltype(gpu_selector_v)& selector, const property_list& = {})
        : dev_(selector) {}
    explicit queue(const decltype(accelerator_selector_v)& selector, const property_list& = {})
        : dev_(selector) {}
    explicit queue(const decltype(default_selector_v)& selector, const property_list& = {})
        : dev_(selector) {}

    template <typename DeviceSelector>
    explicit queue(const DeviceSelector& selector, const property_list& = {}) : dev_(selector) {}

    template <typename DeviceSelector>
    explicit queue(const DeviceSelector& selector, const async_handler&, const property_list& = {})
        : dev_(selector) {}

    explicit queue(const property_list& props)
        : dev_(::paras_extension::select_device_no_selector()), props_(props) {}

    explicit queue(const context& ctx, const device& dev, const async_handler&,
                   const property_list& = {})
        : ctx_(ctx), dev_(dev) {}

    template <typename DeviceSelector>
    explicit queue(const context& ctx, const DeviceSelector& selector, const async_handler&,
                   const property_list& = {})
        : ctx_(ctx), dev_(selector) {}

    explicit queue(const context& ctx, const device& dev, const property_list& props = {});
    explicit queue(const device& dev, const property_list& props = {});

    template <typename CGF>
    event submit(CGF cgf) {
        if (dev_.is_gpu()) {
#if (PARAS_CUDA_BACKEND)
            handler h(get_or_create_gpu_pool());
            cgf(h);
#elif (PARAS_HIP_BACKEND)
            handler h(get_or_create_rocm_pool());
            cgf(h);
#endif
        } else {
            handler h;
            cgf(h);
        }

        return event{get_backend()};
    }

    template <typename KernelName = void, typename Func>
    event single_task(Func f) {
        return submit([&](handler& cgh) { cgh.template single_task<KernelName>(f); });
    }

    template <typename KernelName = void, typename Func>
    event single_task(event, Func f) {
        return single_task<KernelName>(f);
    }

    template <typename KernelName = void, typename Func>
    event single_task(const std::vector<event>&, Func f) {
        return single_task<KernelName>(f);
    }

    template <typename T>
    event fill(T* ptr, const T& pattern, std::size_t count) {
        return submit([&](handler& cgh) { cgh.fill(ptr, pattern, count); });
    }

    template <typename T>
    event fill(T* ptr, const T& pattern, std::size_t count, event) {
        return fill(ptr, pattern, count);
    }

    template <typename T>
    event fill(T* ptr, const T& pattern, std::size_t count, const std::vector<event>&) {
        return fill(ptr, pattern, count);
    }

    void wait();

    void wait_and_throw();

    device get_device() const { return dev_; }
    context get_context() const { return ctx_; }
    bool is_in_order() const { return props_.has_in_order(); }

    backend get_backend() const { return dev_.get_backend(); }

    bool operator==(const queue& rhs) const noexcept { return id_ == rhs.id_; }

    bool operator!=(const queue& rhs) const noexcept { return !(*this == rhs); }

    template <typename KernelName = void, typename Func, int dim>
    event parallel_for(range<dim> r, Func f) {
        return submit([&](handler& cgh) { cgh.template parallel_for<KernelName>(r, f); });
    }

    template <typename KernelName = void, typename Func, int dim>
    event parallel_for(range<dim> r, event, Func f) {
        return parallel_for<KernelName>(r, f);
    }

    template <typename KernelName = void, typename Func, int dim>
    event parallel_for(range<dim> r, const std::vector<event>&, Func f) {
        return parallel_for<KernelName>(r, f);
    }

    template <typename KernelName = void, typename Func, int dim>
    event parallel_for(const nd_range<dim>& r, Func f) {
        return submit([&](handler& cgh) { cgh.template parallel_for<KernelName>(r, f); });
    }

    template <typename KernelName = void, typename Func, int dim>
    event parallel_for(const nd_range<dim>& r, event, Func f) {
        return parallel_for<KernelName>(r, f);
    }

    template <typename KernelName = void, typename Func, int dim>
    event parallel_for(const nd_range<dim>& r, const std::vector<event>&, Func f) {
        return parallel_for<KernelName>(r, f);
    }

    template <typename injectCustomFunc>
    event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
        return this->submit(
            [&](sycl::handler& cgh) { cgh.parasSYCL_enqueue_custom_operation(cgf); });
    }

    event memset(void* ptr, int value, size_t numBytes) {
        return submit([&](handler& cgh) { cgh.memset(ptr, value, numBytes); });
    }

    event memset(void* ptr, int value, size_t numBytes, const std::vector<event>&) {
        return memset(ptr, value, numBytes);
    }
    event memset(void* ptr, int value, size_t numBytes, event) {
        return memset(ptr, value, numBytes);
    }

    template <typename T>
    event copy(const T* src, T* dest, size_t count) {
        return memcpy(dest, src, count * sizeof(T));
    }

    template <typename T>
    event copy(const T* src, T* dest, size_t count, const std::vector<event>&) {
        return copy(src, dest, count);
    }

    template <typename T>
    event copy(const T* src, T* dest, size_t count, event) {
        return copy(src, dest, count);
    }

    event memcpy(void* dest, const void* src, size_t numBytes) {
        return submit([&](handler& cgh) { cgh.memcpy(dest, src, numBytes); });
    }

    event memcpy(void* dest, const void* src, size_t numBytes, event) {
        return memcpy(dest, src, numBytes);
    }
    event memcpy(void* dest, const void* src, size_t numBytes, const std::vector<event>&) {
        return memcpy(dest, src, numBytes);
    }

    event mem_advise(const void*, size_t, int) {
        return submit([&](handler&) {});
    }
    event mem_advise(const void*, size_t, int, event) {
        return submit([&](handler&) {});
    }
    event mem_advise(const void*, size_t, int, const std::vector<event>&) {
        return submit([&](handler&) {});
    }

    event prefetch(const void*, size_t) {
        return submit([&](handler&) {});
    }
    event prefetch(const void*, size_t, event) {
        return submit([&](handler&) {});
    }
    event prefetch(const void*, size_t, const std::vector<event>&) {
        return submit([&](handler&) {});
    }

private:
#if (PARAS_CUDA_BACKEND)
    mutable std::shared_ptr<cuda_threadpool> gpu_pool_{};

    cuda_threadpool& get_or_create_gpu_pool() const;
#elif (PARAS_HIP_BACKEND)
    mutable std::shared_ptr<rocm_threadpool> rocm_pool_{};

    rocm_threadpool& get_or_create_rocm_pool() const;
#endif

    context ctx_{};
    device dev_{};

    std::shared_ptr<int> id_{std::make_shared<int>()};

    property_list props_{};

public:
    const void* paras_identity() const noexcept { return id_.get(); }

private:
    struct async_state {
        std::mutex mtx;
        std::vector<std::exception_ptr> exceptions;
    };

    inline void rec_async_exceptn(std::exception_ptr excp);
    inline void throw_async_exceptns();

    std::shared_ptr<async_state> async_state_ = std::make_shared<async_state>();
};

inline sycl::queue::queue(const sycl::context& ctx, const sycl::device& dev,
                          const sycl::property_list& props)
    : ctx_(ctx), dev_(dev), props_(props) {}

inline sycl::queue::queue(const sycl::device& dev, const sycl::property_list& props)
    : dev_(dev), props_(props) {}

inline void sycl::queue::rec_async_exceptn(std::exception_ptr ex) {
    std::lock_guard<std::mutex> lock(async_state_->mtx);
    async_state_->exceptions.push_back(std::move(ex));
}

#if !PARAS_GPU_BACKEND
inline void sycl::queue::wait() {}
#endif

inline void sycl::queue::throw_async_exceptns() {
    std::vector<std::exception_ptr> excps;

    {
        std::lock_guard<std::mutex> lock(async_state_->mtx);
        excps.swap(async_state_->exceptions);
    }

    for (const auto& ex : excps) {
        std::rethrow_exception(ex);
    }
}

#if !PARAS_GPU_BACKEND
inline void sycl::queue::wait_and_throw() {
    wait();
    throw_async_exceptns();
}
#endif

} // namespace sycl

namespace std {
template <>
struct hash<sycl::queue> {
    template <typename Queue>
    std::size_t operator()(const Queue& q) const noexcept {
        return std::hash<const void*>{}(q.paras_identity());
    }
};
} // namespace std

#endif
