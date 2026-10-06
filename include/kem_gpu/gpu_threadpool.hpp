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

#include "sycl/context.hpp"
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
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <iostream>

#include "sycl/handler.hpp"
#include "sycl/queue.hpp"
#include "utilities/selector_logic.hpp"

#include "gpu_utilities.hpp"

#if (PARAS_CUDA_BACKEND)
#include <cuda_runtime.h>

class cuda_threadpool {

public:
    sycl::backend get_backend() const {
#if PARAS_GPU_BACKEND
        return sycl::backend::cuda;
#else
        return sycl::backend::host;
#endif
    }

    sycl::device get_device() const { return dev_; }
    sycl::context get_context() const { return ctx_; }

    cudaStream_t stream = nullptr;
    bool ownsStream = false;

    cuda_threadpool() : dev_(::paras_extension::select_device_no_selector()) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    explicit cuda_threadpool(const Selector& sel, const sycl::property_list& props = {})
        : dev_(::paras_extension::select_device_with_selector(sel)), props_(props) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    cuda_threadpool(const Selector& sel, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : cuda_threadpool(sel, props) {}

    explicit cuda_threadpool(const sycl::property_list& props)
        : dev_(::paras_extension::select_device_no_selector()), props_(props) {
        initialize_stream_for_device();
    }

    explicit cuda_threadpool(const sycl::async_handler&, const sycl::property_list& props = {})
        : cuda_threadpool(props) {}

    cuda_threadpool(const sycl::device& dev, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : cuda_threadpool(dev, props) {}

    bool operator==(const cuda_threadpool& rhs) const noexcept { return id_ == rhs.id_; }
    bool operator!=(const cuda_threadpool& rhs) const noexcept { return !(*this == rhs); }

    cuda_threadpool(const sycl::queue& q) : dev_(q.get_device()), ctx_(q.get_context()), queue_(q) {
        initialize_stream_for_device();
    }

    explicit cuda_threadpool(const sycl::context& ctx, const sycl::device& dev,
                             const sycl::property_list& props = {})
        : dev_(dev), ctx_(ctx), props_(props) {
        initialize_stream_for_device();
    }

    explicit cuda_threadpool(const sycl::device& dev, const sycl::property_list& props = {})
        : dev_(dev), props_(props) {
        initialize_stream_for_device();
    }

    cuda_threadpool(const cuda_threadpool& other)
        : dev_(other.dev_), ctx_(other.ctx_), props_(other.props_), queue_(other.queue_),
          id_(other.id_) {
        initialize_stream_for_device();
    }

    cuda_threadpool& operator=(const cuda_threadpool& other) {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        dev_ = other.dev_;
        ctx_ = other.ctx_;
        props_ = other.props_;
        queue_ = other.queue_;
        id_ = other.id_;

        initialize_stream_for_device();
        return *this;
    }

    cuda_threadpool(cuda_threadpool&& other) noexcept
        : stream(other.stream), ownsStream(other.ownsStream), dev_(std::move(other.dev_)),
          ctx_(std::move(other.ctx_)), props_(std::move(other.props_)),
          queue_(std::move(other.queue_)), id_(other.id_) {
        other.stream = nullptr;
        other.ownsStream = false;
    }

    cuda_threadpool& operator=(cuda_threadpool&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        stream = other.stream;
        ownsStream = other.ownsStream;
        dev_ = std::move(other.dev_);
        ctx_ = std::move(other.ctx_);
        props_ = std::move(other.props_);
        queue_ = std::move(other.queue_);
        id_ = other.id_;

        other.stream = nullptr;
        other.ownsStream = false;

        return *this;
    }

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

    template <typename Func>
    void submit(Func cgf) {
        sycl::handler cgh(*this);
        cgf(cgh);
    }

    static unsigned gpu_get_num_threads();

    template <typename Func>
    sycl::event spawn_1D(Func f);

    template <typename Func>
    sycl::event spawn_1D_event(Func f);

    template <typename Func>
    void spawn_ND(Func f);

    void wait() {
        ensure_stream();
        cudaError_t errSync = cudaStreamSynchronize(stream);
        if (errSync != cudaSuccess) {
            throw sycl::exception(std::string("cudaStreamSynchronize in queue::wait failed: ") +
                                  cudaGetErrorString(errSync));
        }
    }

    void wait_and_throw() { wait(); }

    template <typename injectCustomFunc>
    sycl::event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
        sycl::handler cgh(*this);
        cgh.parasSYCL_enqueue_custom_operation(std::move(cgf));
        return sycl::event{};
    }

    void wait() const {
        if (stream == nullptr) {
            throw sycl::exception("cuda_threadpool::wait called with null CUDA stream");
        }

        cudaError_t errSync = cudaStreamSynchronize(stream);
        if (errSync != cudaSuccess) {
            throw sycl::exception(std::string("cudaStreamSynchronize in queue::wait failed: ") +
                                  cudaGetErrorString(errSync));
        }
    }

    void wait_and_throw() const { wait(); }

    template <typename Func>
    void gpu_execute_1D(const sycl::range<1>& r, Func f);

    template <typename Func>
    void gpu_execute_1D_async(Func f);

    template <typename Func>
    void gpu_execute_2D(const sycl::range<2>& r, Func f);

    template <typename Func>
    void gpu_execute_3D(const sycl::range<3>& r, Func f);

    template <typename Func>
    void gpu_execute_nd_range_1D(const sycl::nd_range<1>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_2D(const sycl::nd_range<2>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_3D(const sycl::nd_range<3>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(sycl::range<dim> r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(const sycl::nd_range<dim>& r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_nd_range_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_nd_range_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_nd_range_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

#include "kem/queue_shortcuts.hpp"

    sycl::event memcpy(void* dest, const void* src, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        cudaError_t err = cudaMemcpyAsync(dest, src, numBytes, cudaMemcpyDefault, stream);

        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                                     cudaGetErrorString(err));
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize after memcpy failed: ") +
                                     cudaGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    template <typename T>
    sycl::event copy(const T* src, T* dest, size_t count) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        cudaError_t err = cudaMemcpyAsync(static_cast<void*>(dest), static_cast<const void*>(src),
                                          count * sizeof(T), cudaMemcpyDefault, stream);

        if (err != cudaSuccess) {
            throw sycl::exception(cudaGetErrorString(err));
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw sycl::exception(std::string("cudaStreamSynchronize after copy failed: ") +
                                  cudaGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    sycl::event memset(void* ptr, int value, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        cudaError_t err = cudaMemsetAsync(ptr, value, numBytes, stream);

        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemsetAsync failed: ") +
                                     cudaGetErrorString(err));
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize after memset failed: ") +
                                     cudaGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    ~cuda_threadpool() noexcept { release_stream_noexcept(); }

private:
    sycl::device dev_;
    sycl::context ctx_{dev_};
    sycl::property_list props_;
    sycl::queue queue_;
    std::shared_ptr<int> id_{std::make_shared<int>()};

    void initialize_stream_for_device() {
        stream = nullptr;
        ownsStream = false;

        if (!dev_.is_gpu()) {
            return;
        }

        init_cuda_for(dev_);
        create_stream();
        ownsStream = true;
    }

    void ensure_stream() {
        if (stream == nullptr) {
            initialize_stream_for_device();
        }

        if (stream == nullptr) {
            throw std::runtime_error("CUDA operation requested without a valid CUDA stream");
        }

        if (dev_.is_gpu()) {
            const cudaError_t setErr = cudaSetDevice(dev_.get_native_id());
            if (setErr != cudaSuccess) {
                throw sycl::exception(cudaGetErrorString(setErr));
            }
        }

        if (!arch_checked_) {
            paras_detail::check_offload_arch_cuda(dev_.get_native_id());
            arch_checked_ = true;
        }
    }
    bool arch_checked_ = false;

    void create_stream() {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        if (err != cudaSuccess) {
            stream = nullptr;
            ownsStream = false;
            throw sycl::exception(cudaGetErrorString(err));
        }
    }

    void release_stream_noexcept() noexcept {
        if (!ownsStream || stream == nullptr) {
            stream = nullptr;
            ownsStream = false;
            return;
        }

        if (dev_.is_gpu()) {
            cudaError_t setErr = cudaSetDevice(dev_.get_native_id());
            if (setErr != cudaSuccess) {
                std::cerr << "cudaSetDevice before cudaStreamDestroy failed: "
                          << cudaGetErrorString(setErr) << '\n';
            }
        }

        cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            std::cerr << "cudaStreamSynchronize before destroy failed: "
                      << cudaGetErrorString(syncErr) << '\n';
        }

        cudaError_t err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(err) << '\n';
        }

        stream = nullptr;
        ownsStream = false;
    }

    static void init_cuda_for(const sycl::device& d) {
        if (d.is_gpu()) {
            cudaError_t setErr = cudaSetDevice(d.get_native_id());
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

inline cuda_threadpool& sycl::queue::get_or_create_gpu_pool() const {
    if (!gpu_pool_) {
        gpu_pool_ = std::make_shared<cuda_threadpool>(*this);
    }
    return *gpu_pool_;
}

#elif (PARAS_HIP_BACKEND)
#include <hip/hip_runtime.h>

class rocm_threadpool {

public:
    sycl::backend get_backend() const {
#if PARAS_HIP_BACKEND
        return sycl::backend::hip;
#else
        return sycl::backend::host;
#endif
    }

    sycl::device get_device() const { return dev_; }
    sycl::context get_context() const { return ctx_; }

    hipStream_t stream = nullptr;
    bool ownsStream = false;

    rocm_threadpool() : dev_(::paras_extension::select_device_no_selector()) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    explicit rocm_threadpool(const Selector& sel, const sycl::property_list& props = {})
        : dev_(::paras_extension::select_device_with_selector(sel)), props_(props) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    rocm_threadpool(const Selector& sel, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : rocm_threadpool(sel, props) {}

    explicit rocm_threadpool(const sycl::property_list& props)
        : dev_(::paras_extension::select_device_no_selector()), props_(props) {
        initialize_stream_for_device();
    }

    explicit rocm_threadpool(const sycl::async_handler&, const sycl::property_list& props = {})
        : rocm_threadpool(props) {}

    rocm_threadpool(const sycl::device& dev, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : rocm_threadpool(dev, props) {}

    bool operator==(const rocm_threadpool& rhs) const noexcept { return id_ == rhs.id_; }
    bool operator!=(const rocm_threadpool& rhs) const noexcept { return !(*this == rhs); }

    rocm_threadpool(const sycl::queue& q) : dev_(q.get_device()), ctx_(q.get_context()), queue_(q) {
        initialize_stream_for_device();
    }

    explicit rocm_threadpool(const sycl::context& ctx, const sycl::device& dev,
                             const sycl::property_list& props = {})
        : dev_(dev), ctx_(ctx), props_(props) {
        initialize_stream_for_device();
    }

    explicit rocm_threadpool(const sycl::device& dev, const sycl::property_list& props = {})
        : dev_(dev), props_(props) {
        initialize_stream_for_device();
    }

    rocm_threadpool(const rocm_threadpool& other)
        : dev_(other.dev_), ctx_(other.ctx_), props_(other.props_), queue_(other.queue_),
          id_(other.id_) {
        initialize_stream_for_device();
    }

    rocm_threadpool& operator=(const rocm_threadpool& other) {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        dev_ = other.dev_;
        ctx_ = other.ctx_;
        props_ = other.props_;
        queue_ = other.queue_;
        id_ = other.id_;

        initialize_stream_for_device();
        return *this;
    }

    rocm_threadpool(rocm_threadpool&& other) noexcept
        : stream(other.stream), ownsStream(other.ownsStream), dev_(std::move(other.dev_)),
          ctx_(std::move(other.ctx_)), props_(std::move(other.props_)),
          queue_(std::move(other.queue_)), id_(other.id_) {
        other.stream = nullptr;
        other.ownsStream = false;
    }

    rocm_threadpool& operator=(rocm_threadpool&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        stream = other.stream;
        ownsStream = other.ownsStream;
        dev_ = std::move(other.dev_);
        ctx_ = std::move(other.ctx_);
        props_ = std::move(other.props_);
        queue_ = std::move(other.queue_);
        id_ = other.id_;

        other.stream = nullptr;
        other.ownsStream = false;

        return *this;
    }

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

    template <typename Func>
    void submit(Func cgf) {
        sycl::handler cgh(*this);
        cgf(cgh);
    }

    static unsigned gpu_get_num_threads();

    template <typename Func>
    sycl::event spawn_1D(Func f);

    template <typename Func>
    sycl::event spawn_1D_event(Func f);

    template <typename Func>
    void spawn_ND(Func f);

    void wait() {
        ensure_stream();
        hipError_t errSync = hipStreamSynchronize(stream);
        if (errSync != hipSuccess) {
            throw sycl::exception(std::string("hipStreamSynchronize in queue::wait failed: ") +
                                  hipGetErrorString(errSync));
        }
    }

    void wait_and_throw() { wait(); }

    template <typename injectCustomFunc>
    sycl::event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
        sycl::handler cgh(*this);
        cgh.parasSYCL_enqueue_custom_operation(std::move(cgf));
        return sycl::event{};
    }

    void wait() const {
        if (stream == nullptr) {
            throw sycl::exception("rocm_threadpool::wait called with null HIP stream");
        }

        hipError_t errSync = hipStreamSynchronize(stream);
        if (errSync != hipSuccess) {
            throw sycl::exception(std::string("hipStreamSynchronize in queue::wait failed: ") +
                                  hipGetErrorString(errSync));
        }
    }

    void wait_and_throw() const { wait(); }

    template <typename Func>
    void gpu_execute_1D(const sycl::range<1>& r, Func f);

    template <typename Func>
    void gpu_execute_1D_async(Func f);

    template <typename Func>
    void gpu_execute_2D(const sycl::range<2>& r, Func f);
    template <typename Func>
    void gpu_execute_3D(const sycl::range<3>& r, Func f);

    template <typename Func>
    void gpu_execute_nd_range_1D(const sycl::nd_range<1>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_2D(const sycl::nd_range<2>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_3D(const sycl::nd_range<3>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(sycl::range<dim> r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(const sycl::nd_range<dim>& r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_nd_range_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_nd_range_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_nd_range_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

#include "kem/queue_shortcuts.hpp"

    sycl::event memcpy(void* dest, const void* src, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        hipError_t err = hipMemcpyAsync(dest, src, numBytes, hipMemcpyDefault, stream);

        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipMemcpyAsync failed: ") +
                                     hipGetErrorString(err));
        }

        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipStreamSynchronize after memcpy failed: ") +
                                     hipGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    template <typename T>
    sycl::event copy(const T* src, T* dest, size_t count) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        hipError_t err = hipMemcpyAsync(static_cast<void*>(dest), static_cast<const void*>(src),
                                        count * sizeof(T), hipMemcpyDefault, stream);

        if (err != hipSuccess) {
            throw sycl::exception(hipGetErrorString(err));
        }

        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            throw sycl::exception(std::string("hipStreamSynchronize after copy failed: ") +
                                  hipGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    sycl::event memset(void* ptr, int value, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        hipError_t err = hipMemsetAsync(ptr, value, numBytes, stream);

        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipMemsetAsync failed: ") +
                                     hipGetErrorString(err));
        }

        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipStreamSynchronize after memset failed: ") +
                                     hipGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    ~rocm_threadpool() noexcept { release_stream_noexcept(); }

private:
    sycl::device dev_;
    sycl::context ctx_{dev_};
    sycl::property_list props_;
    sycl::queue queue_;
    std::shared_ptr<int> id_{std::make_shared<int>()};

    void initialize_stream_for_device() {
        stream = nullptr;
        ownsStream = false;

        if (!dev_.is_gpu()) {
            return;
        }

        init_rocm_for(dev_);
        create_stream();
        ownsStream = true;
    }

    void ensure_stream() {
        if (stream == nullptr) {
            initialize_stream_for_device();
        }

        if (stream == nullptr) {
            throw std::runtime_error("HIP operation requested without a valid HIP stream");
        }

        if (dev_.is_gpu()) {
            const hipError_t setErr = hipSetDevice(dev_.get_native_id());
            if (setErr != hipSuccess) {
                throw sycl::exception(hipGetErrorString(setErr));
            }
        }

        if (!arch_checked_) {
            paras_detail::check_offload_arch_hip(dev_.get_native_id());
            arch_checked_ = true;
        }
    }
    bool arch_checked_ = false;

    void create_stream() {
        hipError_t err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

        if (err != hipSuccess) {
            stream = nullptr;
            ownsStream = false;
            throw sycl::exception(hipGetErrorString(err));
        }
    }

    void release_stream_noexcept() noexcept {
        if (!ownsStream || stream == nullptr) {
            stream = nullptr;
            ownsStream = false;
            return;
        }

        if (dev_.is_gpu()) {
            hipError_t setErr = hipSetDevice(dev_.get_native_id());
            if (setErr != hipSuccess) {
                std::cerr << "hipSetDevice before hipStreamDestroy failed: "
                          << hipGetErrorString(setErr) << '\n';
            }
        }

        hipError_t syncErr = hipStreamSynchronize(stream);
        if (syncErr != hipSuccess) {
            std::cerr << "hipStreamSynchronize before destroy failed: "
                      << hipGetErrorString(syncErr) << '\n';
        }

        hipError_t err = hipStreamDestroy(stream);
        if (err != hipSuccess) {
            std::cerr << "hipStreamDestroy failed: " << hipGetErrorString(err) << '\n';
        }

        stream = nullptr;
        ownsStream = false;
    }

    static void init_rocm_for(const sycl::device& d) {
        if (d.is_gpu()) {
            hipError_t setErr = hipSetDevice(d.get_native_id());
            if (setErr != hipSuccess) {
                throw sycl::exception(hipGetErrorString(setErr));
            }
        }

        hipError_t err = hipFree(nullptr);
        if (err != hipSuccess) {
            throw sycl::exception(hipGetErrorString(err));
        }
    }
};

inline rocm_threadpool& sycl::queue::get_or_create_rocm_pool() const {
    if (!rocm_pool_) {
        rocm_pool_ = std::make_shared<rocm_threadpool>(*this);
    }
    return *rocm_pool_;
}

#endif

#include "kem_gpu/gpu_threadpool_execute_1D.hpp"
#include "kem_gpu/gpu_threadpool_execute_1D_async.hpp"
#include "kem_gpu/gpu_threadpool_execute_ND.hpp"
#include "kem_gpu/gpu_threadpool_execute_common.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_1D.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_2D.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_3D.hpp"

namespace sycl {

inline void handler::memset(void* ptr, int value, size_t num_bytes) {
#if PARAS_CUDA_BACKEND
    if (gpu_pool_ == nullptr) {
        throw std::runtime_error("handler::memset has no CUDA backend");
    }
    wait_for_dependencies();
    (void)gpu_pool_->memset(ptr, value, num_bytes);
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ == nullptr) {
        throw std::runtime_error("handler::memset has no HIP backend");
    }
    wait_for_dependencies();
    (void)rocm_pool_->memset(ptr, value, num_bytes);
#endif
}

inline void handler::memcpy(void* dest, const void* src, size_t num_bytes) {
#if PARAS_CUDA_BACKEND
    if (gpu_pool_ == nullptr) {
        throw std::runtime_error("handler::memcpy has no CUDA backend");
    }
    wait_for_dependencies();
    (void)gpu_pool_->memcpy(dest, src, num_bytes);
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ == nullptr) {
        throw std::runtime_error("handler::memcpy has no HIP backend");
    }
    wait_for_dependencies();
    (void)rocm_pool_->memcpy(dest, src, num_bytes);
#endif
}

inline void* sycl::interop_handle::get_native_queue() {
    return backend_ptr_;
}

template <>
inline sycl::backend_return_t<sycl::backend::cuda, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::cuda>() const {
    if (backend_ != sycl::backend::cuda) {
        return nullptr;
    }
#if (PARAS_CUDA_BACKEND)
    auto* pool = static_cast<cuda_threadpool*>(backend_ptr_);
    return pool ? pool->stream : nullptr;
#elif (PARAS_HIP_BACKEND)
    return static_cast<sycl::backend_return_t<sycl::backend::cuda, sycl::queue>>(backend_ptr_);
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
#if (PARAS_CUDA_BACKEND)
    return backend_ptr_;
#elif (PARAS_HIP_BACKEND)
    auto* pool = static_cast<rocm_threadpool*>(backend_ptr_);
    return pool ? pool->stream : nullptr;
#endif
}
} // namespace sycl

#if (PARAS_CUDA_BACKEND)
template <typename Func>
sycl::event cuda_threadpool::spawn_1D(Func f) {
    const std::uint64_t paras_t0 = paras_profiling_start();

    sycl::handler cgh(*this);
    f(cgh);

    return paras_finish_event(paras_t0);
}

template <typename Func>
sycl::event cuda_threadpool::spawn_1D_event(Func f) {

    spawn_1D(f);
    return sycl::event();
}

template <typename Func>
void cuda_threadpool::spawn_ND(Func f) {
    sycl::handler cgh(*this); // Currently same as 1D; modify for ND ranges
    f(cgh);
}
#elif (PARAS_HIP_BACKEND)
template <typename Func>
sycl::event rocm_threadpool::spawn_1D(Func f) {
    const std::uint64_t paras_t0 = paras_profiling_start();

    sycl::handler cgh(*this);
    f(cgh);

    return paras_finish_event(paras_t0);
}

template <typename Func>
sycl::event rocm_threadpool::spawn_1D_event(Func f) {

    spawn_1D(f);
    return sycl::event();
}

template <typename Func>
void rocm_threadpool::spawn_ND(Func f) {
    sycl::handler cgh(*this); // Currently same as 1D; modify for ND ranges
    f(cgh);
}
#endif

#include "kem_gpu/handler_impl.hpp"

inline void sycl::queue::wait() {
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_) {
        gpu_pool_->wait();
    }
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_) {
        rocm_pool_->wait();
    }
#endif
}

inline void sycl::queue::wait_and_throw() {
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_) {
        gpu_pool_->wait_and_throw();
    }

    throw_async_exceptns();
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_) {
        rocm_pool_->wait_and_throw();
    }

    throw_async_exceptns();
#endif
}

#endif
