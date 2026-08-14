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

#include <cuda_runtime.h>
#include <iostream>

#include "sycl/handler.hpp"
#include "sycl/queue.hpp"
#include "utilities/selector_logic.hpp"

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

  template <typename Selector>
  explicit cuda_threadpool(const Selector &sel)
      : dev_(::paras_extension::select_device_with_selector(sel)) {
    initialize_stream_for_device();
  }

  cuda_threadpool(const sycl::queue &q)
      : dev_(q.get_device()), ctx_(q.get_context()), queue_(q) {
    initialize_stream_for_device();
  }

  explicit cuda_threadpool(const sycl::context &ctx, const sycl::device &dev,
                           const sycl::property_list &props = {})
      : dev_(dev), ctx_(ctx), props_(props) {
    initialize_stream_for_device();
  }

  explicit cuda_threadpool(const sycl::device &dev,
                           const sycl::property_list &props = {})
      : dev_(dev), props_(props) {
    initialize_stream_for_device();
  }

  cuda_threadpool(const cuda_threadpool &other)
      : dev_(other.dev_), ctx_(other.ctx_), props_(other.props_),
        queue_(other.queue_) {
    initialize_stream_for_device();
  }

  cuda_threadpool &operator=(const cuda_threadpool &other) {
    if (this == &other) {
      return *this;
    }

    release_stream_noexcept();

    dev_ = other.dev_;
    ctx_ = other.ctx_;
    props_ = other.props_;
    queue_ = other.queue_;

    initialize_stream_for_device();
    return *this;
  }

  cuda_threadpool(cuda_threadpool &&other) noexcept
      : stream(other.stream), ownsStream(other.ownsStream),
        dev_(std::move(other.dev_)), ctx_(std::move(other.ctx_)),
        props_(std::move(other.props_)), queue_(std::move(other.queue_)) {
    other.stream = nullptr;
    other.ownsStream = false;
  }

  cuda_threadpool &operator=(cuda_threadpool &&other) noexcept {
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

    other.stream = nullptr;
    other.ownsStream = false;

    return *this;
  }

  template <typename Func> void submit(Func cgf) {
    sycl::handler cgh(*this);
    cgf(cgh);
  }

  static unsigned gpu_get_num_threads();

  template <typename Func> sycl::event spawn_1D(Func f);

  template <typename Func> sycl::event spawn_1D_event(Func f);

  template <typename Func> void spawn_ND(Func f);

  void wait() {
    ensure_stream();
    cudaError_t errSync = cudaStreamSynchronize(stream);
    if (errSync != cudaSuccess) {
      throw sycl::exception(
          std::string("cudaStreamSynchronize in queue::wait failed: ") +
          cudaGetErrorString(errSync));
    }
  }

  void wait_and_throw() { wait(); }

  template <typename Func> void gpu_execute_1D(const sycl::range<1> &r, Func f);

  template <typename Func> void gpu_execute_1D_async(Func f);

  template <typename Func> void gpu_execute_2D(const sycl::range<2> &r, Func f);

  template <typename Func>
  void gpu_execute_nd_range_1D(const sycl::nd_range<1> &r, Func f,
                               std::size_t sharedMemoryBytes = 0);

  template <typename Func>
  void gpu_execute_nd_range_2D(const sycl::nd_range<2> &r, Func f,
                               std::size_t sharedMemoryBytes = 0);

  template <typename Func>
  void gpu_execute_nd_range_3D(const sycl::nd_range<3> &r, Func f,
                               std::size_t sharedMemoryBytes = 0);

  template <typename KernelName, typename Func, int dim>
  void parallel_for(sycl::range<dim> r, Func f) {
    if constexpr (dim == 1) {
      gpu_execute_1D(r, f);
    } else if constexpr (dim == 2) {
      gpu_execute_2D(r, f);
    } else {
      static_assert(dim <= 2, "Only 1D/2D supported");
    }
  }

  template <typename KernelName, typename Func, int dim>
  void parallel_for(const sycl::nd_range<dim> &r, Func f) {
    if constexpr (dim == 1) {
      gpu_execute_nd_range_1D(r, f);
    } else if constexpr (dim == 2) {
      gpu_execute_nd_range_2D(r, f);
    } else if constexpr (dim == 3) {
      gpu_execute_nd_range_3D(r, f);
    } else {
      static_assert(dim <= 3, "Only 1D/2D/3D supported");
    }
  }

  sycl::event memcpy(void *dest, const void *src, size_t numBytes) {
    ensure_stream();

    cudaError_t err =
        cudaMemcpyAsync(dest, src, numBytes, cudaMemcpyDefault, stream);

    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaStreamSynchronize after memcpy failed: ") +
          cudaGetErrorString(err));
    }

    return sycl::event{};
  }

  template <typename T> sycl::event copy(const T *src, T *dest, size_t count) {
    ensure_stream();

    cudaError_t err = cudaMemcpyAsync(
        static_cast<void *>(dest), static_cast<const void *>(src),
        count * sizeof(T), cudaMemcpyDefault, stream);

    if (err != cudaSuccess) {
      throw sycl::exception(cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      throw sycl::exception(
          std::string("cudaStreamSynchronize after copy failed: ") +
          cudaGetErrorString(err));
    }

    return sycl::event{};
  }

  sycl::event memset(void *ptr, int value, size_t numBytes) {
    ensure_stream();

    cudaError_t err = cudaMemsetAsync(ptr, value, numBytes, stream);

    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemsetAsync failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaStreamSynchronize after memset failed: ") +
          cudaGetErrorString(err));
    }

    return sycl::event{};
  }

  ~cuda_threadpool() noexcept { release_stream_noexcept(); }

private:
  sycl::device dev_;
  sycl::context ctx_{};
  sycl::property_list props_;
  sycl::queue queue_;

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
      throw std::runtime_error(
          "CUDA operation requested without a valid CUDA stream");
    }
  }

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
      std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(err)
                << '\n';
    }

    stream = nullptr;
    ownsStream = false;
  }

  static void init_cuda_for(const sycl::device &d) {
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
  wait_for_dependencies();
  (void)gpu_pool_->memset(ptr, value, num_bytes);
}

inline void handler::memcpy(void *dest, const void *src, size_t num_bytes) {
  if (gpu_pool_ == nullptr) {
    throw std::runtime_error("handler::memcpy has no CUDA backend");
  }
  wait_for_dependencies();
  (void)gpu_pool_->memcpy(dest, src, num_bytes);
}

template <typename T> T *malloc_shared(size_t n, const cuda_threadpool &) {
  T *ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, n * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err)
              << "\n";
    return nullptr;
  }
  return ptr;
}

template <typename T> T *malloc_shared(size_t n, const context &) {
  T *ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, n * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err)
              << "\n";
    return nullptr;
  }
  return ptr;
}

template <typename T> T *malloc_host(size_t n, const cuda_threadpool &) {
  T *ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, n * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << "\n";
    return nullptr;
  }
  return ptr;
}

template <typename T> T *malloc_host(size_t n, const context &) {
  T *ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, n * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << "\n";
    return nullptr;
  }
  return ptr;
}

template <typename T> T *malloc_device(size_t n, const cuda_threadpool &) {
  T *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
    return nullptr;
  }
  return ptr;
}

template <typename T>
T *malloc_device(size_t n, const sycl::device &dev, const sycl::context &) {
  if (dev.is_gpu()) {
    const cudaError_t setErr = cudaSetDevice(dev.get_native_id());
    if (setErr != cudaSuccess) {
      std::cerr << "cudaSetDevice failed before cudaMalloc: "
                << cudaGetErrorString(setErr) << "\n";
      return nullptr;
    }
  }

  T *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));

  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
    return nullptr;
  }

  return ptr;
}

template <typename T> T *malloc_device(size_t n, const context &) {
  T *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
    return nullptr;
  }
  return ptr;
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
  freeCudaUsmPointer(ptr);
}

template <typename T> void free(T *ptr, const sycl::context &) {
  freeCudaUsmPointer(static_cast<void *>(ptr));
}

inline void *sycl::interop_handle::get_native_queue() { return backend_ptr_; }

template <>
inline sycl::backend_return_t<sycl::backend::cuda, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::cuda>() const {
  if (backend_ != sycl::backend::cuda) {
    return nullptr;
  }

  auto *pool = static_cast<cuda_threadpool *>(backend_ptr_);
  return pool ? pool->stream : nullptr;
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

  return backend_ptr_;
}
} // namespace sycl

template <typename Func> sycl::event cuda_threadpool::spawn_1D(Func f) {

  sycl::handler cgh(*this);
  f(cgh);

  return sycl::event();
}

template <typename Func> sycl::event cuda_threadpool::spawn_1D_event(Func f) {

  spawn_1D(f);
  return sycl::event();
}

template <typename Func> void cuda_threadpool::spawn_ND(Func f) {
  sycl::handler cgh(*this); // Currently same as 1D; modify for ND ranges
  f(cgh);
}

#include "kem_gpu/handler_impl.hpp"

inline void sycl::queue::wait() {
  if (gpu_pool_) {
    gpu_pool_->wait();
  }
}

inline void sycl::queue::wait_and_throw() {
  if (gpu_pool_) {
    gpu_pool_->wait_and_throw();
  }

  throw_async_exceptns();
}
#endif
