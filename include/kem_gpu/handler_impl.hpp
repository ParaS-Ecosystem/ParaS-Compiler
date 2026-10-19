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

#ifndef __PARAS_GPU_HANDEL_IMPL_HPP__
#define __PARAS_GPU_HANDEL_IMPL_HPP__

#include "kem_gpu/gpu_threadpool.hpp"
#include "sycl/handler.hpp"

namespace sycl {
namespace detail {

#ifdef __CLANG_RDC__

inline void spec_const_upload_all(void *stream) {
  static unsigned long long uploaded = ~0ull;
  static void *uploaded_on = nullptr;
  static spec_const_blob staged;
  unsigned long long gen;
  {
    std::lock_guard<std::mutex> lock(spec_const_mutex());
    gen = spec_const_generation();
  }
  if (gen == uploaded && stream == uploaded_on)
    return;
  spec_const_fill_blob(staged);
  uploaded = gen;
  uploaded_on = stream;
  if (gen == 0 && staged.count == 0)
    return;
  const std::size_t bytes = offsetof(spec_const_blob, data) + staged.used;
  for (const void *table : spec_const_tables())
    if (cudaMemcpyToSymbolAsync(table, &staged, bytes, 0, cudaMemcpyHostToDevice,
                                static_cast<cudaStream_t>(stream)) != cudaSuccess)
      (void)cudaGetLastError();
}
#endif

template <typename Kernel> inline void spec_const_upload(void *stream) {
#ifdef __CLANG_RDC__
  if (stream != nullptr)
    spec_const_upload_all(stream);
  return;
#endif
  static unsigned long long uploaded = ~0ull;
  static void *uploaded_on = nullptr;
  if (stream == nullptr)
    return;
  unsigned long long gen;
  {
    std::lock_guard<std::mutex> lock(spec_const_mutex());
    gen = spec_const_generation();
  }
  if (gen == uploaded && stream == uploaded_on)
    return;
  static spec_const_blob staged;
  spec_const_fill_blob(staged);
  if (gen == 0 && staged.count == 0 && uploaded == ~0ull) {
    uploaded = gen;
    uploaded_on = stream;
    return;
  }
  const std::size_t bytes =
      offsetof(spec_const_blob, data) + staged.used;
  if (cudaMemcpyToSymbolAsync(PARAS_GPU_SYMBOL(paras_spec_const_table), &staged, bytes, 0,
                              cudaMemcpyHostToDevice,
                              static_cast<cudaStream_t>(stream)) == cudaSuccess) {
    uploaded = gen;
    uploaded_on = stream;
  }
}
}

inline void handler::paras_enqueue_dependencies() {
  if (gpu_pool_ != nullptr) {
    gpu_pool_->paras_enqueue_dependencies(deps_);
  } else {
    wait_for_dependencies();
  }
}

inline event handler::paras_gpu_queue_tail() {
  return gpu_pool_ != nullptr ? gpu_pool_->paras_last_command_event(true)
                              : event{};
}

inline void handler::paras_gpu_command_enqueued(bool by_interop) {
  if (gpu_pool_ != nullptr) {
    if (by_interop) {

      gpu_pool_->paras_command_enqueued("custom operation failed");
    }
    result_ = gpu_pool_->paras_last_command_event(want_native_event_);
  }
}

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(range<dim> r, Func f) {
  paras_launch_kernel<KernelName, Func>();
  paras_check_range_kernel<Func>();
  if (gpu_pool_ == nullptr) {
    throw std::runtime_error("GPU handler has no cuda_threadpool backend");
  }

  paras_enqueue_dependencies();
  detail::spec_const_upload<Func>(gpu_pool_->stream);

  if constexpr (dim == 1) {
    if (isAsyncEnabled()) {
      gpu_pool_->gpu_execute_1D_async(f);
    } else {
      gpu_pool_->gpu_execute_1D(r, f);
    }
  } else if constexpr (dim == 2) {
    gpu_pool_->gpu_execute_2D(r, f);
  } else if constexpr (dim == 3) {
    gpu_pool_->gpu_execute_3D(r, f);
  } else {
    static_assert(dim <= 3, "Only 1D/2D/3D supported");
  }
  paras_gpu_command_enqueued();
}

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(const nd_range<dim> &r, Func f) {
  paras_launch_nd_kernel<KernelName, Func>(r);
  if (gpu_pool_ == nullptr) {
    throw std::runtime_error("GPU handler has no cuda_threadpool backend");
  }

  paras_enqueue_dependencies();
  detail::spec_const_upload<Func>(gpu_pool_->stream);

  const std::size_t sharedMemoryBytes = local_memory_size();

  if constexpr (dim == 1) {
    gpu_pool_->gpu_execute_nd_range_1D(r, f, sharedMemoryBytes);
  } else if constexpr (dim == 2) {
    gpu_pool_->gpu_execute_nd_range_2D(r, f, sharedMemoryBytes);
  } else if constexpr (dim == 3) {
    gpu_pool_->gpu_execute_nd_range_3D(r, f, sharedMemoryBytes);
  } else {
    static_assert(dim <= 3, "Only 1D/2D/3D supported");
  }
  paras_gpu_command_enqueued();
}

template <typename KernelName, int Dim, typename Func>
void handler::paras_hierarchical_gpu(range<Dim> groups, range<Dim> local,
                                     Func f) {
  using K = detail::hierarchical_nd_kernel<Dim, Func>;
  const K k{f, groups, local};

  paras_launch_kernel<KernelName, K>();
  paras_enqueue_dependencies();
  detail::spec_const_upload<Func>(gpu_pool_->stream);
  range<Dim> one;
  for (int d = 0; d < Dim; ++d)
    one[d] = 1;
  const nd_range<Dim> r(groups, one);
  const std::size_t sharedMemoryBytes = local_memory_size();
  if constexpr (Dim == 1) {
    gpu_pool_->gpu_execute_nd_range_1D(r, k, sharedMemoryBytes);
  } else if constexpr (Dim == 2) {
    gpu_pool_->gpu_execute_nd_range_2D(r, k, sharedMemoryBytes);
  } else {
    gpu_pool_->gpu_execute_nd_range_3D(r, k, sharedMemoryBytes);
  }
  paras_gpu_command_enqueued();
}

}

#endif
