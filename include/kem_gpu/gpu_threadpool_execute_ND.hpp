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

#ifndef __PARAS_GPU_THREADPOOL_EXECUTE_ND_HPP__
#define __PARAS_GPU_THREADPOOL_EXECUTE_ND_HPP__

#include "gpu_utilities.hpp"
#include "sycl/id.hpp"
#include "sycl/item.hpp"
#include "sycl/range.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

template <typename Wrapped>
__global__ void gpu_execute_2D_grid_kernel(std::size_t rows, std::size_t cols,
                                           Wrapped f) {
  const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    if constexpr (std::is_invocable_v<Wrapped, sycl::item<2>>) {
      f(sycl::item<2>(row, col));
    }
    if constexpr (std::is_invocable_v<Wrapped, sycl::id<2>>) {
      f(sycl::id<2>(row, col));
    }
  }
}

template <typename Func> struct index_descriptors {
  Func f;

  PARAS_KERNEL_HD
  void operator()(sycl::item<2> item) const {
    if constexpr (std::is_invocable_v<const Func &, sycl::item<2>>) {
      f(item);
    }
  }

  PARAS_KERNEL_HD
  void operator()(sycl::id<2> id) const {
    if constexpr (std::is_invocable_v<const Func &, sycl::id<2>>) {
      f(id);
    }
  }
};

template <typename Func>
void cuda_threadpool::gpu_execute_2D(const sycl::range<2> &r, Func f) {
  ensure_stream();
  const std::size_t rows = r[0];
  const std::size_t cols = r[1];
  if (rows == 0 || cols == 0) {
    return;
  }

  paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()),
                           "cudaSetDevice before 2D kernel");

  const dim3 block(16, 16);
  const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x),
                  static_cast<unsigned int>((rows + block.y - 1) / block.y));

  index_descriptors<Func> wrapped{f};
  (void)cudaGetLastError();
  gpu_execute_2D_grid_kernel<<<grid, block, 0, stream>>>(rows, cols, wrapped);
  paras_cuda_detail::check(cudaGetLastError(), "2D kernel launch failed");
  paras_cuda_detail::check(cudaStreamSynchronize(stream),
                           "2D kernel execution failed");
}

#endif		/** End of gpu_threadpool_execute_ND >*/
