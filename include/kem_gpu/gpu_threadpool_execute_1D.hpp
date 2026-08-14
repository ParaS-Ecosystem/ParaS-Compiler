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

#ifndef __PARAS_GPU_THREADPOOL_EXECUTE_1D_HPP__
#define __PARAS_GPU_THREADPOOL_EXECUTE_1D_HPP__

#include "sycl/id.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace paras_cuda_detail {
inline void check(cudaError_t err, const char *where) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(err));
  }
}
} // namespace paras_cuda_detail

template <typename Func>
__global__ void gpu_execute_1D_kernel(std::size_t n, Func f) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    f(sycl::id<1>(i));
  }
}

template <typename Func>
void cuda_threadpool::gpu_execute_1D(const sycl::range<1> &r, Func f) {
  ensure_stream();
  const std::size_t n = r[0];
  if (n == 0) {
    return;
  }

  paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()),
                           "cudaSetDevice before 1D kernel");

  constexpr unsigned int threadsPerBlock = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((n + threadsPerBlock - 1) / threadsPerBlock);

  (void)cudaGetLastError();
  gpu_execute_1D_kernel<<<blocks, threadsPerBlock, 0, stream>>>(n, f);
  paras_cuda_detail::check(cudaGetLastError(), "1D kernel launch failed");
  paras_cuda_detail::check(cudaStreamSynchronize(stream),
                           "1D kernel execution failed");
}

#endif
