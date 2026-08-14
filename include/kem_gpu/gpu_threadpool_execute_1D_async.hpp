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

#ifndef __PARAS_GPU_THREADPOOL_EXECUTE_1D_ASYNC_HPP__
#define __PARAS_GPU_THREADPOOL_EXECUTE_1D_ASYNC_HPP__

#include "sycl/id.hpp"
#include <cstddef>
#include <cuda_runtime.h>

template <typename Func>
__global__ void gpu_execute_1D_async_kernel(std::size_t n, Func f) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    f(sycl::id<1>(i));
  }
}

template <typename Func> void cuda_threadpool::gpu_execute_1D_async(Func f) {
  ensure_stream();
  paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()),
                           "cudaSetDevice before async 1D kernel");

  (void)cudaGetLastError();
  gpu_execute_1D_async_kernel<<<1, 1, 0, stream>>>(1, f);
  paras_cuda_detail::check(cudaGetLastError(), "async 1D kernel launch failed");
  paras_cuda_detail::check(cudaStreamSynchronize(stream),
                           "async 1D kernel execution failed");
}

#endif 		/** End of gpu_threadpool_execute_1D_async >*/
