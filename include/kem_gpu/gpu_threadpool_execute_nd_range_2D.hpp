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

#ifndef __PARAS_GPU_THREADPOOL_EXECUTE_NDRANGE_2D_HPP__
#define __PARAS_GPU_THREADPOOL_EXECUTE_NDRANGE_2D_HPP__

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <sycl/id.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/range.hpp>

template <typename Func>
__global__ void gpu_execute_nd_range_2D_kernel(
    std::size_t global0, std::size_t global1, sycl::range<2> globalRange,
    sycl::range<2> localRange, sycl::range<2> groupRange, Func f) {
  extern __shared__ unsigned char paras_dynamic_shared_memory[];
  (void)paras_dynamic_shared_memory;

  const std::size_t gid0 = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t gid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid0 >= global0 || gid1 >= global1) {
    return;
  }

  sycl::nd_item<2> item(
      sycl::id<2>(gid0, gid1), sycl::id<2>(threadIdx.y, threadIdx.x),
      sycl::id<2>(blockIdx.y, blockIdx.x), globalRange, localRange, groupRange);
  f(item);
}

template <typename Func>
void cuda_threadpool::gpu_execute_nd_range_2D(const sycl::nd_range<2> &r,
                                              Func f,
                                              std::size_t sharedMemoryBytes) {
  ensure_stream();
  const auto globalRange = r.get_global_range();
  const auto localRange = r.get_local_range();
  const std::size_t global0 = globalRange[0];
  const std::size_t global1 = globalRange[1];
  const std::size_t local0 = localRange[0];
  const std::size_t local1 = localRange[1];

  if (global0 == 0 || global1 == 0 || local0 == 0 || local1 == 0) {
    throw std::runtime_error("2D nd_range has a zero dimension");
  }

  paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()),
                           "cudaSetDevice before 2D nd_range kernel");

  cudaDeviceProp props{};
  paras_cuda_detail::check(
      cudaGetDeviceProperties(&props, dev_.get_native_id()),
      "cudaGetDeviceProperties for 2D nd_range");
  if (sharedMemoryBytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    throw std::runtime_error(
        "2D local_accessor allocation exceeds CUDA shared memory per block");
  }

  if (local0 * local1 > static_cast<std::size_t>(props.maxThreadsPerBlock) ||
      local1 > static_cast<std::size_t>(props.maxThreadsDim[0]) ||
      local0 > static_cast<std::size_t>(props.maxThreadsDim[1])) {
    throw std::runtime_error("2D local range exceeds CUDA block limits");
  }

  const std::size_t groups0 = (global0 + local0 - 1) / local0;
  const std::size_t groups1 = (global1 + local1 - 1) / local1;
  const dim3 block(static_cast<unsigned int>(local1),
                   static_cast<unsigned int>(local0));
  const dim3 grid(static_cast<unsigned int>(groups1),
                  static_cast<unsigned int>(groups0));
  const sycl::range<2> launchedGroupRange(groups0, groups1);

  (void)cudaGetLastError();
  gpu_execute_nd_range_2D_kernel<<<grid, block, sharedMemoryBytes, stream>>>(
      global0, global1, globalRange, localRange, launchedGroupRange, f);

  paras_cuda_detail::check(cudaGetLastError(),
                           "2D nd_range kernel launch failed");
  paras_cuda_detail::check(cudaStreamSynchronize(stream),
                           "2D nd_range kernel execution failed");
}

#endif		/** End of gpu_threadpool_execute_nd_range_2D >*/
