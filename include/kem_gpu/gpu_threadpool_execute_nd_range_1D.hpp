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

#ifndef __PARAS_GPU_THREADPOOL_EXECUTE_NDRANGE_1D_HPP__
#define __PARAS_GPU_THREADPOOL_EXECUTE_NDRANGE_1D_HPP__

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include <sycl/id.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/range.hpp>

template <typename Func>
__global__ void gpu_execute_nd_range_1D_kernel(std::size_t globalSize,
                                               sycl::range<1> globalRange,
                                               sycl::range<1> localRange,
                                               sycl::range<1> groupRange,
                                               Func f) {
  extern __shared__ unsigned char paras_dynamic_shared_memory[];

  (void)paras_dynamic_shared_memory;

  const std::size_t gid = static_cast<std::size_t>(blockIdx.x) *
                              static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);

  if (gid >= globalSize) {
    return;
  }

  const sycl::nd_item<1> item(
      sycl::id<1>(gid), sycl::id<1>(static_cast<std::size_t>(threadIdx.x)),
      sycl::id<1>(static_cast<std::size_t>(blockIdx.x)), globalRange,
      localRange, groupRange);

  f(item);
}

template <typename Func>
void cuda_threadpool::gpu_execute_nd_range_1D(const sycl::nd_range<1> &r,
                                              Func f,
                                              std::size_t sharedMemoryBytes) {
  ensure_stream();

  const auto globalRange = r.get_global_range();
  const auto localRange = r.get_local_range();

  const std::size_t globalSize = globalRange[0];
  const std::size_t localSize = localRange[0];

  if (globalSize == 0 || localSize == 0) {
    throw std::runtime_error("1D nd_range has a zero dimension");
  }

  if ((globalSize % localSize) != 0) {
    throw std::runtime_error("1D nd_range global size is not divisible by "
                             "the local size");
  }

  paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()),
                           "cudaSetDevice before 1D nd_range kernel");

  cudaDeviceProp props{};

  paras_cuda_detail::check(
      cudaGetDeviceProperties(&props, dev_.get_native_id()),
      "cudaGetDeviceProperties for 1D nd_range");

  if (localSize > static_cast<std::size_t>(props.maxThreadsPerBlock)) {
    throw std::runtime_error("1D local range exceeds CUDA maxThreadsPerBlock");
  }

  if (sharedMemoryBytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
    throw std::runtime_error("local_accessor allocation exceeds CUDA "
                             "shared memory per block");
  }

  const std::size_t groups = globalSize / localSize;

  const sycl::range<1> launchedGroupRange(groups);

  (void)cudaGetLastError();

  gpu_execute_nd_range_1D_kernel<<<static_cast<unsigned int>(groups),
                                   static_cast<unsigned int>(localSize),
                                   sharedMemoryBytes, stream>>>(
      globalSize, globalRange, localRange, launchedGroupRange, f);

  paras_cuda_detail::check(cudaGetLastError(),
                           "1D nd_range kernel launch failed");

  paras_cuda_detail::check(cudaStreamSynchronize(stream),
                           "1D nd_range kernel execution failed");
}

#endif		/** End of gpu_threadpool_execute_nd_range_1D >*/
