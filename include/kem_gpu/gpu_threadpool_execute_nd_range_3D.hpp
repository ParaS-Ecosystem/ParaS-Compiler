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

#ifndef __PARAS_GPU_THREADPOOL_EXECUTE_NDRANGE_3D_HPP__
#define __PARAS_GPU_THREADPOOL_EXECUTE_NDRANGE_3D_HPP__

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <sycl/id.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/range.hpp>

template <typename Func>
__global__ void gpu_execute_nd_range_3D_kernel(
    std::size_t globalSize0, std::size_t globalSize1, std::size_t globalSize2,
    sycl::range<3> globalRange, sycl::range<3> localRange,
    sycl::range<3> groupRange, Func f) {
  extern __shared__ unsigned char paras_dynamic_shared_memory[];
  (void)paras_dynamic_shared_memory;

  const std::size_t globalId0 = static_cast<std::size_t>(blockIdx.z) *
                                    static_cast<std::size_t>(blockDim.z) +
                                static_cast<std::size_t>(threadIdx.z);

  const std::size_t globalId1 = static_cast<std::size_t>(blockIdx.y) *
                                    static_cast<std::size_t>(blockDim.y) +
                                static_cast<std::size_t>(threadIdx.y);

  const std::size_t globalId2 = static_cast<std::size_t>(blockIdx.x) *
                                    static_cast<std::size_t>(blockDim.x) +
                                static_cast<std::size_t>(threadIdx.x);

  if (globalId0 >= globalSize0 || globalId1 >= globalSize1 ||
      globalId2 >= globalSize2) {
    return;
  }

  const sycl::id<3> globalId(globalId0, globalId1, globalId2);

  const sycl::id<3> localId(static_cast<std::size_t>(threadIdx.z),
                            static_cast<std::size_t>(threadIdx.y),
                            static_cast<std::size_t>(threadIdx.x));

  const sycl::id<3> groupId(static_cast<std::size_t>(blockIdx.z),
                            static_cast<std::size_t>(blockIdx.y),
                            static_cast<std::size_t>(blockIdx.x));

  sycl::nd_item<3> item(globalId, localId, groupId, globalRange, localRange,
                        groupRange);

  f(item);
}

template <typename Func>
void cuda_threadpool::gpu_execute_nd_range_3D(const sycl::nd_range<3> &r,
                                              Func f,
                                              std::size_t sharedMemoryBytes) {
  ensure_stream();

  const sycl::range<3> globalRange = r.get_global_range();
  const sycl::range<3> localRange = r.get_local_range();

  const std::size_t globalSize0 = globalRange[0];
  const std::size_t globalSize1 = globalRange[1];
  const std::size_t globalSize2 = globalRange[2];

  const std::size_t localSize0 = localRange[0];
  const std::size_t localSize1 = localRange[1];
  const std::size_t localSize2 = localRange[2];

  if (globalSize0 == 0 || globalSize1 == 0 || globalSize2 == 0) {
    throw std::runtime_error(
        "3D kernel received a zero global-range dimension");
  }

  if (localSize0 == 0 || localSize1 == 0 || localSize2 == 0) {
    throw std::runtime_error("3D kernel received a zero local-range dimension");
  }

  if (localSize0 > std::numeric_limits<std::size_t>::max() / localSize1 ||
      localSize0 * localSize1 >
          std::numeric_limits<std::size_t>::max() / localSize2) {
    throw std::runtime_error("3D local-range size overflow");
  }

  const std::size_t threadsPerBlock = localSize0 * localSize1 * localSize2;

  int currentDevice = -1;
  cudaError_t err = cudaGetDevice(&currentDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }

  cudaDeviceProp properties{};
  err = cudaGetDeviceProperties(&properties, currentDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
                             cudaGetErrorString(err));
  }

  if (sharedMemoryBytes >
      static_cast<std::size_t>(properties.sharedMemPerBlock)) {
    throw std::runtime_error(
        "3D local_accessor allocation exceeds CUDA shared memory per block");
  }

  if (threadsPerBlock >
      static_cast<std::size_t>(properties.maxThreadsPerBlock)) {
    throw std::runtime_error("3D local range exceeds CUDA maxThreadsPerBlock");
  }

  if (localSize2 > static_cast<std::size_t>(properties.maxThreadsDim[0]) ||
      localSize1 > static_cast<std::size_t>(properties.maxThreadsDim[1]) ||
      localSize0 > static_cast<std::size_t>(properties.maxThreadsDim[2])) {
    throw std::runtime_error(
        "3D local range exceeds a CUDA block-dimension limit");
  }

  const std::size_t groups0 = (globalSize0 + localSize0 - 1) / localSize0;
  const std::size_t groups1 = (globalSize1 + localSize1 - 1) / localSize1;
  const std::size_t groups2 = (globalSize2 + localSize2 - 1) / localSize2;

  if (groups2 > static_cast<std::size_t>(properties.maxGridSize[0]) ||
      groups1 > static_cast<std::size_t>(properties.maxGridSize[1]) ||
      groups0 > static_cast<std::size_t>(properties.maxGridSize[2])) {
    throw std::runtime_error(
        "3D group range exceeds a CUDA grid-dimension limit");
  }

  const dim3 block(static_cast<unsigned int>(localSize2),
                   static_cast<unsigned int>(localSize1),
                   static_cast<unsigned int>(localSize0));

  const dim3 grid(static_cast<unsigned int>(groups2),
                  static_cast<unsigned int>(groups1),
                  static_cast<unsigned int>(groups0));

  const sycl::range<3> launchedGroupRange(groups0, groups1, groups2);

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA failure before 3D kernel launch: ") +
        cudaGetErrorString(err));
  }

  (void)cudaGetLastError();

  gpu_execute_nd_range_3D_kernel<<<grid, block, sharedMemoryBytes, stream>>>(
      globalSize0, globalSize1, globalSize2, globalRange, localRange,
      launchedGroupRange, f);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("3D kernel launch failed: ") +
                             cudaGetErrorString(err));
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("3D kernel execution failed: ") +
                             cudaGetErrorString(err));
  }
}

#endif
