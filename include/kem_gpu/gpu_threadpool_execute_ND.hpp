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
#include <stdexcept>
#include <string>
#include <type_traits>

#include "gpu_utilities.hpp"
#if (PARAS_CUDA_BACKEND)
#include <cuda_runtime.h>
#elif (PARAS_HIP_BACKEND)
#include <hip/hip_runtime.h>
#endif

template <typename Func>
__global__ void gpu_execute_2D_grid_kernel(std::size_t rows, std::size_t cols, Func f) {
    const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        sycl::detail::invoke_range_kernel<2>(f, sycl::id<2>(row, col), sycl::range<2>(rows, cols));
    }
}

template <typename Func>
__global__ void gpu_execute_3D_linear_kernel(std::size_t X, std::size_t Y, std::size_t Z, Func f) {
    const std::size_t total = X * Y * Z;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t lin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         lin < total; lin += stride) {
        const std::size_t k = lin % Z;
        const std::size_t j = (lin / Z) % Y;
        const std::size_t i = lin / (Y * Z);
        sycl::detail::invoke_range_kernel<3>(f, sycl::id<3>(i, j, k), sycl::range<3>(X, Y, Z));
    }
}

#if (PARAS_CUDA_BACKEND)
template <typename Func>
void cuda_threadpool::gpu_execute_2D(const sycl::range<2>& r, Func f) {
    ensure_stream();
    const std::size_t rows = r[0];
    const std::size_t cols = r[1];
    if (rows == 0 || cols == 0) {
        return;
    }

    paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()), "cudaSetDevice before 2D kernel");

    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x),
                    static_cast<unsigned int>((rows + block.y - 1) / block.y));

    (void)cudaGetLastError();
    gpu_execute_2D_grid_kernel<<<grid, block, 0, stream>>>(rows, cols, f);
    paras_cuda_detail::check(cudaGetLastError(), "2D kernel launch failed");
    paras_cuda_detail::check(cudaStreamSynchronize(stream), "2D kernel execution failed");
}

template <typename Func>
void cuda_threadpool::gpu_execute_3D(const sycl::range<3>& r, Func f) {
    ensure_stream();
    const std::size_t X = r[0], Y = r[1], Z = r[2];
    const std::size_t total = X * Y * Z;
    if (total == 0) {
        return;
    }
    paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()), "cudaSetDevice before 3D kernel");
    constexpr unsigned int threadsPerBlock = 256;
    const std::size_t want = (total + threadsPerBlock - 1) / threadsPerBlock;
    const unsigned int blocks =
        static_cast<unsigned int>(want < 65535u * 32u ? want : 65535u * 32u);
    (void)cudaGetLastError();
    gpu_execute_3D_linear_kernel<<<blocks, threadsPerBlock, 0, stream>>>(X, Y, Z, f);
    paras_cuda_detail::check(cudaGetLastError(), "3D kernel launch failed");
    paras_cuda_detail::check(cudaStreamSynchronize(stream), "3D kernel execution failed");
}

#elif (PARAS_HIP_BACKEND)
template <typename Func>
void rocm_threadpool::gpu_execute_2D(const sycl::range<2>& r, Func f) {
    ensure_stream();
    const std::size_t rows = r[0];
    const std::size_t cols = r[1];
    if (rows == 0 || cols == 0) {
        return;
    }

    paras_rocm_detail::check(hipSetDevice(dev_.get_native_id()), "hipSetDevice before 2D kernel");

    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x),
                    static_cast<unsigned int>((rows + block.y - 1) / block.y));

    (void)hipGetLastError();
    gpu_execute_2D_grid_kernel<<<grid, block, 0, stream>>>(rows, cols, f);
    paras_rocm_detail::check(hipGetLastError(), "2D kernel launch failed");
    paras_rocm_detail::check(hipStreamSynchronize(stream), "2D kernel execution failed");
}

template <typename Func>
void rocm_threadpool::gpu_execute_3D(const sycl::range<3>& r, Func f) {
    ensure_stream();
    const std::size_t X = r[0], Y = r[1], Z = r[2];
    const std::size_t total = X * Y * Z;
    if (total == 0) {
        return;
    }
    paras_rocm_detail::check(hipSetDevice(dev_.get_native_id()), "hipSetDevice before 3D kernel");
    constexpr unsigned int threadsPerBlock = 256;
    const std::size_t want = (total + threadsPerBlock - 1) / threadsPerBlock;
    const unsigned int blocks =
        static_cast<unsigned int>(want < 65535u * 32u ? want : 65535u * 32u);
    (void)hipGetLastError();
    gpu_execute_3D_linear_kernel<<<blocks, threadsPerBlock, 0, stream>>>(X, Y, Z, f);
    paras_rocm_detail::check(hipGetLastError(), "3D kernel launch failed");
    paras_rocm_detail::check(hipStreamSynchronize(stream), "3D kernel execution failed");
}
#endif

#endif /** End of gpu_threadpool_execute_ND >*/
