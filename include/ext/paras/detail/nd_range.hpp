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

#ifndef __SYCL_EXT_PARAS_ND_RANGE_HPP__
#define __SYCL_EXT_PARAS_ND_RANGE_HPP__

#if (PARAS_CUDA_BACKEND)
template <typename Func>
void cuda_threadpool::paras_ext_nd_range_1D(const sycl::nd_range<1>& r, Func f,
                                              std::size_t sharedMemoryBytes) {
    paras_detail::validate_nd_range(r);
    ensure_stream();

    const auto globalRange = r.get_global_range();
    const auto localRange = r.get_local_range();

    const std::size_t globalSize = globalRange[0];
    const std::size_t localSize = localRange[0];

    if (globalSize == 0 || localSize == 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "1D nd_range has a zero dimension");
    }

    if ((globalSize % localSize) != 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "1D nd_range global size is not divisible by "
                              "the local size");
    }

    paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()),
                             "cudaSetDevice before 1D nd_range kernel");

    const cudaDeviceProp& props = device_props();

    if (localSize > static_cast<std::size_t>(props.maxThreadsPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "1D local range exceeds CUDA maxThreadsPerBlock");
    }

    if (sharedMemoryBytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                              "local_accessor allocation exceeds CUDA "
                              "shared memory per block");
    }

    const std::size_t groups = globalSize / localSize;

    const sycl::range<1> launchedGroupRange(groups);

    (void)cudaGetLastError();

    gpu_execute_nd_range_1D_kernel<<<static_cast<unsigned int>(groups),
                                     static_cast<unsigned int>(localSize), sharedMemoryBytes,
                                     stream>>>(globalSize, globalRange, localRange,
                                               launchedGroupRange, f);

    paras_cuda_detail::check(cudaGetLastError(), "1D nd_range kernel launch failed");

    if (!async_launch()) {
        paras_cuda_detail::check(cudaStreamSynchronize(stream), "1D nd_range kernel execution failed");
    }
}

template <typename Func>
void cuda_threadpool::paras_ext_nd_range_2D(const sycl::nd_range<2>& r, Func f,
                                              std::size_t sharedMemoryBytes) {
    paras_detail::validate_nd_range(r);
    ensure_stream();
    const auto globalRange = r.get_global_range();
    const auto localRange = r.get_local_range();
    const std::size_t global0 = globalRange[0];
    const std::size_t global1 = globalRange[1];
    const std::size_t local0 = localRange[0];
    const std::size_t local1 = localRange[1];

    if (global0 == 0 || global1 == 0 || local0 == 0 || local1 == 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "2D nd_range has a zero dimension");
    }

    paras_cuda_detail::check(cudaSetDevice(dev_.get_native_id()),
                             "cudaSetDevice before 2D nd_range kernel");

    const cudaDeviceProp& props = device_props();
    if (sharedMemoryBytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                              "2D local_accessor allocation exceeds CUDA shared memory per block");
    }

    if (local0 * local1 > static_cast<std::size_t>(props.maxThreadsPerBlock) ||
        local1 > static_cast<std::size_t>(props.maxThreadsDim[0]) ||
        local0 > static_cast<std::size_t>(props.maxThreadsDim[1])) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "2D local range exceeds CUDA block limits");
    }

    const std::size_t groups0 = (global0 + local0 - 1) / local0;
    const std::size_t groups1 = (global1 + local1 - 1) / local1;
    const dim3 block(static_cast<unsigned int>(local1), static_cast<unsigned int>(local0));
    const dim3 grid(static_cast<unsigned int>(groups1), static_cast<unsigned int>(groups0));
    const sycl::range<2> launchedGroupRange(groups0, groups1);

    (void)cudaGetLastError();
    gpu_execute_nd_range_2D_kernel<<<grid, block, sharedMemoryBytes, stream>>>(
        global0, global1, globalRange, localRange, launchedGroupRange, f);

    paras_cuda_detail::check(cudaGetLastError(), "2D nd_range kernel launch failed");
    if (!async_launch()) {
        paras_cuda_detail::check(cudaStreamSynchronize(stream), "2D nd_range kernel execution failed");
    }
}

template <typename Func>
void cuda_threadpool::paras_ext_nd_range_3D(const sycl::nd_range<3>& r, Func f,
                                              std::size_t sharedMemoryBytes) {
    paras_detail::validate_nd_range(r);
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
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D kernel received a zero global-range dimension");
    }

    if (localSize0 == 0 || localSize1 == 0 || localSize2 == 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D kernel received a zero local-range dimension");
    }

    if (localSize0 > std::numeric_limits<std::size_t>::max() / localSize1 ||
        localSize0 * localSize1 > std::numeric_limits<std::size_t>::max() / localSize2) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D local-range size overflow");
    }

    const std::size_t threadsPerBlock = localSize0 * localSize1 * localSize2;

    int currentDevice = -1;
    cudaError_t err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceProp properties{};
    err = cudaGetDeviceProperties(&properties, currentDevice);
    if (err != cudaSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("cudaGetDeviceProperties failed: ") +
                                  cudaGetErrorString(err));
    }

    if (sharedMemoryBytes > static_cast<std::size_t>(properties.sharedMemPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                              "3D local_accessor allocation exceeds CUDA shared memory per block");
    }

    if (threadsPerBlock > static_cast<std::size_t>(properties.maxThreadsPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D local range exceeds CUDA maxThreadsPerBlock");
    }

    if (localSize2 > static_cast<std::size_t>(properties.maxThreadsDim[0]) ||
        localSize1 > static_cast<std::size_t>(properties.maxThreadsDim[1]) ||
        localSize0 > static_cast<std::size_t>(properties.maxThreadsDim[2])) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D local range exceeds a CUDA block-dimension limit");
    }

    const std::size_t groups0 = (globalSize0 + localSize0 - 1) / localSize0;
    const std::size_t groups1 = (globalSize1 + localSize1 - 1) / localSize1;
    const std::size_t groups2 = (globalSize2 + localSize2 - 1) / localSize2;

    if (groups2 > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        groups1 > static_cast<std::size_t>(properties.maxGridSize[1]) ||
        groups0 > static_cast<std::size_t>(properties.maxGridSize[2])) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D group range exceeds a CUDA grid-dimension limit");
    }

    const dim3 block(static_cast<unsigned int>(localSize2), static_cast<unsigned int>(localSize1),
                     static_cast<unsigned int>(localSize0));

    const dim3 grid(static_cast<unsigned int>(groups2), static_cast<unsigned int>(groups1),
                    static_cast<unsigned int>(groups0));

    const sycl::range<3> launchedGroupRange(groups0, groups1, groups2);

    err = async_launch() ? cudaSuccess : cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("CUDA failure before 3D kernel launch: ") +
                                  cudaGetErrorString(err));
    }

    (void)cudaGetLastError();

    gpu_execute_nd_range_3D_kernel<<<grid, block, sharedMemoryBytes, stream>>>(
        globalSize0, globalSize1, globalSize2, globalRange, localRange, launchedGroupRange, f);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("3D kernel launch failed: ") + cudaGetErrorString(err));
    }

    err = async_launch() ? cudaSuccess : cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("3D kernel execution failed: ") +
                                  cudaGetErrorString(err));
    }
}

#elif (PARAS_HIP_BACKEND)
template <typename Func>
__device__ inline void gpu_execute_nd_range_1D_body(std::size_t globalSize, sycl::range<1> globalRange,
                                                    sycl::range<1> localRange, sycl::range<1> groupRange,
                                                    const Func& f) {
    const std::size_t gid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);

    if (gid >= globalSize) {
        return;
    }

    const sycl::nd_item<1> item(
        sycl::id<1>(gid), sycl::id<1>(static_cast<std::size_t>(threadIdx.x)),
        sycl::id<1>(static_cast<std::size_t>(blockIdx.x)), globalRange, localRange, groupRange);

    f(item);
}

template <typename Func, int MaxThreads, int MinWaves>
__global__ void __launch_bounds__(MaxThreads, MinWaves)
    gpu_execute_nd_range_1D_bounded_kernel(std::size_t globalSize, sycl::range<1> globalRange,
                                           sycl::range<1> localRange, sycl::range<1> groupRange,
                                           Func f) {
    extern __shared__ unsigned char paras_dynamic_shared_memory[];

    (void)paras_dynamic_shared_memory;

    gpu_execute_nd_range_1D_body(globalSize, globalRange, localRange, groupRange, f);
}

template <typename Func>
void rocm_threadpool::paras_ext_nd_range_1D(const sycl::nd_range<1>& r, Func f,
                                              std::size_t sharedMemoryBytes) {
    paras_detail::validate_nd_range(r);
    ensure_stream();

    const auto globalRange = r.get_global_range();
    const auto localRange = r.get_local_range();

    const std::size_t globalSize = globalRange[0];
    const std::size_t localSize = localRange[0];

    if (globalSize == 0 || localSize == 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "1D nd_range has a zero dimension");
    }

    if ((globalSize % localSize) != 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "1D nd_range global size is not divisible by "
                              "the local size");
    }

    paras_rocm_detail::check(hipSetDevice(dev_.get_native_id()),
                             "hipSetDevice before 1D nd_range kernel");

    const hipDeviceProp_t& props = device_props();

    if (localSize > static_cast<std::size_t>(props.maxThreadsPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "1D local range exceeds HIP maxThreadsPerBlock");
    }

    if (sharedMemoryBytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                              "local_accessor allocation exceeds HIP "
                              "shared memory per block");
    }

    const std::size_t groups = globalSize / localSize;

    const sycl::range<1> launchedGroupRange(groups);

    (void)hipGetLastError();

    using Hint = sycl::ext::paras::detail::occupancy_hint<Func>;
    if constexpr (Hint::value) {
        if (localSize > static_cast<std::size_t>(Hint::max_work_group_size)) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                                  "1D local range exceeds the kernel's "
                                  "paras_max_work_group_size");
        }
        gpu_execute_nd_range_1D_bounded_kernel<Func, Hint::max_work_group_size, Hint::min_waves_per_eu>
            <<<static_cast<unsigned int>(groups), static_cast<unsigned int>(localSize),
               sharedMemoryBytes, stream>>>(globalSize, globalRange, localRange, launchedGroupRange, f);
    } else {
        gpu_execute_nd_range_1D_kernel<<<static_cast<unsigned int>(groups),
                                         static_cast<unsigned int>(localSize), sharedMemoryBytes,
                                         stream>>>(globalSize, globalRange, localRange,
                                                   launchedGroupRange, f);
    }

    paras_rocm_detail::check(hipGetLastError(), "1D nd_range kernel launch failed");

    if (!async_launch()) {
        paras_rocm_detail::check(hipStreamSynchronize(stream), "1D nd_range kernel execution failed");
    }
}

template <typename Func>
void rocm_threadpool::paras_ext_nd_range_2D(const sycl::nd_range<2>& r, Func f,
                                              std::size_t sharedMemoryBytes) {
    paras_detail::validate_nd_range(r);
    ensure_stream();
    const auto globalRange = r.get_global_range();
    const auto localRange = r.get_local_range();
    const std::size_t global0 = globalRange[0];
    const std::size_t global1 = globalRange[1];
    const std::size_t local0 = localRange[0];
    const std::size_t local1 = localRange[1];

    if (global0 == 0 || global1 == 0 || local0 == 0 || local1 == 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "2D nd_range has a zero dimension");
    }

    paras_rocm_detail::check(hipSetDevice(dev_.get_native_id()),
                             "hipSetDevice before 2D nd_range kernel");

    const hipDeviceProp_t& props = device_props();
    if (sharedMemoryBytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                              "2D local_accessor allocation exceeds HIP shared memory per block");
    }

    if (local0 * local1 > static_cast<std::size_t>(props.maxThreadsPerBlock) ||
        local1 > static_cast<std::size_t>(props.maxThreadsDim[0]) ||
        local0 > static_cast<std::size_t>(props.maxThreadsDim[1])) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "2D local range exceeds HIP block limits");
    }

    const std::size_t groups0 = (global0 + local0 - 1) / local0;
    const std::size_t groups1 = (global1 + local1 - 1) / local1;
    const dim3 block(static_cast<unsigned int>(local1), static_cast<unsigned int>(local0));
    const dim3 grid(static_cast<unsigned int>(groups1), static_cast<unsigned int>(groups0));
    const sycl::range<2> launchedGroupRange(groups0, groups1);

    (void)hipGetLastError();
    gpu_execute_nd_range_2D_kernel<<<grid, block, sharedMemoryBytes, stream>>>(
        global0, global1, globalRange, localRange, launchedGroupRange, f);

    paras_rocm_detail::check(hipGetLastError(), "2D nd_range kernel launch failed");
    if (!async_launch()) {
        paras_rocm_detail::check(hipStreamSynchronize(stream), "2D nd_range kernel execution failed");
    }
}

template <typename Func>
void rocm_threadpool::paras_ext_nd_range_3D(const sycl::nd_range<3>& r, Func f,
                                              std::size_t sharedMemoryBytes) {
    paras_detail::validate_nd_range(r);
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
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D kernel received a zero global-range dimension");
    }

    if (localSize0 == 0 || localSize1 == 0 || localSize2 == 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D kernel received a zero local-range dimension");
    }

    if (localSize0 > std::numeric_limits<std::size_t>::max() / localSize1 ||
        localSize0 * localSize1 > std::numeric_limits<std::size_t>::max() / localSize2) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D local-range size overflow");
    }

    const std::size_t threadsPerBlock = localSize0 * localSize1 * localSize2;

    int currentDevice = -1;
    hipError_t err = hipGetDevice(&currentDevice);
    if (err != hipSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("hipGetDevice failed: ") + hipGetErrorString(err));
    }

    hipDeviceProp_t properties{};
    err = hipGetDeviceProperties(&properties, currentDevice);
    if (err != hipSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("hipGetDeviceProperties failed: ") +
                                  hipGetErrorString(err));
    }

    if (sharedMemoryBytes > static_cast<std::size_t>(properties.sharedMemPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                              "3D local_accessor allocation exceeds HIP shared memory per block");
    }

    if (threadsPerBlock > static_cast<std::size_t>(properties.maxThreadsPerBlock)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D local range exceeds HIP maxThreadsPerBlock");
    }

    if (localSize2 > static_cast<std::size_t>(properties.maxThreadsDim[0]) ||
        localSize1 > static_cast<std::size_t>(properties.maxThreadsDim[1]) ||
        localSize0 > static_cast<std::size_t>(properties.maxThreadsDim[2])) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D local range exceeds a HIP block-dimension limit");
    }

    const std::size_t groups0 = (globalSize0 + localSize0 - 1) / localSize0;
    const std::size_t groups1 = (globalSize1 + localSize1 - 1) / localSize1;
    const std::size_t groups2 = (globalSize2 + localSize2 - 1) / localSize2;

    if (groups2 > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        groups1 > static_cast<std::size_t>(properties.maxGridSize[1]) ||
        groups0 > static_cast<std::size_t>(properties.maxGridSize[2])) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                              "3D group range exceeds a HIP grid-dimension limit");
    }

    const dim3 block(static_cast<unsigned int>(localSize2), static_cast<unsigned int>(localSize1),
                     static_cast<unsigned int>(localSize0));

    const dim3 grid(static_cast<unsigned int>(groups2), static_cast<unsigned int>(groups1),
                    static_cast<unsigned int>(groups0));

    const sycl::range<3> launchedGroupRange(groups0, groups1, groups2);

    err = async_launch() ? hipSuccess : hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("HIP failure before 3D kernel launch: ") +
                                  hipGetErrorString(err));
    }

    (void)hipGetLastError();

    gpu_execute_nd_range_3D_kernel<<<grid, block, sharedMemoryBytes, stream>>>(
        globalSize0, globalSize1, globalSize2, globalRange, localRange, launchedGroupRange, f);

    err = hipGetLastError();
    if (err != hipSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("3D kernel launch failed: ") + hipGetErrorString(err));
    }

    err = async_launch() ? hipSuccess : hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("3D kernel execution failed: ") + hipGetErrorString(err));
    }
}
#endif

#endif
