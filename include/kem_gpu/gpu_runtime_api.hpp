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

#ifndef __PARAS_GPU_RUNTIME_API_HPP__
#define __PARAS_GPU_RUNTIME_API_HPP__

#include "kem_gpu/gpu_utilities.hpp"

#if PARAS_HIP_BACKEND

#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaMemoryType hipMemoryType

#define cudaSuccess hipSuccess
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaErrorNotReady hipErrorNotReady
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeManaged hipMemoryTypeManaged

#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventQuery hipEventQuery
#define cudaEventDestroy hipEventDestroy
#define cudaMalloc hipMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define PARAS_GPU_SYMBOL(x) HIP_SYMBOL(x)
#define cudaPointerGetAttributes hipPointerGetAttributes

#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

#elif PARAS_CUDA_BACKEND

#define PARAS_GPU_SYMBOL(x) x

#include "kem_gpu/gpu_runtime_api.hpp"

#endif

#endif
