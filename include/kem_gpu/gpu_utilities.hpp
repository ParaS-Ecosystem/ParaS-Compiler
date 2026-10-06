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

#ifndef __GPU_UTILITIES_HPP__
#define __GPU_UTILITIES_HPP__

#include <cstdint>
#include "kem/host_group_runtime.hpp" // host work-group barrier (no-op in device passes)

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

#ifndef PARAS_KERNEL_HD
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__HIPCC__)
#define PARAS_KERNEL_HD __host__ __device__
#else
#define PARAS_KERNEL_HD
#endif
#endif

PARAS_KERNEL_HD
inline unsigned char* paras_get_dynamic_shared_memory() noexcept {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    extern __shared__ unsigned char paras_dynamic_shared_memory[];
    return paras_dynamic_shared_memory;
#else
    return nullptr;
#endif
}

#ifndef PARAS_GPU_BACKEND
#if defined(__CUDA_ARCH__) || defined(__NVPTX__) || defined(__CUDACC__) || defined(__AMDGCN__) ||  \
    defined(__SPIRV__) || defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__) ||                 \
    defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
#define PARAS_GPU_BACKEND 1
#else
#define PARAS_GPU_BACKEND 0
#endif
#endif

#ifndef __paras_if_target_host
#if PARAS_GPU_BACKEND
#define __paras_if_target_host(...)                                                                \
    if constexpr (false) {                                                                         \
        __VA_ARGS__                                                                                \
    }
#else
#define ____paras_if_target_host(...)                                                              \
    if constexpr (true) {                                                                          \
        __VA_ARGS__                                                                                \
    }
#endif
#endif

#ifndef __paras_if_target_host
#if PARAS_GPU_BACKEND
#define __paras_if_target_host false
#else
#define __paras_if_target_host true
#endif
#endif

#if !defined(PARAS_AMD_WAVEFRONT_SIZE) && defined(__AMDGCN__)
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
#define PARAS_AMD_WAVEFRONT_SIZE 32
#else
#define PARAS_AMD_WAVEFRONT_SIZE 64
#endif
#endif
#if defined(__AMDGCN__) && defined(__HIP_DEVICE_COMPILE__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-pragma"
#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != PARAS_AMD_WAVEFRONT_SIZE
#error "PARAS_AMD_WAVEFRONT_SIZE does not match the target's wavefront size"
#endif
#pragma clang diagnostic pop
#endif

#ifndef PARAS_WARP_SIZE
#if defined(PARAS_AMD_WAVEFRONT_SIZE)
#define PARAS_WARP_SIZE PARAS_AMD_WAVEFRONT_SIZE
#elif defined(__HIP__) || defined(__HIPCC__)
#define PARAS_WARP_SIZE 64
#elif defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define PARAS_WARP_SIZE 32
#else
#define PARAS_WARP_SIZE 1
#endif
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_AMD__)
#define PARAS_KERNEL_D __device__
#else
#define PARAS_KERNEL_D
#endif

PARAS_KERNEL_HD
inline bool paras_any_sync(std::uint64_t mask, bool pred) {
#if defined(__CUDA_ARCH__)
    return __any_sync(static_cast<unsigned>(mask), pred);
#elif defined(__AMDGCN__)
    (void)mask;
#if PARAS_AMD_WAVEFRONT_SIZE == 32
    return __builtin_amdgcn_ballot_w32(pred) != 0;
#elif PARAS_AMD_WAVEFRONT_SIZE == 64
    return __builtin_amdgcn_ballot_w64(pred) != 0;
#else
#error "Unsupported AMDGCN wavefront size"
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
    (void)mask;
    return __any(pred);
#else
    (void)mask;
    return pred;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_down(std::uint64_t mask, T v, int delta) {
#if defined(__CUDA_ARCH__)
    return __shfl_down_sync(static_cast<unsigned>(mask), v, delta);
#elif defined(__AMDGCN__)
    (void)mask;
    return __shfl_down(v, delta);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __shfl_down_sync(mask, v, delta);
#else
    (void)mask;
    (void)delta;
    return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_up(std::uint64_t mask, T v, int delta) {
#if defined(__CUDA_ARCH__)
    return __shfl_up_sync(static_cast<unsigned>(mask), v, delta);
#elif defined(__AMDGCN__)
    (void)mask;
    return __shfl_up(v, delta);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __shfl_up_sync(mask, v, delta);
#else
    (void)mask;
    (void)delta;
    return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl(std::uint64_t mask, T v, int lane) {
#if defined(__CUDA_ARCH__)
    return __shfl_sync(static_cast<unsigned>(mask), v, lane);
#elif defined(__AMDGCN__)
    (void)mask;
    return __shfl(v, lane);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __shfl_sync(mask, v, lane);
#else
    (void)mask;
    (void)lane;
    return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_xor(std::uint64_t mask, T v, int lane_mask) {
#if defined(__CUDA_ARCH__)
    return __shfl_xor_sync(static_cast<unsigned>(mask), v, lane_mask);
#elif defined(__AMDGCN__)
    (void)mask;
    return __shfl_xor(v, lane_mask);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __shfl_xor_sync(mask, v, lane_mask);
#else
    (void)mask;
    (void)lane_mask;
    return v;
#endif
}

PARAS_KERNEL_HD
inline std::uint64_t paras_active_mask() {
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint64_t>(__activemask());

#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_NVIDIA__)
    return static_cast<std::uint64_t>(__activemask());

#elif defined(__AMDGCN__)
#if PARAS_AMD_WAVEFRONT_SIZE == 32
    return static_cast<std::uint64_t>(__builtin_amdgcn_read_exec_lo());
#elif PARAS_AMD_WAVEFRONT_SIZE == 64
    return static_cast<std::uint64_t>(__builtin_amdgcn_read_exec());
#else
#error "Unsupported AMDGCN wavefront size"
#endif

#else
    return 0xffffffffu;
#endif
}

PARAS_KERNEL_HD
inline void paras_syncwarp(std::uint64_t mask = 0xffffffffu) {
#if defined(__CUDA_ARCH__)
    __syncwarp(static_cast<unsigned>(mask));
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_NVIDIA__)
    __syncwarp(static_cast<unsigned>(mask));
#elif defined(__AMDGCN__)
    (void)mask;
    __builtin_amdgcn_wave_barrier();
#else
    (void)mask;
#endif
}

PARAS_KERNEL_HD
inline void paras_syncthreads() {
#if defined(__CUDA_ARCH__)
    __syncthreads();
#elif defined(__AMDGCN__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#elif defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
#else
    paras_host_detail::work_group_barrier();
#endif
}

#if PARAS_GPU_BACKEND

#ifdef GMX_DEVICE_ATTRIBUTE
#undef GMX_DEVICE_ATTRIBUTE
#endif
#define GMX_DEVICE_ATTRIBUTE __host__ __device__

#ifdef GMX_HOSTDEVICE_ATTRIBUTE
#undef GMX_HOSTDEVICE_ATTRIBUTE
#endif
#define GMX_HOSTDEVICE_ATTRIBUTE __host__ __device__

template <typename PointerType, typename IndexType, bool aligned>
PARAS_KERNEL_HD inline PointerType indexedAddress(PointerType address, IndexType index) {
    return address + index;
}

#ifdef GMX_ALWAYS_INLINE
#undef GMX_ALWAYS_INLINE
#endif
#define GMX_ALWAYS_INLINE inline

#endif

#ifndef PARAS_CUDA_BACKEND
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#define PARAS_CUDA_BACKEND 1
#else
#define PARAS_CUDA_BACKEND 0
#endif
#endif

#ifndef PARAS_HIP_BACKEND
#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define PARAS_HIP_BACKEND 1
#else
#define PARAS_HIP_BACKEND 0
#endif
#endif

#endif // End of gpu_utilities
