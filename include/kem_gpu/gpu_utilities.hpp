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

#ifndef PARAS_KERNEL_HD
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#define PARAS_KERNEL_HD __host__ __device__
#else
#define PARAS_KERNEL_HD
#endif
#endif

PARAS_KERNEL_HD
inline unsigned char *paras_get_dynamic_shared_memory() noexcept {
#if defined(__CUDA_ARCH__)
  extern __shared__ unsigned char paras_dynamic_shared_memory[];
  return paras_dynamic_shared_memory;
#else
  return nullptr;
#endif
}

#ifndef PARAS_GPU_BACKEND
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__NVPTX__)  
    
#define PARAS_GPU_BACKEND 1
#else
#define PARAS_GPU_BACKEND 0
#endif
#endif

#ifndef __paras_if_target_host
#if PARAS_GPU_BACKEND
#define __paras_if_target_host(...)                                            \
  if constexpr (false) {                                                       \
    __VA_ARGS__                                                                \
  }
#else
#define ____paras_if_target_host(...)                                          \
  if constexpr (true) {                                                        \
    __VA_ARGS__                                                                \
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

#if defined(__CUDA_ARCH__)
#define PARAS_KERNEL_D __device__
#else
#define PARAS_KERNEL_D
#endif

PARAS_KERNEL_HD
inline bool paras_any_sync(unsigned mask, bool pred) {
#if defined(__CUDA_ARCH__)
  return __any_sync(mask, pred);
#else
  (void)mask;
  return pred;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_down(unsigned mask, T v, int delta) {
#if defined(__CUDA_ARCH__)
  return __shfl_down_sync(mask, v, delta);
#else
  return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_up(unsigned mask, T v, int delta) {
#if defined(__CUDA_ARCH__)
  return __shfl_up_sync(mask, v, delta);
#else
  return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl(unsigned mask, T v, int lane) {
#if defined(__CUDA_ARCH__)
  return __shfl_sync(mask, v, lane);
#else
  return v;
#endif
}

PARAS_KERNEL_HD
inline unsigned paras_active_mask() {
#if defined(__CUDA_ARCH__)
  return __activemask();
#else
  return 0xffffffffu;
#endif
}

PARAS_KERNEL_HD
inline void paras_syncwarp(unsigned mask = 0xffffffff) {
#if defined(__CUDA_ARCH__)
  __syncwarp(mask);
#else
  (void)mask;
#endif
}

PARAS_KERNEL_HD
inline void paras_syncthreads() {
#if defined(__CUDA_ARCH__)
  __syncthreads();
#else

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
PARAS_KERNEL_HD inline PointerType indexedAddress(PointerType address,
                                                  IndexType index) {
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


#endif // End of gpu_utilities
