/**<
 * Copyright 2025 The ParaS-Compiler Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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

#ifndef PARAS_GPU_BACKEND
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__NVPTX__) ||     \
    defined(__AMDGCN__) || defined(__SPIRV__)
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
#define __paras_if_target_host(...)                                            \
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
PARAS_KERNEL_HD inline T paras_shfl(unsigned mask, T v, int lane) {
#if defined(__CUDA_ARCH__)
  return __shfl_sync(mask, v, lane);
#else
  return v;
#endif
}

PARAS_KERNEL_HD
inline void paras_syncwarp(unsigned mask = 0xffffffff) {
#if defined(__CUDA_ARCH__)
  __syncwarp(mask);
#else

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

#endif
