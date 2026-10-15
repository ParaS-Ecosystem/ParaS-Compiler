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
#include "kem/host_group_runtime.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define PARAS_DEVICE_PASS 1
#else
#define PARAS_DEVICE_PASS 0
#endif

#ifndef PARAS_KERNEL_HD
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__HIP__) ||      \
    defined(__HIPCC__)
#define PARAS_KERNEL_HD __host__ __device__
#else
#define PARAS_KERNEL_HD
#endif
#endif

PARAS_KERNEL_HD
inline unsigned char *paras_get_dynamic_shared_memory() noexcept {
#if PARAS_DEVICE_PASS
  extern __shared__ unsigned char paras_dynamic_shared_memory[];
  return paras_dynamic_shared_memory;
#else
  return nullptr;
#endif
}

#ifndef PARAS_GPU_BACKEND
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__NVPTX__) ||     \
    defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define PARAS_GPU_BACKEND 1
#else
#define PARAS_GPU_BACKEND 0
#endif
#endif

#ifndef PARAS_HIP_BACKEND
#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define PARAS_HIP_BACKEND 1
#else
#define PARAS_HIP_BACKEND 0
#endif
#endif

#ifndef __paras_if_target_host
#if PARAS_GPU_BACKEND
#define __paras_if_target_host(...)                                            \
  if constexpr (false) {                                                       \
    __VA_ARGS__                                                                \
  }
#else
#define __paras_if_target_host(...)                                          \
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

#ifndef PARAS_AMDGCN_WAVE_SIZE
#  if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-pragma"
#  endif
#  if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ == 32
#    define PARAS_AMDGCN_WAVE_SIZE 32
#  elif defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ == 64
#    define PARAS_AMDGCN_WAVE_SIZE 64
#  elif defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE == 32
#    define PARAS_AMDGCN_WAVE_SIZE 32
#  elif defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE == 64
#    define PARAS_AMDGCN_WAVE_SIZE 64
#  elif defined(__GFX9__)
#    define PARAS_AMDGCN_WAVE_SIZE 64
#  elif defined(__AMDGCN__)
#    define PARAS_AMDGCN_WAVE_SIZE 32
#  else
#    define PARAS_AMDGCN_WAVE_SIZE 0
#  endif
#  if defined(__clang__)
#    pragma clang diagnostic pop
#  endif
#endif

#ifndef PARAS_WARP_SIZE
#  if PARAS_AMDGCN_WAVE_SIZE
#    define PARAS_WARP_SIZE PARAS_AMDGCN_WAVE_SIZE
#  elif defined(__HIP__) || defined(__HIPCC__)
#    define PARAS_WARP_SIZE 64
#  elif defined(__CUDACC__) || defined(__CUDA_ARCH__)
#    define PARAS_WARP_SIZE 32
#  else
#    define PARAS_WARP_SIZE 1
#  endif
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
  #if !PARAS_AMDGCN_WAVE_SIZE
    #error "AMDGCN wavefront size is unavailable"
  #elif PARAS_AMDGCN_WAVE_SIZE == 32
    return __builtin_amdgcn_ballot_w32(pred) != 0;
  #elif PARAS_AMDGCN_WAVE_SIZE == 64
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
PARAS_KERNEL_HD inline T paras_shfl_down_native(std::uint64_t mask, T v, int delta) {
#if defined(__CUDA_ARCH__)
  return __shfl_down_sync(static_cast<unsigned>(mask), v, delta);
#elif defined(__AMDGCN__)

  (void)mask;
  return __shfl_down(v, delta);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl_down_sync(mask, v, delta);
#else
  return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_up_native(std::uint64_t mask, T v, int delta) {
#if defined(__CUDA_ARCH__)
  return __shfl_up_sync(static_cast<unsigned>(mask), v, delta);
#elif defined(__AMDGCN__)

  (void)mask;
  return __shfl_up(v, delta);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl_up_sync(mask, v, delta);
#else
  return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_native(std::uint64_t mask, T v, int lane) {
#if defined(__CUDA_ARCH__)
  return __shfl_sync(static_cast<unsigned>(mask), v, lane);
#elif defined(__AMDGCN__)

  (void)mask;
  return __shfl(v, lane);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl_sync(mask, v, lane);
#else
  return v;
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T paras_shfl_xor_native(std::uint64_t mask, T v, int lane_mask) {
#if defined(__CUDA_ARCH__)
  return __shfl_xor_sync(static_cast<unsigned>(mask), v, lane_mask);
#elif defined(__AMDGCN__)
  (void)mask;
  return __shfl_xor(v, lane_mask);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl_xor_sync(mask, v, lane_mask);
#else
  (void)mask; (void)lane_mask;
  return v;
#endif
}

template <typename T>
struct paras_shfl_native_type
    : std::integral_constant<bool,
          std::is_same<T, int>::value || std::is_same<T, unsigned>::value ||
          std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
          std::is_same<T, long long>::value ||
          std::is_same<T, unsigned long long>::value ||
          std::is_same<T, float>::value || std::is_same<T, double>::value> {};

template <typename T, typename Op>
PARAS_KERNEL_HD inline T paras_shfl_words(T v, Op op) {
  constexpr int N = static_cast<int>((sizeof(T) + 3) / 4);
  unsigned w[N] = {};
  __builtin_memcpy(w, &v, sizeof(T));
  for (int k = 0; k < N; ++k) w[k] = op(w[k]);
  T r = v;
  __builtin_memcpy(&r, w, sizeof(T));
  return r;
}

#define PARAS_SHFL_WRAP(NAME, ARG)                                            \
  template <typename T>                                                       \
  PARAS_KERNEL_HD inline T NAME(std::uint64_t mask, T v, int ARG) {           \
    if constexpr (paras_shfl_native_type<T>::value) {                         \
      return NAME##_native(mask, v, ARG);                                     \
    } else {                                                                  \
      return paras_shfl_words(v, [=](unsigned x) {                            \
        return NAME##_native(mask, x, ARG);                                   \
      });                                                                     \
    }                                                                         \
  }
PARAS_SHFL_WRAP(paras_shfl_down, delta)
PARAS_SHFL_WRAP(paras_shfl_up, delta)
PARAS_SHFL_WRAP(paras_shfl, lane)
PARAS_SHFL_WRAP(paras_shfl_xor, lane_mask)
#undef PARAS_SHFL_WRAP

PARAS_KERNEL_HD
inline std::uint64_t paras_active_mask() {
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint64_t>(__activemask());

#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_NVIDIA__)
  return static_cast<std::uint64_t>(__activemask());

#elif defined(__AMDGCN__)
  #if !PARAS_AMDGCN_WAVE_SIZE
    #error "AMDGCN wavefront size is unavailable"
  #elif PARAS_AMDGCN_WAVE_SIZE == 32
    return static_cast<std::uint64_t>(
        __builtin_amdgcn_read_exec_lo());
  #elif PARAS_AMDGCN_WAVE_SIZE == 64
    return static_cast<std::uint64_t>(
        __builtin_amdgcn_read_exec());
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
#if PARAS_DEVICE_PASS
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

#endif
