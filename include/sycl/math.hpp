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

#ifndef __PARAS_MATH_HPP__
#define __PARAS_MATH_HPP__

#include "kem_gpu/gpu_utilities.hpp"
#include "utilities/internal_utils.hpp"
#include <algorithm>
#include <cmath>

namespace sycl {

PARAS_KERNEL_HD
inline double sin(double x) { return ::sin(x); }

PARAS_KERNEL_HD
inline double cos(double x) { return ::cos(x); }

PARAS_KERNEL_HD
inline double erf(double x) { return ::erf(x); }

PARAS_KERNEL_HD
inline double tan(double x) { return ::tan(x); }

PARAS_KERNEL_HD
inline double sqrt(double x) { return ::sqrt(x); }

PARAS_KERNEL_HD
inline float sqrt(float x) { return ::sqrtf(x); }

#if PARAS_GPU_BACKEND
PARAS_KERNEL_HD
inline double rsqrt(double x) { return ::rsqrt(x); }

PARAS_KERNEL_HD
inline float rsqrt(float x) { return ::rsqrtf(x); }
#else
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }

inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
#endif

PARAS_KERNEL_HD
inline double dot(double *a, double *b, size_t n) {
  double result = 0.0;
  for (size_t i = 0; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

PARAS_KERNEL_HD
inline double exp(double x) { return ::exp(x); }

PARAS_KERNEL_HD
inline double acos(double x) { return ::acos(x); }

PARAS_KERNEL_HD
inline double atan(double x) { return ::atan(x); }

PARAS_KERNEL_HD
inline double atan2(double y, double x) { return ::atan2(y, x); }

PARAS_KERNEL_HD
inline double asin(double x) { return ::asin(x); }

PARAS_KERNEL_HD
inline double sinh(double x) { return ::sinh(x); }

PARAS_KERNEL_HD
inline double asinh(double x) { return ::asinh(x); }

PARAS_KERNEL_HD
inline double acosh(double x) { return ::acosh(x); }

PARAS_KERNEL_HD
inline double cosh(double x) { return ::cosh(x); }

PARAS_KERNEL_HD
inline double rint(double x) { return ::rint(x); }

PARAS_KERNEL_HD
inline float rint(float x) { return ::rintf(x); }

PARAS_KERNEL_HD
inline bool isfinite(double x) { return std::isfinite(x); }

PARAS_KERNEL_HD
inline bool isfinite(float x) { return std::isfinite(x); }

template <typename T> PARAS_KERNEL_HD inline T tanh(T x) {
#if PARAS_GPU_BACKEND
  return ::tanh(x);
#else
  using std::tanh;
  return tanh(x);
#endif
}

template <typename T, typename U, typename V>
PARAS_KERNEL_HD inline std::enable_if_t<
    std::is_floating_point_v<std::common_type_t<T, U, V>>,
    std::common_type_t<T, U, V>>
fma(T x, U y, V z) {
  using R = std::common_type<T, U, V> ;

  R a = static_cast<R>(x) ;
  R b = static_cast<R>(y) ;
  R c = static_cast<R>(z) ;

  if constexpr (std::is_same_v<R, float>) {
    return ::fmaf(a, b, c) ;
  } else {
    return ::fma(a, b, c) ;
  }
}

template <typename T, typename U>
PARAS_KERNEL_HD inline std::enable_if_t<
    std::is_floating_point_v<std::common_type_t<T, U>>,
    std::common_type_t<T, U>>
fmax(T x, U y) {
  using R = std::common_type_t<T, U>;

  R a = static_cast<R>(x) ;
  R b = static_cast<R>(y) ;

  if constexpr (std::is_same_v<R, float>) {
    return ::fmaxf(a, b);
  } else {
    return ::fmax(a, b);
  }
}

template <typename T, typename U>
PARAS_KERNEL_HD inline std::enable_if_t<
    std::is_floating_point_v<std::common_type_t<T, U>>,
    std::common_type_t<T, U>>
fmin(T x, U y) {
  using R = std::common_type_t<T, U>;

  R a = static_cast<R>(x) ;
  R b = static_cast<R>(y) ; 

  if constexpr (std::is_same_v<R, float>) {
    return ::fminf(a, b);
  } else {
    return ::fmin(a, b);
  }
}

template <typename T, typename U>
PARAS_KERNEL_HD inline std::enable_if_t<
    std::is_floating_point_v<std::common_type_t<T, U>>,
    std::common_type_t<T, U>>
fdim(T x, U y) {
  using R = std::common_type_t<T, U>;

  if constexpr (std::is_same_v<R, float>) {
    return ::fdimf(static_cast<R>(x), static_cast<R>(y));
  } else {
    return ::fdim(static_cast<R>(x), static_cast<R>(y));
  }
}

PARAS_KERNEL_HD
inline int popcount(unsigned int x) {
  int count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }

  return count;
}

template <typename T, typename U> PARAS_KERNEL_HD inline auto max(T a, U b) {
  using type = std::common_type_t<T, U>;
  const type x = static_cast<type>(a);
  const type y = static_cast<type>(b);
  return (x > y) ? x : y;
}

template <typename T, typename U> PARAS_KERNEL_HD inline auto min(T a, U b) {
  using type = std::common_type_t<T, U>;
  const type x = static_cast<type>(a);
  const type y = static_cast<type>(b);
  return (x < y) ? x : y;
}

PARAS_KERNEL_HD
inline float logf(float x) { return ::logf(x); }

PARAS_KERNEL_HD
inline double log(double x) { return ::log(x); }

template <typename T> PARAS_KERNEL_HD inline T log10(T x) {
#if PARAS_GPU_BACKEND
  return ::log10(x);
#else
  using std::log10;
  return log10(x);
#endif
}

template <typename T> PARAS_KERNEL_HD inline T log1p(T x) {
#if PARAS_GPU_BACKEND
  return ::log1p(x);
#else
  using std::log1p;
  return log1p(x);
#endif
}

template <typename T> PARAS_KERNEL_HD inline T log2(T x) {
#if PARAS_GPU_BACKEND
  return ::log2(x);
#else
  using std::log2;
  return log2(x);
#endif
}

template <typename T> PARAS_KERNEL_HD inline T logb(T x) {
#if PARAS_GPU_BACKEND
  return ::logb(x);
#else
  using std::logb;
  return logb(x);
#endif
}

PARAS_KERNEL_HD
inline double exp2(double x) { return ::exp2(x); }

PARAS_KERNEL_HD
inline float exp2(float x) { return ::exp2f(x); }

PARAS_KERNEL_HD
inline double exp10(double x) {
#if PARAS_GPU_BACKEND
  return ::exp10(x);
#else
  return ::pow(10.0, x);
#endif
}

PARAS_KERNEL_HD
inline float exp10(float x) {
#if PARAS_GPU_BACKEND
  return ::exp10f(x);
#else
  return ::powf(10.0f, x);
#endif
}

PARAS_KERNEL_HD
inline double expm1(double x) { return ::expm1(x); }

PARAS_KERNEL_HD
inline float expm1(float x) { return ::expm1f(x); }

PARAS_KERNEL_HD
inline double pow(double x, double y) { return ::pow(x, y); }

PARAS_KERNEL_HD
inline float pow(float x, float y) { return ::powf(x, y); }

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T ceil(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::ceilf(x);
  } else {
    return ::ceil(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T floor(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::floorf(x);
  } else {
    return ::floor(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T trunc(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::truncf(x);
  } else {
    return ::trunc(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T round(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::roundf(x);
  } else {
    return ::round(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T nearbyint(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::nearbyintf(x);
  } else {
    return ::nearbyint(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T llrint(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::llrintf(x);
  } else {
    return ::llrint(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T llround(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::llroundf(x);
  } else {
    return ::llround(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T lrint(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::lrintf(x);
  } else {
    return ::lrint(x);
  }
}

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                               std::is_same_v<std::remove_cv_t<T>, double>,
                           int> = 0>
PARAS_KERNEL_HD inline T lround(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return ::lroundf(x);
  } else {
    return ::lround(x);
  }
}

template <typename T> struct plus {
  PARAS_KERNEL_HD
  constexpr T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T> struct minimum {
  PARAS_KERNEL_HD
  constexpr T operator()(const T &a, const T &b) const {
    return (a < b) ? a : b;
  }
};

template <typename T> struct maximum {
  PARAS_KERNEL_HD
  constexpr T operator()(const T &a, const T &b) const {
    return (a > b) ? a : b;
  }
};

} // namespace sycl

#endif
