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

#include <algorithm>
#include <cmath>

#define PARAS_KERNEL_D
#define PARAS_KERNEL_HD

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

#if PARAS_GPU_BACKEND
PARAS_KERNEL_HD
inline double rsqrt(double x) { return ::rsqrt(x); }
#else
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
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
inline double isfinite(double x) { return std::isfinite(x); }

template <typename T> PARAS_KERNEL_HD inline T tanh(T x) {
#if PARAS_GPU_BACKEND
  return ::tanh(x);
#else
  using std::tanh;
  return tanh(x);
#endif
}

PARAS_KERNEL_HD
inline double fma(double x, double y, double z) { return ::fma(x, y, z); }

PARAS_KERNEL_HD
inline double fmax(double x, double y) { return ::fmax(x, y); }

PARAS_KERNEL_HD
inline double fmin(double x, double y) { return ::fmin(x, y); }

PARAS_KERNEL_HD
inline double fdim(double x, double y) { return ::fdim(x, y); }

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
  type x = static_cast<type>(x);
  type y = static_cast<type>(y);
  return (x > y) ? x : y;
}

template <typename T, typename U> PARAS_KERNEL_HD inline auto min(T a, U b) {
  using type = std::common_type_t<T, U>;
  type x = static_cast<type>(x);
  type y = static_cast<type>(y);
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
