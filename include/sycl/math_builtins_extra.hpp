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

#ifndef __PARAS_MATH_BUILTINS_EXTRA_HPP__
#define __PARAS_MATH_BUILTINS_EXTRA_HPP__

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <limits>

namespace sycl {

namespace detail {

template <typename T>
inline constexpr bool is_vector_like_v = is_vec_v<T> || is_marray_v<T>;

template <typename T, typename = void>
struct scalar_type_of {
  using type = T;
};
template <typename T>
struct scalar_type_of<T, std::enable_if_t<is_vector_like_v<T>>> {
  using type = typename T::value_type;
};
template <typename T>
using scalar_type_of_t = typename scalar_type_of<T>::type;

template <typename T>
inline constexpr bool is_valid_math_arg_v =
    is_vector_like_v<T> || std::is_arithmetic_v<T> || std::is_same_v<T, half>;

template <typename T>
inline constexpr bool is_valid_geninteger_arg_v =
    is_vector_like_v<T> || (std::is_arithmetic_v<T> && !std::is_floating_point_v<T> && !std::is_same_v<T, bool>);

template <typename S>
using wide_int_t = std::conditional_t<std::is_signed_v<S>, std::int64_t, std::uint64_t>;

template <typename S>
using same_size_uint_t = std::conditional_t<sizeof(S) == 4, std::uint32_t, std::uint64_t>;
}

#define PARAS_SYCL_UNARY_MATH(NAME, S, x, SCALAR_BODY)                 \
  template <typename T,                                                \
            typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<T>>> \
  PARAS_KERNEL_HD inline T NAME(T x_arg) {                             \
    if constexpr (detail::is_vector_like_v<T>) {                       \
      T result{};                                                      \
      for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) {  \
        result[paras_i] = NAME(x_arg[paras_i]);                        \
      }                                                                \
      return result;                                                   \
    } else {                                                           \
      using S = T;                                                     \
      S x = x_arg;                                                     \
      return (SCALAR_BODY);                                            \
    }                                                                  \
  }

#define PARAS_SYCL_UNARY_GENFLOAT(NAME, S, x, SCALAR_BODY)               \
  PARAS_KERNEL_HD inline float NAME(float x_arg) { using S = float; S x = x_arg; return (SCALAR_BODY); } \
  PARAS_KERNEL_HD inline double NAME(double x_arg) { using S = double; S x = x_arg; return (SCALAR_BODY); } \
  PARAS_KERNEL_HD inline half NAME(half x_arg) { using S = half; S x = x_arg; return (SCALAR_BODY); } \
  template <typename T,                                                  \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>> \
  PARAS_KERNEL_HD inline T NAME(T x_arg) {                                \
    if constexpr (detail::is_vector_like_v<T>) {                          \
      T result{};                                                         \
      for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i)       \
        result[paras_i] = NAME(x_arg[paras_i]);                           \
      return result;                                                      \
    } else {                                                              \
      using S = T;                                                        \
      S x = x_arg;                                                        \
      return (SCALAR_BODY);                                               \
    }                                                                     \
  }

#define PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(NAME, S, x, SCALAR_BODY)     \
  PARAS_KERNEL_HD inline half NAME(half x_arg) { using S = half; S x = x_arg; return (SCALAR_BODY); } \
  template <typename T,                                                  \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>> \
  PARAS_KERNEL_HD inline T NAME(T x_arg) {                                \
    if constexpr (detail::is_vector_like_v<T>) {                          \
      T result{};                                                         \
      for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i)       \
        result[paras_i] = NAME(x_arg[paras_i]);                           \
      return result;                                                      \
    } else {                                                              \
      using S = T;                                                        \
      S x = x_arg;                                                        \
      return (SCALAR_BODY);                                               \
    }                                                                     \
  }

PARAS_SYCL_UNARY_GENFLOAT(sin, S, x, static_cast<S>(std::sin(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(cos, S, x, static_cast<S>(std::cos(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(tan, S, x, static_cast<S>(std::tan(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(asin, S, x, static_cast<S>(std::asin(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(acos, S, x, static_cast<S>(std::acos(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(atan, S, x, static_cast<S>(std::atan(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(sinh, S, x, static_cast<S>(std::sinh(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(cosh, S, x, static_cast<S>(std::cosh(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(asinh, S, x, static_cast<S>(std::asinh(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(acosh, S, x, static_cast<S>(std::acosh(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(log, S, x, static_cast<S>(std::log(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(exp, S, x, static_cast<S>(std::exp(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(sqrt, S, x, static_cast<S>(std::sqrt(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(exp2, S, x, static_cast<S>(std::exp2(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(exp10, S, x, static_cast<S>(std::pow(10.0, static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(expm1, S, x, static_cast<S>(std::expm1(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(rsqrt, S, x, static_cast<S>(1.0 / std::sqrt(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(fabs, S, x, static_cast<S>(std::fabs(static_cast<double>(x))))

PARAS_SYCL_UNARY_GENFLOAT(sinpi, S, x, static_cast<S>(std::sin(static_cast<double>(x) * M_PI)))
PARAS_SYCL_UNARY_GENFLOAT(cospi, S, x, static_cast<S>(std::cos(static_cast<double>(x) * M_PI)))
PARAS_SYCL_UNARY_GENFLOAT(tanpi, S, x, static_cast<S>(std::tan(static_cast<double>(x) * M_PI)))
PARAS_SYCL_UNARY_GENFLOAT(acospi, S, x, static_cast<S>(std::acos(static_cast<double>(x)) / M_PI))
PARAS_SYCL_UNARY_GENFLOAT(asinpi, S, x, static_cast<S>(std::asin(static_cast<double>(x)) / M_PI))
PARAS_SYCL_UNARY_GENFLOAT(atanpi, S, x, static_cast<S>(std::atan(static_cast<double>(x)) / M_PI))
PARAS_SYCL_UNARY_GENFLOAT(cbrt, S, x, static_cast<S>(std::cbrt(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(atanh, S, x, static_cast<S>(std::atanh(static_cast<double>(x))))

PARAS_SYCL_UNARY_GENFLOAT(tanh, S, x, static_cast<S>(std::tanh(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(log10, S, x, static_cast<S>(std::log10(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(log1p, S, x, static_cast<S>(std::log1p(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(log2, S, x, static_cast<S>(std::log2(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(logb, S, x, static_cast<S>(std::logb(static_cast<double>(x))))

PARAS_SYCL_UNARY_GENFLOAT(degrees, S, x, static_cast<S>(static_cast<double>(x) * (180.0 / M_PI)))
PARAS_SYCL_UNARY_GENFLOAT(radians, S, x, static_cast<S>(static_cast<double>(x) * (M_PI / 180.0)))
PARAS_SYCL_UNARY_GENFLOAT(sign, S, x, (x > S(0)) ? S(1) : ((x < S(0)) ? S(-1) : S(0)))

PARAS_SYCL_UNARY_GENFLOAT(erf, S, x, static_cast<S>(std::erf(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(erfc, S, x, static_cast<S>(std::erfc(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(tgamma, S, x, static_cast<S>(std::tgamma(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(lgamma, S, x, static_cast<S>(std::lgamma(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT(round, S, x, static_cast<S>(std::round(static_cast<double>(x))))

PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(ceil, S, x, static_cast<S>(std::ceil(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(floor, S, x, static_cast<S>(std::floor(static_cast<double>(x))))
PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(trunc, S, x, static_cast<S>(std::trunc(static_cast<double>(x))))

PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY(rint, S, x, static_cast<S>(std::rint(static_cast<double>(x))))

#undef PARAS_SYCL_UNARY_GENFLOAT
#undef PARAS_SYCL_UNARY_GENFLOAT_HALF_ONLY

PARAS_SYCL_UNARY_MATH(clz, S, x, [&]() -> S {
  if (x == 0) return static_cast<S>(sizeof(S) * 8);
  using U = std::make_unsigned_t<S>;
  U u = static_cast<U>(x);
  S n = 0;
  U mask = static_cast<U>(U(1) << (sizeof(S) * 8 - 1));
  while ((u & mask) == 0) { ++n; mask >>= 1; }
  return n;
}())

PARAS_SYCL_UNARY_MATH(ctz, S, x, [&]() -> S {
  if (x == 0) return static_cast<S>(sizeof(S) * 8);
  using U = std::make_unsigned_t<S>;
  U u = static_cast<U>(x);
  S n = 0;
  while ((u & U(1)) == 0) { ++n; u >>= 1; }
  return n;
}())
PARAS_SYCL_UNARY_MATH(popcount, S, x, [&]() -> S {
  using U = std::make_unsigned_t<S>;
  U u = static_cast<U>(x);
  S n = 0;
  while (u) { n += static_cast<S>(u & U(1)); u >>= 1; }
  return n;
}())

#undef PARAS_SYCL_UNARY_MATH

#define PARAS_SYCL_BINARY_MATH(NAME, S, a, b, SCALAR_BODY)              \
  template <typename T,                                                 \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>> \
  PARAS_KERNEL_HD inline T NAME(T a_arg, T b_arg) {                     \
    if constexpr (detail::is_vector_like_v<T>) {                        \
      T result{};                                                       \
      for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) {   \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]);         \
      }                                                                 \
      return result;                                                    \
    } else {                                                            \
      using S = T;                                                      \
      S a = a_arg;                                                      \
      S b = b_arg;                                                      \
      return (SCALAR_BODY);                                             \
    }                                                                   \
  }

PARAS_KERNEL_HD inline float atan2(float a, float b) { return static_cast<float>(std::atan2(static_cast<double>(a), static_cast<double>(b))); }
PARAS_KERNEL_HD inline half atan2(half a, half b) { return static_cast<half>(std::atan2(static_cast<double>(a), static_cast<double>(b))); }
PARAS_KERNEL_HD inline float fmax(float a, float b) { return static_cast<float>(std::fmax(static_cast<double>(a), static_cast<double>(b))); }
PARAS_KERNEL_HD inline half fmax(half a, half b) { return static_cast<half>(std::fmax(static_cast<double>(a), static_cast<double>(b))); }
PARAS_KERNEL_HD inline float fmin(float a, float b) { return static_cast<float>(std::fmin(static_cast<double>(a), static_cast<double>(b))); }
PARAS_KERNEL_HD inline half fmin(half a, half b) { return static_cast<half>(std::fmin(static_cast<double>(a), static_cast<double>(b))); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 atan2(T1 a_arg, T2 b_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto b = [&]() { if constexpr (detail::is_vector_like_v<T2>) return b_arg[paras_i]; else return b_arg; }();
      result[paras_i] = atan2(a_arg[paras_i], b);
    }
    return result;
  } else {
    return atan2(a_arg, static_cast<T1>(b_arg));
  }
}
template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 fmax(T1 a_arg, T2 b_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto b = [&]() { if constexpr (detail::is_vector_like_v<T2>) return b_arg[paras_i]; else return b_arg; }();
      result[paras_i] = fmax(a_arg[paras_i], b);
    }
    return result;
  } else {
    return fmax(a_arg, static_cast<T1>(b_arg));
  }
}
template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 fmin(T1 a_arg, T2 b_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto b = [&]() { if constexpr (detail::is_vector_like_v<T2>) return b_arg[paras_i]; else return b_arg; }();
      result[paras_i] = fmin(a_arg[paras_i], b);
    }
    return result;
  } else {
    return fmin(a_arg, static_cast<T1>(b_arg));
  }
}

#undef PARAS_SYCL_BINARY_MATH

#define PARAS_SYCL_BINARY_MATH(NAME, S, a, b, SCALAR_BODY)               \
  PARAS_KERNEL_HD inline float NAME(float a_arg, float b_arg) {                          \
    using S = float; S a = a_arg; S b = b_arg; return (SCALAR_BODY);     \
  }                                                                       \
  PARAS_KERNEL_HD inline double NAME(double a_arg, double b_arg) {                       \
    using S = double; S a = a_arg; S b = b_arg; return (SCALAR_BODY);    \
  }                                                                       \
  PARAS_KERNEL_HD inline half NAME(half a_arg, half b_arg) {                             \
    using S = half; S a = a_arg; S b = b_arg; return (SCALAR_BODY);      \
  }                                                                       \
  template <typename T,                                                  \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>> \
  PARAS_KERNEL_HD inline T NAME(T a_arg, T b_arg) {                      \
    if constexpr (detail::is_vector_like_v<T>) {                         \
      T result{};                                                        \
      for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) {    \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]);          \
      }                                                                  \
      return result;                                                     \
    } else {                                                             \
      using S = T;                                                       \
      S a = a_arg;                                                       \
      S b = b_arg;                                                       \
      return (SCALAR_BODY);                                              \
    }                                                                    \
  }

PARAS_SYCL_BINARY_MATH(fmod, S, a, b, static_cast<S>(std::fmod(static_cast<double>(a), static_cast<double>(b))))
PARAS_SYCL_BINARY_MATH(hypot, S, a, b, static_cast<S>(std::hypot(static_cast<double>(a), static_cast<double>(b))))
PARAS_SYCL_BINARY_MATH(copysign, S, a, b, static_cast<S>(std::copysign(static_cast<double>(a), static_cast<double>(b))))

PARAS_SYCL_BINARY_MATH(atan2pi, S, a, b, static_cast<S>(std::atan2(static_cast<double>(a), static_cast<double>(b)) / M_PI))
PARAS_SYCL_BINARY_MATH(maxmag, S, a, b, (std::fabs(static_cast<double>(a)) > std::fabs(static_cast<double>(b))) ? a : ((std::fabs(static_cast<double>(a)) < std::fabs(static_cast<double>(b))) ? b : (a > b ? a : b)))
PARAS_SYCL_BINARY_MATH(minmag, S, a, b, (std::fabs(static_cast<double>(a)) < std::fabs(static_cast<double>(b))) ? a : ((std::fabs(static_cast<double>(a)) > std::fabs(static_cast<double>(b))) ? b : (a < b ? a : b)))
PARAS_SYCL_BINARY_MATH(remainder, S, a, b, static_cast<S>(std::remainder(static_cast<double>(a), static_cast<double>(b))))
PARAS_SYCL_BINARY_MATH(powr, S, a, b, static_cast<S>(std::pow(static_cast<double>(a), static_cast<double>(b))))

#undef PARAS_SYCL_BINARY_MATH

#define PARAS_SYCL_BINARY_MATH_HALF_ONLY(NAME, S, a, b, SCALAR_BODY)     \
  PARAS_KERNEL_HD inline half NAME(half a_arg, half b_arg) {                             \
    using S = half; S a = a_arg; S b = b_arg; return (SCALAR_BODY);      \
  }                                                                       \
  template <typename T,                                                  \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>> \
  PARAS_KERNEL_HD inline T NAME(T a_arg, T b_arg) {                      \
    if constexpr (detail::is_vector_like_v<T>) {                         \
      T result{};                                                        \
      for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i)      \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]);          \
      return result;                                                     \
    } else {                                                             \
      using S = T; S a = a_arg; S b = b_arg; return (SCALAR_BODY);       \
    }                                                                    \
  }

PARAS_SYCL_BINARY_MATH_HALF_ONLY(nextafter, S, a, b, static_cast<S>(std::nextafter(static_cast<double>(a), static_cast<double>(b))))
PARAS_SYCL_BINARY_MATH_HALF_ONLY(pow, S, a, b, static_cast<S>(std::pow(static_cast<double>(a), static_cast<double>(b))))

#undef PARAS_SYCL_BINARY_MATH_HALF_ONLY

PARAS_KERNEL_HD inline int ilogb(float x) { return std::ilogb(x); }
PARAS_KERNEL_HD inline int ilogb(double x) { return std::ilogb(x); }
PARAS_KERNEL_HD inline int ilogb(half x) { return std::ilogb(static_cast<float>(x)); }
template <typename T,
          typename = std::enable_if_t<detail::is_vector_like_v<T>>>
PARAS_KERNEL_HD inline auto ilogb(T x) {
  using E = typename T::value_type;
  using R = std::conditional_t<is_vec_v<T>, vec<std::int32_t, T::size()>, marray<int, T::size()>>;
  R result{};
  for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) result[paras_i] = ilogb(x[paras_i]);
  return result;
}

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 ldexp(T1 x_arg, T2 n_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto n = [&]() {
        if constexpr (detail::is_vector_like_v<T2>) return n_arg[paras_i];
        else return n_arg;
      }();
      result[paras_i] = ldexp(x_arg[paras_i], n);
    }
    return result;
  } else {
    return static_cast<T1>(std::ldexp(static_cast<double>(x_arg), static_cast<int>(n_arg)));
  }
}
PARAS_KERNEL_HD inline float ldexp(float x, int n) { return std::ldexp(x, n); }
PARAS_KERNEL_HD inline double ldexp(double x, int n) { return std::ldexp(x, n); }
PARAS_KERNEL_HD inline half ldexp(half x, int n) { return static_cast<half>(std::ldexp(static_cast<float>(x), n)); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 pown(T1 x_arg, T2 n_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto n = [&]() {
        if constexpr (detail::is_vector_like_v<T2>) return n_arg[paras_i];
        else return n_arg;
      }();
      result[paras_i] = pown(x_arg[paras_i], n);
    }
    return result;
  } else {
    return static_cast<T1>(std::pow(static_cast<double>(x_arg), static_cast<double>(static_cast<int>(n_arg))));
  }
}
PARAS_KERNEL_HD inline float pown(float x, int n) { return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(n))); }
PARAS_KERNEL_HD inline double pown(double x, int n) { return std::pow(x, static_cast<double>(n)); }
PARAS_KERNEL_HD inline half pown(half x, int n) { return static_cast<half>(std::pow(static_cast<double>(x), static_cast<double>(n))); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 rootn(T1 x_arg, T2 n_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto n = [&]() {
        if constexpr (detail::is_vector_like_v<T2>) return n_arg[paras_i];
        else return n_arg;
      }();
      result[paras_i] = rootn(x_arg[paras_i], n);
    }
    return result;
  } else {
    return static_cast<T1>(std::pow(static_cast<double>(x_arg), 1.0 / static_cast<double>(static_cast<int>(n_arg))));
  }
}
PARAS_KERNEL_HD inline float rootn(float x, int n) { return static_cast<float>(std::pow(static_cast<double>(x), 1.0 / static_cast<double>(n))); }
PARAS_KERNEL_HD inline double rootn(double x, int n) { return std::pow(x, 1.0 / static_cast<double>(n)); }
PARAS_KERNEL_HD inline half rootn(half x, int n) { return static_cast<half>(std::pow(static_cast<double>(x), 1.0 / static_cast<double>(n))); }

#define PARAS_SYCL_PTR_OUTPUT_UNARY(NAME, SCALAR_BODY)                   \
  template <typename Ptr, typename T,                                    \
            typename = std::enable_if_t<!detail::is_vector_like_v<T>>>   \
  PARAS_KERNEL_HD inline T NAME(T x_arg, Ptr ptr_arg) { SCALAR_BODY }                    \
  template <typename T, typename Ptr,                                    \
            typename = std::enable_if_t<detail::is_vector_like_v<T>>>    \
  PARAS_KERNEL_HD inline T NAME(T x_arg, Ptr ptr_arg) {                         \
    T result{};                                                           \
    for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) {       \
      result[paras_i] = NAME(x_arg[paras_i], &(*ptr_arg)[paras_i]);       \
    }                                                                     \
    return result;                                                        \
  }

PARAS_SYCL_PTR_OUTPUT_UNARY(frexp, {
  int e;
  T m = static_cast<T>(std::frexp(static_cast<double>(x_arg), &e));
  *ptr_arg = static_cast<typename std::remove_reference_t<decltype(*ptr_arg)>>(e);
  return m;
})

PARAS_SYCL_PTR_OUTPUT_UNARY(fract, {
  double ip;
  double f = std::modf(static_cast<double>(x_arg), &ip);
  if (f < 0) { f += 1.0; ip -= 1.0; }
  *ptr_arg = static_cast<typename std::remove_reference_t<decltype(*ptr_arg)>>(ip);
  return static_cast<T>(f);
})

PARAS_SYCL_PTR_OUTPUT_UNARY(modf, {
  double ip;
  double f = std::modf(static_cast<double>(x_arg), &ip);
  *ptr_arg = static_cast<typename std::remove_reference_t<decltype(*ptr_arg)>>(ip);
  return static_cast<T>(f);
})

PARAS_SYCL_PTR_OUTPUT_UNARY(lgamma_r, {
  T result = static_cast<T>(std::lgamma(static_cast<double>(x_arg)));
  int sign = (std::tgamma(static_cast<double>(x_arg)) < 0.0) ? -1 : 1;
  *ptr_arg = static_cast<typename std::remove_reference_t<decltype(*ptr_arg)>>(sign);
  return result;
})

PARAS_SYCL_PTR_OUTPUT_UNARY(sincos, {
  *ptr_arg = static_cast<typename std::remove_reference_t<decltype(*ptr_arg)>>(std::cos(static_cast<double>(x_arg)));
  return static_cast<T>(std::sin(static_cast<double>(x_arg)));
})

#undef PARAS_SYCL_PTR_OUTPUT_UNARY

template <typename Ptr, typename T,
          typename = std::enable_if_t<!detail::is_vector_like_v<T>>>
PARAS_KERNEL_HD inline T remquo(T x, T y, Ptr quo) {
  int q;
  T result = static_cast<T>(std::remquo(static_cast<double>(x), static_cast<double>(y), &q));
  *quo = static_cast<typename std::remove_reference_t<decltype(*quo)>>(q);
  return result;
}

template <typename T1, typename T2, typename Ptr,
          typename = std::enable_if_t<detail::is_vector_like_v<T1>>>
PARAS_KERNEL_HD inline T1 remquo(T1 x, T2 y, Ptr quo) {
  T1 result{};
  for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
    result[paras_i] = remquo(x[paras_i], y[paras_i], &(*quo)[paras_i]);
  }
  return result;
}

PARAS_KERNEL_HD inline float nan(std::uint32_t) { return std::numeric_limits<float>::quiet_NaN(); }
PARAS_KERNEL_HD inline double nan(std::uint64_t) { return std::numeric_limits<double>::quiet_NaN(); }
PARAS_KERNEL_HD inline half nan(std::uint16_t) { return half(std::numeric_limits<float>::quiet_NaN()); }

template <typename T,
          typename = std::enable_if_t<
              detail::is_vector_like_v<T> ||
              std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::uint64_t>>>
PARAS_KERNEL_HD inline auto nan(T nancode) {
  if constexpr (detail::is_vector_like_v<T>) {
    using E = typename T::value_type;
    using R = std::conditional_t<sizeof(E) == 2, half, std::conditional_t<sizeof(E) == 4, float, double>>;
    using Result = std::conditional_t<is_vec_v<T>, vec<R, T::size()>, marray<R, T::size()>>;
    Result result{};
    for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) result[paras_i] = nan(nancode[paras_i]);
    return result;
  } else if constexpr (sizeof(T) == 8) {
    return nan(static_cast<std::uint64_t>(nancode));
  } else if constexpr (sizeof(T) == 2) {
    return nan(static_cast<std::uint16_t>(nancode));
  } else {
    return nan(static_cast<std::uint32_t>(nancode));
  }
}

PARAS_KERNEL_HD inline float fdim(float x, float y) { return static_cast<float>(std::fdim(static_cast<double>(x), static_cast<double>(y))); }
PARAS_KERNEL_HD inline half fdim(half x, half y) { return static_cast<half>(std::fdim(static_cast<double>(x), static_cast<double>(y))); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 fdim(T1 x_arg, T2 y_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      result[paras_i] = fdim(x_arg[paras_i], y_arg[paras_i]);
    }
    return result;
  } else {
    return fdim(x_arg, static_cast<T1>(y_arg));
  }
}

#define PARAS_SYCL_BINARY_MATH_2T(NAME, S, a, b, SCALAR_BODY)            \
  template <typename T1, typename T2,                                    \
            typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<T1> && \
                                         detail::is_valid_geninteger_arg_v<T2>>> \
  PARAS_KERNEL_HD inline T1 NAME(T1 a_arg, T2 b_arg) {                   \
    if constexpr (detail::is_vector_like_v<T1>) {                        \
      T1 result{};                                                       \
      for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {   \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]);          \
      }                                                                  \
      return result;                                                     \
    } else {                                                             \
      using S = T1;                                                      \
      S a = a_arg;                                                       \
      S b = static_cast<S>(b_arg);                                       \
      return (SCALAR_BODY);                                              \
    }                                                                    \
  }

PARAS_SYCL_BINARY_MATH_2T(abs_diff, S, a, b, [&]() -> S {
  using U = std::make_unsigned_t<S>;
  U ua = static_cast<U>(a); U ub = static_cast<U>(b);
  return static_cast<S>(ua > ub ? ua - ub : ub - ua);
}())
PARAS_SYCL_BINARY_MATH_2T(add_sat, S, a, b, [&]() -> S {
  using Wide = detail::wide_int_t<S>;
  Wide r = static_cast<Wide>(a) + static_cast<Wide>(b);
  Wide lo = static_cast<Wide>(std::numeric_limits<S>::min());
  Wide hi = static_cast<Wide>(std::numeric_limits<S>::max());
  return static_cast<S>(r < lo ? lo : (r > hi ? hi : r));
}())
PARAS_SYCL_BINARY_MATH_2T(sub_sat, S, a, b, [&]() -> S {
  using Wide = detail::wide_int_t<S>;
  Wide r = static_cast<Wide>(a) - static_cast<Wide>(b);
  Wide lo = static_cast<Wide>(std::numeric_limits<S>::min());
  Wide hi = static_cast<Wide>(std::numeric_limits<S>::max());
  return static_cast<S>(r < lo ? lo : (r > hi ? hi : r));
}())
PARAS_SYCL_BINARY_MATH_2T(hadd, S, a, b, [&]() -> S {
  using Wide = detail::wide_int_t<S>;
  return static_cast<S>((static_cast<Wide>(a) + static_cast<Wide>(b)) >> 1);
}())
PARAS_SYCL_BINARY_MATH_2T(rhadd, S, a, b, [&]() -> S {
  using Wide = detail::wide_int_t<S>;
  return static_cast<S>((static_cast<Wide>(a) + static_cast<Wide>(b) + 1) >> 1);
}())
PARAS_SYCL_BINARY_MATH_2T(mul_hi, S, a, b, [&]() -> S {
  using Wide = detail::wide_int_t<S>;
  Wide r = static_cast<Wide>(a) * static_cast<Wide>(b);
  return static_cast<S>(r >> (sizeof(S) * 8));
}())

template <typename T1, typename T2, typename T3,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1> &&
                                       detail::is_valid_math_arg_v<T2> &&
                                       detail::is_valid_math_arg_v<T3>>>
PARAS_KERNEL_HD inline T1 clamp(T1 x_arg, T2 minval_arg, T3 maxval_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto lo = [&]() {
        if constexpr (detail::is_vector_like_v<T2>) return minval_arg[paras_i];
        else return minval_arg;
      }();
      auto hi = [&]() {
        if constexpr (detail::is_vector_like_v<T3>) return maxval_arg[paras_i];
        else return maxval_arg;
      }();
      result[paras_i] = clamp(x_arg[paras_i], lo, hi);
    }
    return result;
  } else {
    T1 x = x_arg;
    T1 minval = static_cast<T1>(minval_arg);
    T1 maxval = static_cast<T1>(maxval_arg);
    return (x < minval ? minval : (x > maxval ? maxval : x));
  }
}

PARAS_KERNEL_HD inline float mix(float x, float y, float a) { return x + (y - x) * a; }
PARAS_KERNEL_HD inline double mix(double x, double y, double a) { return x + (y - x) * a; }
PARAS_KERNEL_HD inline half mix(half x, half y, half a) {
  float xf = x, yf = y, af = a;
  return static_cast<half>(xf + (yf - xf) * af);
}
PARAS_KERNEL_HD inline float smoothstep(float edge0, float edge1, float x) {
  float t = (x - edge0) / (edge1 - edge0);
  t = (t < 0.0f) ? 0.0f : ((t > 1.0f) ? 1.0f : t);
  return t * t * (3.0f - 2.0f * t);
}
PARAS_KERNEL_HD inline double smoothstep(double edge0, double edge1, double x) {
  double t = (x - edge0) / (edge1 - edge0);
  t = (t < 0.0) ? 0.0 : ((t > 1.0) ? 1.0 : t);
  return t * t * (3.0 - 2.0 * t);
}
PARAS_KERNEL_HD inline half smoothstep(half edge0, half edge1, half x) {
  float e0 = edge0, e1 = edge1, xf = x;
  float t = (xf - e0) / (e1 - e0);
  t = (t < 0.0f) ? 0.0f : ((t > 1.0f) ? 1.0f : t);
  return static_cast<half>(t * t * (3.0f - 2.0f * t));
}
PARAS_KERNEL_HD inline float step(float edge, float x) { return (x < edge) ? 0.0f : 1.0f; }
PARAS_KERNEL_HD inline double step(double edge, double x) { return (x < edge) ? 0.0 : 1.0; }
PARAS_KERNEL_HD inline half step(half edge, half x) { return (static_cast<float>(x) < static_cast<float>(edge)) ? half(0.0f) : half(1.0f); }

template <typename T1, typename T2, typename T3,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 mix(T1 x_arg, T2 y_arg, T3 a_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto a = [&]() {
        if constexpr (detail::is_vector_like_v<T3>) return a_arg[paras_i];
        else return a_arg;
      }();
      result[paras_i] = mix(x_arg[paras_i], y_arg[paras_i], a);
    }
    return result;
  } else {
    T1 x = x_arg;
    T1 y = static_cast<T1>(y_arg);
    T1 a = static_cast<T1>(a_arg);
    return x + (y - x) * a;
  }
}

template <typename T3, typename T1, typename T2,
          typename = std::enable_if_t<detail::is_vector_like_v<T3> && !detail::is_vector_like_v<T1>>>
PARAS_KERNEL_HD inline T3 smoothstep(T1 edge0_arg, T2 edge1_arg, T3 x_arg) {
  T3 result{};
  for (std::size_t paras_i = 0; paras_i < T3::size(); ++paras_i) {
    auto e1 = [&]() {
      if constexpr (detail::is_vector_like_v<T2>) return edge1_arg[paras_i];
      else return edge1_arg;
    }();
    result[paras_i] = smoothstep(edge0_arg, e1, x_arg[paras_i]);
  }
  return result;
}

template <typename T1, typename T2, typename T3,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T3> && (!detail::is_vector_like_v<T3> || detail::is_vector_like_v<T1>)>>
PARAS_KERNEL_HD inline T3 smoothstep(T1 edge0_arg, T2 edge1_arg, T3 x_arg) {
  if constexpr (detail::is_vector_like_v<T3>) {
    T3 result{};
    for (std::size_t paras_i = 0; paras_i < T3::size(); ++paras_i) {
      auto e0 = edge0_arg[paras_i];
      auto e1 = [&]() {
        if constexpr (detail::is_vector_like_v<T2>) return edge1_arg[paras_i];
        else return edge1_arg;
      }();
      result[paras_i] = smoothstep(e0, e1, x_arg[paras_i]);
    }
    return result;
  } else {
    using S = T3;
    S edge0 = static_cast<S>(edge0_arg);
    S edge1 = static_cast<S>(edge1_arg);
    S x = x_arg;
    S t = (x - edge0) / (edge1 - edge0);
    t = (t < S(0)) ? S(0) : ((t > S(1)) ? S(1) : t);
    return t * t * (S(3) - S(2) * t);
  }
}

template <typename T2, typename T1,
          typename = std::enable_if_t<detail::is_vector_like_v<T2> && !detail::is_vector_like_v<T1>>>
PARAS_KERNEL_HD inline T2 step(T1 edge_arg, T2 x_arg) {
  T2 result{};
  for (std::size_t paras_i = 0; paras_i < T2::size(); ++paras_i) {
    result[paras_i] = step(edge_arg, x_arg[paras_i]);
  }
  return result;
}

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T2> && (!detail::is_vector_like_v<T2> || detail::is_vector_like_v<T1>)>>
PARAS_KERNEL_HD inline T2 step(T1 edge_arg, T2 x_arg) {
  if constexpr (detail::is_vector_like_v<T2>) {
    T2 result{};
    for (std::size_t paras_i = 0; paras_i < T2::size(); ++paras_i) {
      result[paras_i] = step(edge_arg[paras_i], x_arg[paras_i]);
    }
    return result;
  } else {
    T2 edge = static_cast<T2>(edge_arg);
    T2 x = x_arg;
    return (x < edge) ? T2(0) : T2(1);
  }
}

#define PARAS_SYCL_TERNARY_MATH_3T(NAME, S, a, b, c, SCALAR_BODY)        \
  template <typename T1, typename T2, typename T3,                       \
            typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<T1> && \
                                         detail::is_valid_geninteger_arg_v<T2> && \
                                         detail::is_valid_geninteger_arg_v<T3>>> \
  PARAS_KERNEL_HD inline T1 NAME(T1 a_arg, T2 b_arg, T3 c_arg) {         \
    if constexpr (detail::is_vector_like_v<T1>) {                        \
      T1 result{};                                                       \
      for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {   \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i], c_arg[paras_i]); \
      }                                                                  \
      return result;                                                     \
    } else {                                                             \
      using S = T1;                                                      \
      S a = a_arg;                                                       \
      S b = static_cast<S>(b_arg);                                       \
      S c = static_cast<S>(c_arg);                                       \
      return (SCALAR_BODY);                                              \
    }                                                                    \
  }

PARAS_SYCL_TERNARY_MATH_3T(mad_sat, S, a, b, c, [&]() -> S {
  using Wide = detail::wide_int_t<S>;
  Wide r = static_cast<Wide>(a) * static_cast<Wide>(b) + static_cast<Wide>(c);
  Wide lo = static_cast<Wide>(std::numeric_limits<S>::min());
  Wide hi = static_cast<Wide>(std::numeric_limits<S>::max());
  return static_cast<S>(r < lo ? lo : (r > hi ? hi : r));
}())
PARAS_SYCL_TERNARY_MATH_3T(mad_hi, S, a, b, c, [&]() -> S {
  using Wide = detail::wide_int_t<S>;
  Wide r = static_cast<Wide>(a) * static_cast<Wide>(b);
  return static_cast<S>((r >> (sizeof(S) * 8)) + static_cast<Wide>(c));
}())

PARAS_SYCL_TERNARY_MATH_3T(mad24, S, a, b, c, static_cast<S>(a * b + c))

#define PARAS_SYCL_MUL24(NAME)                                           \
  template <typename T1, typename T2,                                    \
            typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<T1>>> \
  PARAS_KERNEL_HD inline T1 NAME(T1 a_arg, T2 b_arg) {                   \
    if constexpr (detail::is_vector_like_v<T1>) {                        \
      T1 result{};                                                       \
      for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i)     \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]);          \
      return result;                                                     \
    } else {                                                             \
      return static_cast<T1>(a_arg * static_cast<T1>(b_arg));            \
    }                                                                    \
  }
PARAS_SYCL_MUL24(mul24)
#undef PARAS_SYCL_MUL24

namespace detail {
template <typename Hi>
using upsampled_t = std::conditional_t<
    sizeof(Hi) == 1, std::conditional_t<std::is_signed_v<Hi>, std::int16_t, std::uint16_t>,
    std::conditional_t<sizeof(Hi) == 2, std::conditional_t<std::is_signed_v<Hi>, std::int32_t, std::uint32_t>,
    std::conditional_t<std::is_signed_v<Hi>, std::int64_t, std::uint64_t>>>;
}

template <typename Hi, typename Lo,
          typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<Hi>>>
PARAS_KERNEL_HD inline auto upsample(Hi hi, Lo lo) {
  if constexpr (detail::is_vector_like_v<Hi>) {
    using EHi = typename Hi::value_type;
    using EResult = detail::upsampled_t<EHi>;
    using R = std::conditional_t<is_vec_v<Hi>, vec<EResult, Hi::size()>, marray<EResult, Hi::size()>>;
    R result{};
    for (std::size_t paras_i = 0; paras_i < Hi::size(); ++paras_i) result[paras_i] = upsample(hi[paras_i], lo[paras_i]);
    return result;
  } else {
    using R = detail::upsampled_t<Hi>;
    using U = std::make_unsigned_t<R>;
    constexpr unsigned lo_bits = sizeof(Hi) * 8;
    U u = (static_cast<U>(static_cast<std::make_unsigned_t<Hi>>(hi)) << lo_bits) | static_cast<U>(lo);
    return static_cast<R>(u);
  }
}

PARAS_KERNEL_HD inline float fma(float a, float b, float c) { return static_cast<float>(std::fma(static_cast<double>(a), static_cast<double>(b), static_cast<double>(c))); }
PARAS_KERNEL_HD inline half fma(half a, half b, half c) { return static_cast<half>(std::fma(static_cast<double>(a), static_cast<double>(b), static_cast<double>(c))); }

template <typename T1, typename T2, typename T3,
          typename = std::enable_if_t<detail::is_vector_like_v<T1>>>
PARAS_KERNEL_HD inline T1 fma(T1 a_arg, T2 b_arg, T3 c_arg) {
  T1 result{};
  for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
    result[paras_i] = fma(a_arg[paras_i], b_arg[paras_i], c_arg[paras_i]);
  }
  return result;
}

template <typename T,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T> && !detail::is_vector_like_v<T>>>
PARAS_KERNEL_HD inline T fma(T a_arg, T b_arg, T c_arg) {
  return fma(a_arg, b_arg, static_cast<T>(c_arg));
}

#define PARAS_SYCL_RELATIONAL_MATH(NAME, S, a, b, SCALAR_BODY)          \
  PARAS_KERNEL_HD inline bool NAME(float a_arg, float b_arg) {                          \
    using S = float; S a = a_arg; S b = b_arg;                         \
    return static_cast<bool>(SCALAR_BODY);                              \
  }                                                                      \
  PARAS_KERNEL_HD inline bool NAME(double a_arg, double b_arg) {                        \
    using S = double; S a = a_arg; S b = b_arg;                        \
    return static_cast<bool>(SCALAR_BODY);                              \
  }                                                                      \
  PARAS_KERNEL_HD inline bool NAME(half a_arg, half b_arg) {                            \
    using S = half; S a = a_arg; S b = b_arg;                          \
    return static_cast<bool>(SCALAR_BODY);                              \
  }                                                                      \
  template <typename T1, typename T2,                                   \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>> \
  PARAS_KERNEL_HD inline auto NAME(T1 a_arg, T2 b_arg) {                \
    if constexpr (is_vec_v<T1>) {                                       \
      using E = typename T1::value_type;                                \
      using R = typename vec_rel_t<E>::type;                            \
      vec<R, T1::size()> result{};                                      \
      for (int paras_i = 0; paras_i < T1::size(); ++paras_i) {          \
        bool paras_b = NAME(a_arg[paras_i], b_arg[paras_i]);            \
        result[paras_i] = paras_b ? static_cast<R>(-1) : static_cast<R>(0); \
      }                                                                 \
      return result;                                                    \
    } else if constexpr (is_marray_v<T1>) {                             \
      marray<bool, T1::size()> result{};                                \
      for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {  \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]);         \
      }                                                                 \
      return result;                                                    \
    } else {                                                            \
      return NAME(a_arg, static_cast<T1>(b_arg));                       \
    }                                                                   \
  }

PARAS_SYCL_RELATIONAL_MATH(isequal, S, a, b, (a == b))
PARAS_SYCL_RELATIONAL_MATH(isnotequal, S, a, b, (a != b))
PARAS_SYCL_RELATIONAL_MATH(isgreater, S, a, b, (a > b))
PARAS_SYCL_RELATIONAL_MATH(isgreaterequal, S, a, b, (a >= b))
PARAS_SYCL_RELATIONAL_MATH(isless, S, a, b, (a < b))
PARAS_SYCL_RELATIONAL_MATH(islessequal, S, a, b, (a <= b))
PARAS_SYCL_RELATIONAL_MATH(islessgreater, S, a, b, (a < b || a > b))

#undef PARAS_SYCL_RELATIONAL_MATH

PARAS_KERNEL_HD inline bool isnan(float x_arg) { return std::isnan(x_arg); }
PARAS_KERNEL_HD inline bool isnan(double x_arg) { return std::isnan(x_arg); }
PARAS_KERNEL_HD inline bool isnan(half x_arg) { return std::isnan(static_cast<double>(x_arg)); }

template <typename T,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline auto isnan(T x_arg) {
  if constexpr (is_vec_v<T>) {
    using E = typename T::value_type;
    using R = typename vec_rel_t<E>::type;
    vec<R, T::size()> result{};
    for (int i = 0; i < T::size(); ++i)
      result[i] = isnan(x_arg[i]) ? static_cast<R>(-1) : static_cast<R>(0);
    return result;
  } else if constexpr (is_marray_v<T>) {
    marray<bool, T::size()> result{};
    for (std::size_t i = 0; i < T::size(); ++i)
      result[i] = isnan(x_arg[i]);
    return result;
  } else {
    return isnan(static_cast<T>(x_arg));
  }
}

#define PARAS_SYCL_UNARY_RELATIONAL(NAME, EXPR)                          \
  PARAS_KERNEL_HD inline bool NAME(float x_arg) { double x_arg_d = x_arg; return (EXPR); } \
  PARAS_KERNEL_HD inline bool NAME(double x_arg_d) { return (EXPR); }                    \
  PARAS_KERNEL_HD inline bool NAME(half x_arg) { double x_arg_d = static_cast<double>(x_arg); return (EXPR); } \
  template <typename T,                                                  \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>> \
  PARAS_KERNEL_HD inline auto NAME(T x_arg) {                            \
    if constexpr (is_vec_v<T>) {                                         \
      using E = typename T::value_type;                                  \
      using R = typename vec_rel_t<E>::type;                             \
      vec<R, T::size()> result{};                                        \
      for (int i = 0; i < T::size(); ++i)                                \
        result[i] = NAME(x_arg[i]) ? static_cast<R>(-1) : static_cast<R>(0); \
      return result;                                                     \
    } else if constexpr (is_marray_v<T>) {                               \
      marray<bool, T::size()> result{};                                  \
      for (std::size_t i = 0; i < T::size(); ++i)                        \
        result[i] = NAME(x_arg[i]);                                      \
      return result;                                                     \
    } else {                                                             \
      return NAME(static_cast<T>(x_arg));                                \
    }                                                                    \
  }

PARAS_SYCL_UNARY_RELATIONAL(isfinite, std::isfinite(x_arg_d))
PARAS_SYCL_UNARY_RELATIONAL(isinf, std::isinf(x_arg_d))
PARAS_SYCL_UNARY_RELATIONAL(isnormal, std::isnormal(x_arg_d))
PARAS_SYCL_UNARY_RELATIONAL(signbit, std::signbit(x_arg_d))

#undef PARAS_SYCL_UNARY_RELATIONAL

PARAS_KERNEL_HD inline bool isordered(float a, float b) { return !(std::isnan(a) || std::isnan(b)); }
PARAS_KERNEL_HD inline bool isordered(double a, double b) { return !(std::isnan(a) || std::isnan(b)); }
PARAS_KERNEL_HD inline bool isordered(half a, half b) { return !(std::isnan(static_cast<double>(a)) || std::isnan(static_cast<double>(b))); }
PARAS_KERNEL_HD inline bool isunordered(float a, float b) { return std::isnan(a) || std::isnan(b); }
PARAS_KERNEL_HD inline bool isunordered(double a, double b) { return std::isnan(a) || std::isnan(b); }
PARAS_KERNEL_HD inline bool isunordered(half a, half b) { return std::isnan(static_cast<double>(a)) || std::isnan(static_cast<double>(b)); }

#define PARAS_SYCL_BINARY_RELATIONAL(NAME)                               \
  template <typename T1, typename T2,                                    \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>> \
  PARAS_KERNEL_HD inline auto NAME(T1 a_arg, T2 b_arg) {                 \
    if constexpr (is_vec_v<T1>) {                                        \
      using E = typename T1::value_type;                                 \
      using R = typename vec_rel_t<E>::type;                             \
      vec<R, T1::size()> result{};                                       \
      for (int paras_i = 0; paras_i < T1::size(); ++paras_i)             \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]) ? static_cast<R>(-1) : static_cast<R>(0); \
      return result;                                                     \
    } else if constexpr (is_marray_v<T1>) {                              \
      marray<bool, T1::size()> result{};                                 \
      for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i)     \
        result[paras_i] = NAME(a_arg[paras_i], b_arg[paras_i]);          \
      return result;                                                     \
    } else {                                                             \
      return NAME(a_arg, static_cast<T1>(b_arg));                        \
    }                                                                    \
  }

PARAS_SYCL_BINARY_RELATIONAL(isordered)
PARAS_SYCL_BINARY_RELATIONAL(isunordered)

#undef PARAS_SYCL_BINARY_RELATIONAL

template <typename T,
          typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<T>>>
PARAS_KERNEL_HD inline T abs(T x_arg) {
  if constexpr (detail::is_vector_like_v<T>) {
    T result{};
    for (std::size_t i = 0; i < T::size(); ++i) result[i] = abs(x_arg[i]);
    return result;
  } else {
    using U = std::make_unsigned_t<T>;
    return x_arg < 0 ? static_cast<T>(static_cast<U>(-(static_cast<U>(x_arg)))) : x_arg;
  }
}

template <typename T,
          typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<T>>>
PARAS_KERNEL_HD inline auto all(T x_arg) {
  if constexpr (is_vec_v<T>) {
    for (std::size_t i = 0; i < T::size(); ++i)
      if (!all(x_arg[i])) return 0;
    return 1;
  } else if constexpr (is_marray_v<T>) {
    for (std::size_t i = 0; i < T::size(); ++i)
      if (!all(x_arg[i])) return false;
    return true;
  } else {
    using U = std::make_unsigned_t<T>;
    U mask = static_cast<U>(U(1) << (sizeof(T) * 8 - 1));
    return (static_cast<U>(x_arg) & mask) != 0;
  }
}

template <typename T,
          typename = std::enable_if_t<detail::is_valid_geninteger_arg_v<T>>>
PARAS_KERNEL_HD inline auto any(T x_arg) {
  if constexpr (is_vec_v<T>) {
    for (std::size_t i = 0; i < T::size(); ++i)
      if (any(x_arg[i])) return 1;
    return 0;
  } else if constexpr (is_marray_v<T>) {
    for (std::size_t i = 0; i < T::size(); ++i)
      if (any(x_arg[i])) return true;
    return false;
  } else {
    using U = std::make_unsigned_t<T>;
    U mask = static_cast<U>(U(1) << (sizeof(T) * 8 - 1));
    return (static_cast<U>(x_arg) & mask) != 0;
  }
}

template <typename T1, typename T2>
PARAS_KERNEL_HD inline T1 max(T1 a_arg, T2 b_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto bv = [&]() {
        if constexpr (detail::is_vector_like_v<T2>) return b_arg[paras_i];
        else return b_arg;
      }();
      result[paras_i] = max(a_arg[paras_i], bv);
    }
    return result;
  } else {
    T1 a = a_arg;
    T1 b = static_cast<T1>(b_arg);
    return (a > b) ? a : b;
  }
}

template <typename T1, typename T2>
PARAS_KERNEL_HD inline T1 min(T1 a_arg, T2 b_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      auto bv = [&]() {
        if constexpr (detail::is_vector_like_v<T2>) return b_arg[paras_i];
        else return b_arg;
      }();
      result[paras_i] = min(a_arg[paras_i], bv);
    }
    return result;
  } else {
    T1 a = a_arg;
    T1 b = static_cast<T1>(b_arg);
    return (a < b) ? a : b;
  }
}

template <typename T1, typename T2, typename T3>
PARAS_KERNEL_HD inline T1 select(T1 a_arg, T2 b_arg, T3 c_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t i = 0; i < T1::size(); ++i) result[i] = select(a_arg[i], b_arg[i], c_arg[i]);
    return result;
  } else if constexpr (std::is_same_v<T3, bool>) {
    return c_arg ? b_arg : a_arg;
  } else {
    using U = std::make_unsigned_t<T3>;
    U mask = static_cast<U>(U(1) << (sizeof(T3) * 8 - 1));
    return (static_cast<U>(c_arg) & mask) ? b_arg : a_arg;
  }
}

#define PARAS_SYCL_GEO_CROSS(TYPE, N)                                    \
  PARAS_KERNEL_HD inline vec<TYPE, N> cross(vec<TYPE, N> a, vec<TYPE, N> b) {            \
    vec<TYPE, N> result{};                                               \
    result[0] = a[1] * b[2] - a[2] * b[1];                               \
    result[1] = a[2] * b[0] - a[0] * b[2];                               \
    result[2] = a[0] * b[1] - a[1] * b[0];                               \
    if constexpr (N == 4) result[3] = TYPE(0);                           \
    return result;                                                       \
  }                                                                       \
  PARAS_KERNEL_HD inline marray<TYPE, N> cross(marray<TYPE, N> a, marray<TYPE, N> b) {   \
    marray<TYPE, N> result{};                                            \
    result[0] = a[1] * b[2] - a[2] * b[1];                               \
    result[1] = a[2] * b[0] - a[0] * b[2];                               \
    result[2] = a[0] * b[1] - a[1] * b[0];                               \
    if constexpr (N == 4) result[3] = TYPE(0);                           \
    return result;                                                       \
  }

PARAS_SYCL_GEO_CROSS(float, 3)
PARAS_SYCL_GEO_CROSS(float, 4)
PARAS_SYCL_GEO_CROSS(double, 3)
PARAS_SYCL_GEO_CROSS(double, 4)

#undef PARAS_SYCL_GEO_CROSS

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_vector_like_v<T1>>>
PARAS_KERNEL_HD inline T1 cross(T1 a, T2 b) {
  T1 result{};
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
  if constexpr (T1::size() == 4) result[3] = typename T1::value_type(0);
  return result;
}

#define PARAS_SYCL_GEO_DOT(TYPE, N)                                      \
  PARAS_KERNEL_HD inline TYPE dot(vec<TYPE, N> a, vec<TYPE, N> b) {                      \
    TYPE result = TYPE(0);                                               \
    for (int paras_i = 0; paras_i < N; ++paras_i) result += a[paras_i] * b[paras_i]; \
    return result;                                                       \
  }                                                                       \
  PARAS_KERNEL_HD inline TYPE dot(marray<TYPE, N> a, marray<TYPE, N> b) {                \
    TYPE result = TYPE(0);                                               \
    for (std::size_t paras_i = 0; paras_i < N; ++paras_i) result += a[paras_i] * b[paras_i]; \
    return result;                                                       \
  }

PARAS_SYCL_GEO_DOT(float, 2)
PARAS_SYCL_GEO_DOT(float, 3)
PARAS_SYCL_GEO_DOT(float, 4)
PARAS_SYCL_GEO_DOT(double, 2)
PARAS_SYCL_GEO_DOT(double, 3)
PARAS_SYCL_GEO_DOT(double, 4)

#undef PARAS_SYCL_GEO_DOT

PARAS_KERNEL_HD inline float dot(float a, float b) { return a * b; }
PARAS_KERNEL_HD inline double dot(double a, double b) { return a * b; }
PARAS_KERNEL_HD inline half dot(half a, half b) { return static_cast<half>(static_cast<float>(a) * static_cast<float>(b)); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline auto dot(T1 a, T2 b) {
  if constexpr (detail::is_vector_like_v<T1>) {
    using E = typename T1::value_type;
    E result = E(0);
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) result += a[paras_i] * b[paras_i];
    return result;
  } else {
    return dot(a, static_cast<T1>(b));
  }
}

#define PARAS_SYCL_GEO_DISTANCE(TYPE, N)                                 \
  PARAS_KERNEL_HD inline TYPE distance(vec<TYPE, N> a, vec<TYPE, N> b) {                 \
    TYPE sum = TYPE(0);                                                  \
    for (int paras_i = 0; paras_i < N; ++paras_i) {                      \
      TYPE d = a[paras_i] - b[paras_i];                                  \
      sum += d * d;                                                      \
    }                                                                    \
    return static_cast<TYPE>(std::sqrt(static_cast<double>(sum)));       \
  }                                                                       \
  PARAS_KERNEL_HD inline TYPE distance(marray<TYPE, N> a, marray<TYPE, N> b) {           \
    TYPE sum = TYPE(0);                                                  \
    for (std::size_t paras_i = 0; paras_i < N; ++paras_i) {              \
      TYPE d = a[paras_i] - b[paras_i];                                  \
      sum += d * d;                                                      \
    }                                                                    \
    return static_cast<TYPE>(std::sqrt(static_cast<double>(sum)));       \
  }

PARAS_SYCL_GEO_DISTANCE(float, 2)
PARAS_SYCL_GEO_DISTANCE(float, 3)
PARAS_SYCL_GEO_DISTANCE(float, 4)
PARAS_SYCL_GEO_DISTANCE(double, 2)
PARAS_SYCL_GEO_DISTANCE(double, 3)
PARAS_SYCL_GEO_DISTANCE(double, 4)

#undef PARAS_SYCL_GEO_DISTANCE

PARAS_KERNEL_HD inline float distance(float a, float b) { return std::fabs(a - b); }
PARAS_KERNEL_HD inline double distance(double a, double b) { return std::fabs(a - b); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline auto distance(T1 a, T2 b) {
  if constexpr (detail::is_vector_like_v<T1>) {
    using E = typename T1::value_type;
    E sum = E(0);
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      E d = a[paras_i] - b[paras_i];
      sum += d * d;
    }
    return static_cast<E>(std::sqrt(static_cast<double>(sum)));
  } else {
    return distance(a, static_cast<T1>(b));
  }
}

#define PARAS_SYCL_GEO_LENGTH(TYPE, N)                                   \
  PARAS_KERNEL_HD inline TYPE length(vec<TYPE, N> v) {                                   \
    TYPE sum = TYPE(0);                                                  \
    for (int paras_i = 0; paras_i < N; ++paras_i) sum += v[paras_i] * v[paras_i]; \
    return static_cast<TYPE>(std::sqrt(static_cast<double>(sum)));       \
  }                                                                       \
  PARAS_KERNEL_HD inline TYPE length(marray<TYPE, N> v) {                                \
    TYPE sum = TYPE(0);                                                  \
    for (std::size_t paras_i = 0; paras_i < N; ++paras_i) sum += v[paras_i] * v[paras_i]; \
    return static_cast<TYPE>(std::sqrt(static_cast<double>(sum)));       \
  }

PARAS_SYCL_GEO_LENGTH(float, 2)
PARAS_SYCL_GEO_LENGTH(float, 3)
PARAS_SYCL_GEO_LENGTH(float, 4)
PARAS_SYCL_GEO_LENGTH(double, 2)
PARAS_SYCL_GEO_LENGTH(double, 3)
PARAS_SYCL_GEO_LENGTH(double, 4)

#undef PARAS_SYCL_GEO_LENGTH

PARAS_KERNEL_HD inline float length(float x) { return std::fabs(x); }
PARAS_KERNEL_HD inline double length(double x) { return std::fabs(x); }

template <typename T,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline auto length(T v) {
  if constexpr (detail::is_vector_like_v<T>) {
    using E = typename T::value_type;
    E sum = E(0);
    for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) sum += v[paras_i] * v[paras_i];
    return static_cast<E>(std::sqrt(static_cast<double>(sum)));
  } else {
    return length(static_cast<T>(v));
  }
}

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline auto fast_distance(T1 a, T2 b) {

  return distance(a, b);
}
PARAS_KERNEL_HD inline float fast_distance(float a, float b) { return distance(a, b); }
PARAS_KERNEL_HD inline double fast_distance(double a, double b) { return distance(a, b); }

template <typename T,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline auto fast_length(T v) {
  return length(v);
}
PARAS_KERNEL_HD inline float fast_length(float x) { return length(x); }
PARAS_KERNEL_HD inline double fast_length(double x) { return length(x); }

template <typename T,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline T fast_normalize(T v) {
  if constexpr (detail::is_vector_like_v<T>) {
    auto len = length(v);
    T result{};
    for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) result[paras_i] = v[paras_i] / len;
    return result;
  } else {
    return v < T(0) ? T(-1) : T(1);
  }
}

PARAS_KERNEL_HD inline float fast_normalize(float x) { return x < 0.0f ? -1.0f : 1.0f; }
PARAS_KERNEL_HD inline double fast_normalize(double x) { return x < 0.0 ? -1.0 : 1.0; }

template <typename T,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline T normalize(T v) {
  if constexpr (detail::is_vector_like_v<T>) {
    auto len = length(v);
    T result{};
    for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) result[paras_i] = v[paras_i] / len;
    return result;
  } else {
    return v < T(0) ? T(-1) : T(1);
  }
}
PARAS_KERNEL_HD inline float normalize(float x) { return x < 0.0f ? -1.0f : 1.0f; }
PARAS_KERNEL_HD inline double normalize(double x) { return x < 0.0 ? -1.0 : 1.0; }

PARAS_SYCL_BINARY_MATH_2T(rotate, S, x, shift, [&]() -> S {
  using U = std::make_unsigned_t<S>;
  constexpr unsigned bits = sizeof(S) * 8;
  U ux = static_cast<U>(x);
  unsigned s = static_cast<unsigned>(static_cast<U>(shift) % bits);
  if (s == 0) return static_cast<S>(ux);
  return static_cast<S>(static_cast<U>((ux << s) | (ux >> (bits - s))));
}())

PARAS_KERNEL_HD inline float mad(float a, float b, float c) { return a * b + c; }
PARAS_KERNEL_HD inline double mad(double a, double b, double c) { return a * b + c; }
PARAS_KERNEL_HD inline half mad(half a, half b, half c) {
  return static_cast<half>(static_cast<float>(a) * static_cast<float>(b) + static_cast<float>(c));
}

template <typename T1, typename T2, typename T3,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 mad(T1 a_arg, T2 b_arg, T3 c_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t i = 0; i < T1::size(); ++i) result[i] = mad(a_arg[i], b_arg[i], c_arg[i]);
    return result;
  } else {
    return mad(a_arg, static_cast<T1>(b_arg), static_cast<T1>(c_arg));
  }
}

template <typename T1, typename T2, typename T3,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1> &&
                                       detail::is_valid_math_arg_v<T2> &&
                                       detail::is_valid_math_arg_v<T3>>>
PARAS_KERNEL_HD inline T1 bitselect(T1 a_arg, T2 b_arg, T3 c_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) {
      result[paras_i] = bitselect(a_arg[paras_i], b_arg[paras_i], c_arg[paras_i]);
    }
    return result;
  } else {
    using S = T1;
    S a = a_arg;
    S b = static_cast<S>(b_arg);
    S c = static_cast<S>(c_arg);
    if constexpr (std::is_floating_point_v<S>) {
      using U = detail::same_size_uint_t<S>;
      U ua; U ub; U uc;
      std::memcpy(&ua, &a, sizeof(S));
      std::memcpy(&ub, &b, sizeof(S));
      std::memcpy(&uc, &c, sizeof(S));
      U ur = (ua & ~uc) | (ub & uc);
      S r;
      std::memcpy(&r, &ur, sizeof(S));
      return r;
    } else if constexpr (std::is_same_v<S, half>) {
      std::uint16_t ua, ub, uc;
      std::memcpy(&ua, &a, sizeof(std::uint16_t));
      std::memcpy(&ub, &b, sizeof(std::uint16_t));
      std::memcpy(&uc, &c, sizeof(std::uint16_t));
      std::uint16_t ur = (ua & ~uc) | (ub & uc);
      S result;
      std::memcpy(&result, &ur, sizeof(std::uint16_t));
      return result;
    } else {
      using U = std::make_unsigned_t<S>;
      U ua = static_cast<U>(a); U ub = static_cast<U>(b); U uc = static_cast<U>(c);
      return static_cast<S>((ua & ~uc) | (ub & uc));
    }
  }
}

#undef PARAS_SYCL_BINARY_MATH_2T
#undef PARAS_SYCL_TERNARY_MATH_3T

#define PARAS_SYCL_NATIVE_UNARY(NAME)                                    \
  PARAS_KERNEL_HD inline float NAME(float x) { return sycl::NAME(x); }                   \
  PARAS_KERNEL_HD inline double NAME(double x) { return sycl::NAME(x); }                 \
  PARAS_KERNEL_HD inline half NAME(half x) { return sycl::NAME(x); }                     \
  template <typename T,                                                  \
            typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>> \
  PARAS_KERNEL_HD inline T NAME(T x) {                                    \
    if constexpr (detail::is_vector_like_v<T>) {                          \
      T r{};                                                              \
      for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i)       \
        r[paras_i] = NAME(x[paras_i]);                                    \
      return r;                                                           \
    } else {                                                              \
      return sycl::NAME(x);                                               \
    }                                                                     \
  }

namespace native {

PARAS_KERNEL_HD inline double rsqrt(double x) { return sycl::rsqrt(x); }
PARAS_KERNEL_HD inline half rsqrt(half x) { return sycl::rsqrt(x); }
template <typename T,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline T rsqrt(T x) {
  if constexpr (detail::is_vector_like_v<T>) {
    T r{};
    for (std::size_t paras_i = 0; paras_i < T::size(); ++paras_i) r[paras_i] = rsqrt(x[paras_i]);
    return r;
  } else {
    return sycl::rsqrt(x);
  }
}

PARAS_SYCL_NATIVE_UNARY(sin)
PARAS_SYCL_NATIVE_UNARY(cos)
PARAS_SYCL_NATIVE_UNARY(tan)
PARAS_SYCL_NATIVE_UNARY(exp)
PARAS_SYCL_NATIVE_UNARY(exp2)
PARAS_SYCL_NATIVE_UNARY(exp10)
PARAS_SYCL_NATIVE_UNARY(log)
PARAS_SYCL_NATIVE_UNARY(log2)
PARAS_SYCL_NATIVE_UNARY(log10)
PARAS_SYCL_NATIVE_UNARY(sqrt)

PARAS_KERNEL_HD inline float recip(float x) { return 1.0f / x; }
PARAS_KERNEL_HD inline double recip(double x) { return 1.0 / x; }
PARAS_KERNEL_HD inline half recip(half x) { return static_cast<half>(1.0f / static_cast<float>(x)); }
template <typename T, typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline T recip(T x) {
  if constexpr (detail::is_vector_like_v<T>) {
    T r{}; for (std::size_t i = 0; i < T::size(); ++i) r[i] = recip(x[i]); return r;
  } else {
    return recip(static_cast<T>(x));
  }
}

PARAS_KERNEL_HD inline float powr(float x, float y) { return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(y))); }
PARAS_KERNEL_HD inline double powr(double x, double y) { return std::pow(x, y); }
PARAS_KERNEL_HD inline half powr(half x, half y) { return static_cast<half>(std::pow(static_cast<double>(x), static_cast<double>(y))); }
template <typename T1, typename T2, typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 powr(T1 x, T2 y) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 r{}; for (std::size_t i = 0; i < T1::size(); ++i) r[i] = powr(x[i], y[i]); return r;
  } else {
    return powr(x, static_cast<T1>(y));
  }
}

PARAS_KERNEL_HD inline float divide(float x, float y) { return x / y; }
PARAS_KERNEL_HD inline double divide(double x, double y) { return x / y; }
PARAS_KERNEL_HD inline half divide(half x, half y) { return static_cast<half>(static_cast<float>(x) / static_cast<float>(y)); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 divide(T1 x_arg, T2 y_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) result[paras_i] = x_arg[paras_i] / y_arg[paras_i];
    return result;
  } else {
    return divide(x_arg, static_cast<T1>(y_arg));
  }
}
}

namespace half_precision {
PARAS_SYCL_NATIVE_UNARY(sin)
PARAS_SYCL_NATIVE_UNARY(cos)
PARAS_SYCL_NATIVE_UNARY(tan)
PARAS_SYCL_NATIVE_UNARY(exp)
PARAS_SYCL_NATIVE_UNARY(exp2)
PARAS_SYCL_NATIVE_UNARY(exp10)
PARAS_SYCL_NATIVE_UNARY(log)
PARAS_SYCL_NATIVE_UNARY(log2)
PARAS_SYCL_NATIVE_UNARY(log10)
PARAS_SYCL_NATIVE_UNARY(sqrt)
PARAS_SYCL_NATIVE_UNARY(rsqrt)

PARAS_KERNEL_HD inline float recip(float x) { return 1.0f / x; }
PARAS_KERNEL_HD inline double recip(double x) { return 1.0 / x; }
PARAS_KERNEL_HD inline half recip(half x) { return static_cast<half>(1.0f / static_cast<float>(x)); }
template <typename T, typename = std::enable_if_t<detail::is_valid_math_arg_v<T>>>
PARAS_KERNEL_HD inline T recip(T x) {
  if constexpr (detail::is_vector_like_v<T>) {
    T r{}; for (std::size_t i = 0; i < T::size(); ++i) r[i] = recip(x[i]); return r;
  } else {
    return recip(static_cast<T>(x));
  }
}

PARAS_KERNEL_HD inline float powr(float x, float y) { return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(y))); }
PARAS_KERNEL_HD inline double powr(double x, double y) { return std::pow(x, y); }
PARAS_KERNEL_HD inline half powr(half x, half y) { return static_cast<half>(std::pow(static_cast<double>(x), static_cast<double>(y))); }
template <typename T1, typename T2, typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 powr(T1 x, T2 y) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 r{}; for (std::size_t i = 0; i < T1::size(); ++i) r[i] = powr(x[i], y[i]); return r;
  } else {
    return powr(x, static_cast<T1>(y));
  }
}

PARAS_KERNEL_HD inline float divide(float x, float y) { return x / y; }
PARAS_KERNEL_HD inline double divide(double x, double y) { return x / y; }
PARAS_KERNEL_HD inline half divide(half x, half y) { return static_cast<half>(static_cast<float>(x) / static_cast<float>(y)); }

template <typename T1, typename T2,
          typename = std::enable_if_t<detail::is_valid_math_arg_v<T1>>>
PARAS_KERNEL_HD inline T1 divide(T1 x_arg, T2 y_arg) {
  if constexpr (detail::is_vector_like_v<T1>) {
    T1 result{};
    for (std::size_t paras_i = 0; paras_i < T1::size(); ++paras_i) result[paras_i] = x_arg[paras_i] / y_arg[paras_i];
    return result;
  } else {
    return divide(x_arg, static_cast<T1>(y_arg));
  }
}
}

#undef PARAS_SYCL_NATIVE_UNARY

}

#endif
