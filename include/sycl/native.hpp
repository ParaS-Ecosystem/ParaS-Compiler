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

#ifndef __PARAS_NATIVE_HPP__
#define __PARAS_NATIVE_HPP__

#include "math.hpp"

namespace sycl {
namespace native {

PARAS_KERNEL_HD
inline float rsqrt(float x) {
    return 1.0f / sycl::sqrt(x);
}

} // namespace native

#define PARAS_NATIVE_UNARY(NAME, EXPR, ATTR)                                                       \
    ATTR PARAS_KERNEL_HD inline float NAME(float x) { return EXPR; }                               \
    template <typename NonScalar,                                                                  \
              std::enable_if_t<::sycl::detail::paras_bi::is_ns<NonScalar> &&                       \
                                   std::is_same_v<::sycl::detail::paras_bi::elem_t<NonScalar>,     \
                                                  float>,                                          \
                               int> = 0>                                                           \
    ATTR PARAS_KERNEL_HD inline ::sycl::detail::paras_bi::ret_t<NonScalar> NAME(NonScalar x) {     \
        ::sycl::detail::paras_bi::ret_t<NonScalar> r{};                                            \
        for (std::size_t i = 0; i < ::sycl::detail::paras_bi::ns<NonScalar>::n; ++i)               \
            ::sycl::detail::paras_bi::put(r, i, NAME(::sycl::detail::paras_bi::at(x, i)));         \
        return r;                                                                                  \
    }
#define PARAS_NATIVE_BINARY(NAME, EXPR, ATTR)                                                      \
    ATTR PARAS_KERNEL_HD inline float NAME(float x, float y) { return EXPR; }                      \
    template <typename NonScalar1, typename NonScalar2,                                            \
              std::enable_if_t<::sycl::detail::paras_bi::same_shape<NonScalar1, NonScalar2> &&     \
                                   std::is_same_v<::sycl::detail::paras_bi::elem_t<NonScalar1>,    \
                                                  float>,                                          \
                               int> = 0>                                                           \
    ATTR PARAS_KERNEL_HD inline ::sycl::detail::paras_bi::ret_t<NonScalar1> NAME(NonScalar1 x,     \
                                                                                NonScalar2 y) {    \
        ::sycl::detail::paras_bi::ret_t<NonScalar1> r{};                                           \
        for (std::size_t i = 0; i < ::sycl::detail::paras_bi::ns<NonScalar1>::n; ++i)              \
            ::sycl::detail::paras_bi::put(r, i, NAME(::sycl::detail::paras_bi::at(x, i),           \
                                                     ::sycl::detail::paras_bi::at(y, i)));         \
        return r;                                                                                  \
    }
#define PARAS_NATIVE_SET(ATTR, RSQRT_SCALAR)                                                       \
    PARAS_NATIVE_UNARY(cos, ::sycl::cos(x), ATTR)                                                  \
    PARAS_NATIVE_BINARY(divide, x / y, ATTR)                                                       \
    PARAS_NATIVE_UNARY(exp, ::sycl::exp(x), ATTR)                                                  \
    PARAS_NATIVE_UNARY(exp2, ::sycl::exp2(x), ATTR)                                                \
    PARAS_NATIVE_UNARY(exp10, ::sycl::exp10(x), ATTR)                                              \
    PARAS_NATIVE_UNARY(log, ::sycl::log(x), ATTR)                                                  \
    PARAS_NATIVE_UNARY(log2, ::sycl::log2(x), ATTR)                                                \
    PARAS_NATIVE_UNARY(log10, ::sycl::log10(x), ATTR)                                              \
    PARAS_NATIVE_BINARY(powr, ::sycl::powr(x, y), ATTR)                                            \
    PARAS_NATIVE_UNARY(recip, 1.0f / x, ATTR)                                                      \
    RSQRT_SCALAR                                                                                   \
    PARAS_NATIVE_UNARY(sin, ::sycl::sin(x), ATTR)                                                  \
    PARAS_NATIVE_UNARY(sqrt, ::sycl::sqrt(x), ATTR)                                                \
    PARAS_NATIVE_UNARY(tan, ::sycl::tan(x), ATTR)

#define PARAS_NATIVE_RSQRT_NS                                                                      \
    template <typename NonScalar,                                                                  \
              std::enable_if_t<::sycl::detail::paras_bi::is_ns<NonScalar> &&                       \
                                   std::is_same_v<::sycl::detail::paras_bi::elem_t<NonScalar>,     \
                                                  float>,                                          \
                               int> = 0>                                                           \
    PARAS_KERNEL_HD inline ::sycl::detail::paras_bi::ret_t<NonScalar> rsqrt(NonScalar x) {         \
        ::sycl::detail::paras_bi::ret_t<NonScalar> r{};                                            \
        for (std::size_t i = 0; i < ::sycl::detail::paras_bi::ns<NonScalar>::n; ++i)               \
            ::sycl::detail::paras_bi::put(r, i, rsqrt(::sycl::detail::paras_bi::at(x, i)));        \
        return r;                                                                                  \
    }
#define PARAS_NO_ATTR

namespace native {
PARAS_NATIVE_SET(PARAS_NO_ATTR, PARAS_NATIVE_RSQRT_NS)
} // namespace native

namespace half_precision {
#define PARAS_HP_DEPRECATED [[deprecated("half_precision functions are deprecated in SYCL 2020")]]
PARAS_NATIVE_SET(PARAS_HP_DEPRECATED, PARAS_NATIVE_UNARY(rsqrt, ::sycl::rsqrt(x), PARAS_HP_DEPRECATED))
#undef PARAS_HP_DEPRECATED
}

#undef PARAS_NATIVE_UNARY
#undef PARAS_NATIVE_BINARY
#undef PARAS_NATIVE_SET
#undef PARAS_NATIVE_RSQRT_NS
#undef PARAS_NO_ATTR
} // namespace sycl

#endif
