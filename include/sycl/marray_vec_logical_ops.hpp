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

#ifndef __PARAS_MARRAY_VEC_LOGIC_OPS_HPP__
#define __PARAS_MARRAY_VEC_LOGIC_OPS_HPP__

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
namespace sycl {

template <typename T>
struct paras_type_identity {
    using type = T;
};
template <typename T>
using paras_type_identity_t = typename paras_type_identity<T>::type;

template <typename T, typename>
struct paras_is_swizzle_ref : std::false_type {};
template <typename T>
struct paras_is_swizzle_ref<T, std::void_t<decltype(std::declval<T>().base)>> : std::true_type {};
template <typename T>
inline constexpr bool paras_is_swizzle_ref_v = paras_is_swizzle_ref<T>::value;

template <typename T>
struct vec_rel_t {
    using type = std::conditional_t<
        sizeof(T) == 1, std::int8_t,
        std::conditional_t<sizeof(T) == 2, std::int16_t,
                           std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>>>;
};

#define PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(OP)                                                \
    template <typename T, std::size_t N>                                                           \
    constexpr marray<T, N> operator OP(const marray<T, N>& lhs, const marray<T, N>& rhs) {         \
        marray<T, N> result{};                                                                     \
        for (std::size_t i = 0; i < N; ++i)                                                        \
            result[i] = lhs[i] OP rhs[i];                                                          \
        return result;                                                                             \
    }

#define PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(OP)                                                     \
    template <typename T, std::size_t N>                                                           \
    constexpr marray<T, N> operator OP(const marray<T, N>& lhs, const T& rhs) {                    \
        marray<T, N> result{};                                                                     \
        for (std::size_t i = 0; i < N; ++i)                                                        \
            result[i] = lhs[i] OP rhs;                                                             \
        return result;                                                                             \
    }

#define PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(OP)                                            \
    template <typename T, std::size_t N>                                                           \
    constexpr marray<T, N> operator OP(const T& lhs, const marray<T, N>& rhs) {                    \
        marray<T, N> result{};                                                                     \
        for (std::size_t i = 0; i < N; ++i)                                                        \
            result[i] = lhs OP rhs[i];                                                             \
        return result;                                                                             \
    }

#define PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_OP(OP)                                                   \
    template <typename T, int N>                                                                   \
    constexpr vec<T, N> operator OP(const vec<T, N>& lhs, const vec<T, N>& rhs) {                  \
        vec<T, N> result{};                                                                        \
        for (int i = 0; i < N; ++i)                                                                \
            result[i] = lhs[i] OP rhs[i];                                                          \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(+)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(*)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(&)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(|)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(^)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(-)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(/)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(%)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(<<)
PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP(>>)

PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(+)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(-)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(*)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(/)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(%)

PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(&)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(|)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(^)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(<<)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP(>>)

PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(+)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(-)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(*)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(/)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(%)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(&)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(|)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(^)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(<<)
PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP_REVERSED(>>)

PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_OP(+)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_OP(*)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_OP(&)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_OP(|)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_OP(^)

#undef PARAS_SYCL_DEFINE_MARRAY_ELEMENTWISE_OP
#undef PARAS_SYCL_DEFINE_MARRAY_SCALAR_OP
#undef PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_OP

template <typename T, std::size_t N>
constexpr marray<bool, N> operator&&(const marray<T, N>& lhs, const marray<T, N>& rhs) {
    marray<bool, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<bool>(lhs[i]) && static_cast<bool>(rhs[i]);
    }
    return result;
}

template <typename T, std::size_t N>
constexpr marray<bool, N> operator||(const marray<T, N>& lhs, const marray<T, N>& rhs) {
    marray<bool, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<bool>(lhs[i]) || static_cast<bool>(rhs[i]);
    }
    return result;
}

template <typename T, std::size_t N>
constexpr marray<bool, N> operator&&(const marray<T, N>& lhs, const T& rhs) {
    marray<bool, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<bool>(lhs[i]) && static_cast<bool>(rhs);
    }
    return result;
}

template <typename T, std::size_t N>
constexpr marray<bool, N> operator&&(const T& lhs, const marray<T, N>& rhs) {
    marray<bool, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<bool>(lhs) && static_cast<bool>(rhs[i]);
    }
    return result;
}

template <typename T, std::size_t N>
constexpr marray<bool, N> operator||(const marray<T, N>& lhs, const T& rhs) {
    marray<bool, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<bool>(lhs[i]) || static_cast<bool>(rhs);
    }
    return result;
}

template <typename T, std::size_t N>
constexpr marray<bool, N> operator||(const T& lhs, const marray<T, N>& rhs) {
    marray<bool, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<bool>(lhs) || static_cast<bool>(rhs[i]);
    }
    return result;
}

template <typename T, int N>
constexpr vec<typename vec_rel_t<T>::type, N> operator&&(const vec<T, N>& lhs,
                                                         const vec<T, N>& rhs) {
    using RelT = typename vec_rel_t<T>::type;
    vec<RelT, N> result{};
    for (int i = 0; i < N; ++i) {
        result[i] = (static_cast<bool>(lhs[i]) && static_cast<bool>(rhs[i])) ? RelT(-1) : RelT(0);
    }
    return result;
}

template <typename T, int N>
constexpr vec<typename vec_rel_t<T>::type, N> operator&&(const paras_type_identity_t<T>& lhs,
                                                         const vec<T, N>& rhs) {
    using RelT = typename vec_rel_t<T>::type;
    vec<RelT, N> result{};
    for (int i = 0; i < N; ++i) {
        result[i] = (static_cast<bool>(lhs) && static_cast<bool>(rhs[i])) ? RelT(-1) : RelT(0);
    }
    return result;
}

template <typename T, int N>
constexpr vec<typename vec_rel_t<T>::type, N> operator||(const vec<T, N>& lhs,
                                                         const vec<T, N>& rhs) {
    using RelT = typename vec_rel_t<T>::type;
    vec<RelT, N> result{};
    for (int i = 0; i < N; ++i) {
        result[i] = (static_cast<bool>(lhs[i]) || static_cast<bool>(rhs[i])) ? RelT(-1) : RelT(0);
    }
    return result;
}

template <typename T, int N>
constexpr vec<typename vec_rel_t<T>::type, N> operator&&(const vec<T, N>& lhs,
                                                         const paras_type_identity_t<T>& rhs) {
    using RelT = typename vec_rel_t<T>::type;
    vec<RelT, N> result{};
    for (int i = 0; i < N; ++i) {
        result[i] = (static_cast<bool>(lhs[i]) && static_cast<bool>(rhs)) ? RelT(-1) : RelT(0);
    }
    return result;
}

template <typename T, int N>
constexpr vec<typename vec_rel_t<T>::type, N> operator||(const vec<T, N>& lhs,
                                                         const paras_type_identity_t<T>& rhs) {
    using RelT = typename vec_rel_t<T>::type;
    vec<RelT, N> result{};
    for (int i = 0; i < N; ++i) {
        result[i] = (static_cast<bool>(lhs[i]) || static_cast<bool>(rhs)) ? RelT(-1) : RelT(0);
    }
    return result;
}

template <typename T, int N>
constexpr vec<typename vec_rel_t<T>::type, N> operator||(const paras_type_identity_t<T>& lhs,
                                                         const vec<T, N>& rhs) {
    using RelT = typename vec_rel_t<T>::type;
    vec<RelT, N> result{};
    for (int i = 0; i < N; ++i) {
        result[i] = (static_cast<bool>(lhs) || static_cast<bool>(rhs[i])) ? RelT(-1) : RelT(0);
    }
    return result;
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator&&(const typename SwizzleRefT::element_type& lhs,
                          const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs && static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator||(const typename SwizzleRefT::element_type& lhs,
                          const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs || static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator+(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs + static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator-(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs - static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator*(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs * static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator/(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs / static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator^(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs ^ static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator|(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs | static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator&(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs & static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator<<(const typename SwizzleRefT::element_type& lhs,
                          const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs << static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator>>(const typename SwizzleRefT::element_type& lhs,
                          const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs >> static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

template <typename SwizzleRefT, typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>
constexpr auto operator%(const typename SwizzleRefT::element_type& lhs,
                         const SwizzleRefT& rhs) noexcept {
    using T = typename SwizzleRefT::element_type;
    return lhs % static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);
}

#define PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP(OP)                               \
    template <typename SwizzleRefT,                                                                \
              typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>>>                    \
    constexpr auto operator OP(const typename SwizzleRefT::element_type& lhs,                      \
                               const SwizzleRefT& rhs) noexcept {                                  \
        using T = typename SwizzleRefT::element_type;                                              \
        return lhs OP static_cast<vec<T, static_cast<int>(SwizzleRefT::size())>>(rhs);             \
    }

PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP(==)
PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP(!=)
PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP(<)
PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP(>)
PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP(<=)
PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP(>=)

#undef PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_SCALAR_FIRST_OP

#define PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP(OP)                                                 \
    template <typename T, std::size_t N>                                                           \
    constexpr marray<bool, N> operator OP(const marray<T, N>& lhs, const marray<T, N>& rhs) {      \
        marray<bool, N> result{};                                                                  \
        for (std::size_t i = 0; i < N; ++i)                                                        \
            result[i] = lhs[i] OP rhs[i];                                                          \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP(==)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP(!=)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP(<)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP(>)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP(<=)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP(>=)

#undef PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_OP

#define PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP(OP)                                          \
    template <typename T, std::size_t N>                                                           \
    constexpr marray<bool, N> operator OP(const marray<T, N>& lhs, const T& rhs) {                 \
        marray<bool, N> result{};                                                                  \
        for (std::size_t i = 0; i < N; ++i)                                                        \
            result[i] = lhs[i] OP rhs;                                                             \
        return result;                                                                             \
    }                                                                                              \
    template <typename T, std::size_t N>                                                           \
    constexpr marray<bool, N> operator OP(const T& lhs, const marray<T, N>& rhs) {                 \
        marray<bool, N> result{};                                                                  \
        for (std::size_t i = 0; i < N; ++i)                                                        \
            result[i] = lhs OP rhs[i];                                                             \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP(<=)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP(>=)

PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP(==)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP(!=)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP(<)
PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP(>)

#undef PARAS_SYCL_DEFINE_MARRAY_RELATIONAL_SCALAR_OP

#define PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP(OP)                                                    \
    template <typename T, int N>                                                                   \
    constexpr vec<typename vec_rel_t<T>::type, N> operator OP(const vec<T, N>& lhs,                \
                                                              const vec<T, N>& rhs) {              \
        using RelT = typename vec_rel_t<T>::type;                                                  \
        vec<RelT, N> result{};                                                                     \
        for (int i = 0; i < N; ++i)                                                                \
            result[i] = (lhs[i] OP rhs[i]) ? RelT(-1) : RelT(0);                                   \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP(==)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP(!=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP(<)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP(>)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP(<=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP(>=)

#undef PARAS_SYCL_DEFINE_VEC_RELATIONAL_OP

#define PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_OP(OP)                                            \
    template <typename T, int N>                                                                   \
    constexpr vec<T, N> operator OP(const vec<T, N>& lhs, const paras_type_identity_t<T>& rhs) {   \
        vec<T, N> result{};                                                                        \
        for (int i = 0; i < N; ++i)                                                                \
            result[i] = lhs[i] OP rhs;                                                             \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_OP(+)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_OP(*)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_OP(&)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_OP(|)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_OP(^)

#undef PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_OP

#define PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_FIRST_COMMUTATIVE_OP(OP)                          \
    template <typename T, int N>                                                                   \
    constexpr vec<T, N> operator OP(const paras_type_identity_t<T>& lhs, const vec<T, N>& rhs) {   \
        vec<T, N> result{};                                                                        \
        for (int i = 0; i < N; ++i)                                                                \
            result[i] = lhs OP rhs[i];                                                             \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_FIRST_COMMUTATIVE_OP(+)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_FIRST_COMMUTATIVE_OP(*)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_FIRST_COMMUTATIVE_OP(^)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_FIRST_COMMUTATIVE_OP(|)
PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_FIRST_COMMUTATIVE_OP(&)
#undef PARAS_SYCL_DEFINE_VEC_ELEMENTWISE_SCALAR_FIRST_COMMUTATIVE_OP

template <typename T, int N>
constexpr vec<T, N> operator<<(const paras_type_identity_t<T>& lhs, const vec<T, N>& rhs) {
    vec<T, N> result{};
    for (int i = 0; i < N; ++i)
        result[i] = lhs << rhs[i];
    return result;
}

template <typename T, int N>
constexpr vec<T, N> operator>>(const paras_type_identity_t<T>& lhs, const vec<T, N>& rhs) {
    vec<T, N> result{};
    for (int i = 0; i < N; ++i)
        result[i] = lhs >> rhs[i];
    return result;
}

template <typename T, int N>
constexpr vec<T, N> operator%(const paras_type_identity_t<T>& lhs, const vec<T, N>& rhs) {
    vec<T, N> result{};
    for (int i = 0; i < N; ++i)
        result[i] = lhs % rhs[i];
    return result;
}

template <typename T, int N>
constexpr vec<T, N> operator-(const paras_type_identity_t<T>& lhs, const vec<T, N>& rhs) {
    vec<T, N> result{};
    for (int i = 0; i < N; ++i)
        result[i] = lhs - rhs[i];
    return result;
}

template <typename T, int N>
constexpr vec<T, N> operator/(const paras_type_identity_t<T>& lhs, const vec<T, N>& rhs) {
    vec<T, N> result{};
    for (int i = 0; i < N; ++i)
        result[i] = lhs / rhs[i];
    return result;
}

#define PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(OP)                                            \
    template <typename T, int N, typename SwizzleRefT,                                             \
              typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>&& static_cast<int>(  \
                                              SwizzleRefT::size()) == N>>                          \
    constexpr vec<T, N> operator OP(const vec<T, N>& lhs, const SwizzleRefT& rhs) noexcept {       \
        return lhs OP static_cast<vec<T, N>>(rhs);                                                 \
    }

PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(+)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(-)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(*)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(/)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(^)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(|)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(<<)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(>>)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(&)
PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP(%)

#undef PARAS_SYCL_DEFINE_VEC_ARITHMETIC_SWIZZLE_OP

#define PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(OP)                                            \
    template <typename T, int N, typename SwizzleRefT,                                             \
              typename = std::enable_if_t<paras_is_swizzle_ref_v<SwizzleRefT>&& static_cast<int>(  \
                                              SwizzleRefT::size()) == N>>                          \
    constexpr vec<typename vec_rel_t<T>::type, N> operator OP(const vec<T, N>& lhs,                \
                                                              const SwizzleRefT& rhs) noexcept {   \
        return lhs OP static_cast<vec<T, N>>(rhs);                                                 \
    }

PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(==)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(!=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(<)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(>)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(<=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(>=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(&&)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP(||)

#undef PARAS_SYCL_DEFINE_VEC_RELATIONAL_SWIZZLE_OP

#define PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP(OP)                                             \
    template <typename T, int N>                                                                   \
    constexpr vec<typename vec_rel_t<T>::type, N> operator OP(                                     \
        const vec<T, N>& lhs, const paras_type_identity_t<T>& rhs) {                               \
        using RelT = typename vec_rel_t<T>::type;                                                  \
        vec<RelT, N> result{};                                                                     \
        for (int i = 0; i < N; ++i)                                                                \
            result[i] = (lhs[i] OP rhs) ? RelT(-1) : RelT(0);                                      \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP(==)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP(!=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP(<)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP(>)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP(<=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP(>=)

#undef PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_OP

#define PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP(OP)                                       \
    template <typename T, int N>                                                                   \
    constexpr vec<typename vec_rel_t<T>::type, N> operator OP(const paras_type_identity_t<T>& lhs, \
                                                              const vec<T, N>& rhs) {              \
        using RelT = typename vec_rel_t<T>::type;                                                  \
        vec<RelT, N> result{};                                                                     \
        for (int i = 0; i < N; ++i)                                                                \
            result[i] = (lhs OP rhs[i]) ? RelT(-1) : RelT(0);                                      \
        return result;                                                                             \
    }

PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP(==)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP(!=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP(<)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP(>)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP(<=)
PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP(>=)

#undef PARAS_SYCL_DEFINE_VEC_RELATIONAL_SCALAR_FIRST_OP

} // namespace sycl

#endif
