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

#ifndef __PARAS_BUILTINS_HPP__
#define __PARAS_BUILTINS_HPP__

#include <cmath>
#include <cstdint>
#include <climits>
#include <limits>
#include <type_traits>
#include "access.hpp"
#include "data_types.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {
namespace detail {
namespace paras_bi {

template <class T>
using rm_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T, class = void>
struct is_swz : std::false_type {};
template <class T>
struct is_swz<T, std::void_t<decltype(std::declval<T&>().base), typename T::value_type,
                             decltype(T::size())>>
    : std::bool_constant<!is_vec_v<T> && !is_marray_v<T>> {};

template <class T, class = void>
struct ns {
    static constexpr bool value = false;
    static constexpr int kind = 0;
    static constexpr std::size_t n = 1;
    using elem = void;
    using ret = void;
    template <class U>
    using rebind = void;
};
template <class E, std::size_t N>
struct ns<marray<E, N>, void> {
    static constexpr bool value = true;
    static constexpr int kind = 1;
    static constexpr std::size_t n = N;
    using elem = E;
    using ret = marray<E, N>;
    template <class U>
    using rebind = marray<U, N>;
};
template <class E, int N>
struct ns<vec<E, N>, void> {
    static constexpr bool value = true;
    static constexpr int kind = 2;
    static constexpr std::size_t n = static_cast<std::size_t>(N);
    using elem = E;
    using ret = vec<E, N>;
    template <class U>
    using rebind = vec<U, N>;
};
template <class T>
struct ns<T, std::enable_if_t<is_swz<T>::value>> {
    static constexpr bool value = true;
    static constexpr int kind = 2;
    static constexpr std::size_t n = T::size();
    using elem = std::remove_cv_t<typename T::value_type>;
    using ret = vec<elem, static_cast<int>(T::size())>;
    template <class U>
    using rebind = vec<U, static_cast<int>(T::size())>;
};

template <class T>
inline constexpr bool is_ns = ns<T>::value;
template <class T>
using elem_t = typename ns<T>::elem;
template <class T>
using ret_t = typename ns<T>::ret;
template <class T, class U>
using rebind_t = typename ns<T>::template rebind<U>;

template <class T>
using el_t = std::conditional_t<is_ns<T>, elem_t<T>, T>;
template <class T, bool = is_ns<T>>
struct gret {
    using type = T;
};
template <class T>
struct gret<T, true> {
    using type = ret_t<T>;
};
template <class T>
using gret_t = typename gret<T>::type;

template <class E>
inline constexpr bool is_fp = std::is_same_v<E, float> || std::is_same_v<E, double> ||
                              std::is_same_v<E, half>;

template <class E>
inline constexpr bool is_int_scalar =
    std::is_same_v<E, char> || std::is_same_v<E, signed char> || std::is_same_v<E, short> ||
    std::is_same_v<E, int> || std::is_same_v<E, long> || std::is_same_v<E, long long> ||
    std::is_same_v<E, unsigned char> || std::is_same_v<E, unsigned short> ||
    std::is_same_v<E, unsigned int> || std::is_same_v<E, unsigned long> ||
    std::is_same_v<E, unsigned long long>;
template <class E>
inline constexpr bool is_vec_int =
    std::is_same_v<E, std::int8_t> || std::is_same_v<E, std::int16_t> ||
    std::is_same_v<E, std::int32_t> || std::is_same_v<E, std::int64_t> ||
    std::is_same_v<E, std::uint8_t> || std::is_same_v<E, std::uint16_t> ||
    std::is_same_v<E, std::uint32_t> || std::is_same_v<E, std::uint64_t>;

template <class T>
inline constexpr bool fp_ns = is_ns<T> && is_fp<elem_t<T>>;
template <class T>
inline constexpr bool gen_float = is_fp<T> || fp_ns<T>;
template <class T>
inline constexpr bool gen_int = is_int_scalar<T> ||
                                (ns<T>::kind == 1 && is_int_scalar<elem_t<T>>) ||
                                (ns<T>::kind == 2 && is_vec_int<elem_t<T>>);
template <class A, class B>
inline constexpr bool same_shape = is_ns<A> && is_ns<B> && ns<A>::kind == ns<B>::kind &&
                                   ns<A>::n == ns<B>::n &&
                                   std::is_same_v<elem_t<A>, elem_t<B>>;
template <class A, class B>
inline constexpr bool gen_match = (ns<A>::kind == 2) ? same_shape<A, B> : std::is_same_v<A, B>;
template <class A, class B>
inline constexpr bool fp_pair = fp_ns<A> && same_shape<A, B>;
template <class A, class B, class C>
inline constexpr bool fp_triple = fp_ns<A> && same_shape<A, B> && same_shape<A, C>;
template <class A, class B>
inline constexpr bool int_partner = fp_ns<A> && is_ns<B> && ns<A>::kind == ns<B>::kind &&
                                    ns<A>::n == ns<B>::n &&
                                    std::is_same_v<elem_t<B>, std::int32_t>;
template <class T>
inline constexpr bool geo = is_fp<T> || (fp_ns<T> && ns<T>::n >= 2 && ns<T>::n <= 4);
template <class T>
inline constexpr bool geo_float = geo<T> && std::is_same_v<el_t<T>, float>;

template <class T>
PARAS_KERNEL_HD inline auto at(const T& x, std::size_t i) {
    if constexpr (ns<T>::kind == 1)
        return static_cast<elem_t<T>>(x[i]);
    else if constexpr (ns<T>::kind == 2)
        return static_cast<elem_t<T>>(x[static_cast<int>(i)]);
    else
        return x;
}
template <class R>
PARAS_KERNEL_HD inline void put(R& r, std::size_t i, const elem_t<R>& v) {
    if constexpr (ns<R>::kind == 1)
        r[i] = v;
    else
        r[static_cast<int>(i)] = v;
}

template <class P, class = void>
struct ptr {
    static constexpr bool ok = false;
    using type = void;
};
template <class U>
struct ptr<U*, void> {
    static constexpr bool ok = !std::is_const_v<U> && !std::is_volatile_v<U>;
    using type = U;
};
template <class P>
struct ptr<P, std::void_t<typename P::value_type, decltype(P::address_space),
                          decltype(std::declval<P>().get_raw())>> {
    static constexpr bool ok = P::address_space != access::address_space::constant_space &&
                               !std::is_const_v<typename P::value_type>;
    using type = typename P::value_type;
};
template <class P, class T>
inline constexpr bool ptr_to = ptr<P>::ok && std::is_same_v<typename ptr<P>::type, T>;
template <class P, class T>
PARAS_KERNEL_HD inline void store(P p, const T& v) {
    if constexpr (std::is_pointer_v<P>)
        *p = v;
    else
        *p.get_raw() = v;
}

constexpr double pi = 3.141592653589793238462643383279502884;

namespace s {
template <class F>
PARAS_KERNEL_HD inline F acospi(F x) {
    return static_cast<F>(std::acos(static_cast<double>(x)) / pi);
}
PARAS_KERNEL_HD inline double acospi(double x) { return std::acos(x) / pi; }
template <class F>
PARAS_KERNEL_HD inline F asinpi(F x) {
    return static_cast<F>(std::asin(static_cast<double>(x)) / pi);
}
PARAS_KERNEL_HD inline double asinpi(double x) { return std::asin(x) / pi; }
template <class F>
PARAS_KERNEL_HD inline F atanpi(F x) {
    return static_cast<F>(std::atan(static_cast<double>(x)) / pi);
}
PARAS_KERNEL_HD inline double atanpi(double x) { return std::atan(x) / pi; }
template <class F>
PARAS_KERNEL_HD inline F atan2pi(F y, F x) {
    return static_cast<F>(std::atan2(static_cast<double>(y), static_cast<double>(x)) / pi);
}
PARAS_KERNEL_HD inline double atan2pi(double y, double x) { return std::atan2(y, x) / pi; }
PARAS_KERNEL_HD inline double reduce2(double x) { return x - 2.0 * std::nearbyint(x * 0.5); }
PARAS_KERNEL_HD inline double sinpi_d(double x) {
    if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    if (x == std::trunc(x)) return std::copysign(0.0, x);
    double r = reduce2(x);
    if (r > 0.5) r = 1.0 - r;
    else if (r < -0.5) r = -1.0 - r;
    return std::sin(pi * r);
}
PARAS_KERNEL_HD inline double cospi_d(double x) {
    if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    double r = std::fabs(reduce2(x));
    if (r == 0.5) return 0.0;
    return sinpi_d(0.5 - r);
}
PARAS_KERNEL_HD inline double tanpi_d(double x) {
    if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    double r = x - std::nearbyint(x);
    if (r == 0.0) return std::copysign(0.0, x);
    if (std::fabs(r) == 0.5) return std::copysign(std::numeric_limits<double>::infinity(), r);
    return sinpi_d(r) / cospi_d(r);
}
template <class F>
PARAS_KERNEL_HD inline F sinpi(F x) { return static_cast<F>(sinpi_d(x)); }
template <class F>
PARAS_KERNEL_HD inline F cospi(F x) { return static_cast<F>(cospi_d(x)); }
template <class F>
PARAS_KERNEL_HD inline F tanpi(F x) { return static_cast<F>(tanpi_d(x)); }
template <class F>
PARAS_KERNEL_HD inline F exp10(F x) { return static_cast<F>(std::pow(10.0, static_cast<double>(x))); }
PARAS_KERNEL_HD inline double exp10(double x) { return std::pow(10.0, x); }
template <class F>
PARAS_KERNEL_HD inline F rsqrt(F x) { return static_cast<F>(1.0 / std::sqrt(static_cast<double>(x))); }
PARAS_KERNEL_HD inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
template <class F>
PARAS_KERNEL_HD inline F powr(F x, F y) {
    if (x < F(0)) return std::numeric_limits<F>::quiet_NaN();
    return std::pow(x, y);
}
template <class F>
PARAS_KERNEL_HD inline F pown(F x, int y) { return std::pow(x, static_cast<F>(y)); }
template <class F>
PARAS_KERNEL_HD inline F rootn(F x, int y) {
    if (y == 0) return std::numeric_limits<F>::quiet_NaN();
    if (x < F(0)) {
        if (y % 2 == 0) return std::numeric_limits<F>::quiet_NaN();
        return -static_cast<F>(std::pow(-static_cast<double>(x), 1.0 / y));
    }
    return static_cast<F>(std::pow(static_cast<double>(x), 1.0 / y));
}
PARAS_KERNEL_HD inline double rootn(double x, int y) {
    if (y == 0) return std::numeric_limits<double>::quiet_NaN();
    if (x < 0.0) {
        if (y % 2 == 0) return std::numeric_limits<double>::quiet_NaN();
        return -std::pow(-x, 1.0 / y);
    }
    return std::pow(x, 1.0 / y);
}
template <class F>
PARAS_KERNEL_HD inline F maxmag(F x, F y) {
    F ax = std::fabs(x), ay = std::fabs(y);
    if (ax > ay) return x;
    if (ay > ax) return y;
    return std::fmax(x, y);
}
template <class F>
PARAS_KERNEL_HD inline F minmag(F x, F y) {
    F ax = std::fabs(x), ay = std::fabs(y);
    if (ax < ay) return x;
    if (ay < ax) return y;
    return std::fmin(x, y);
}
template <class F>
PARAS_KERNEL_HD inline F fract(F x, F& ip) {
    ip = std::floor(x);
    if (std::isinf(x)) return std::copysign(F(0), x);
    return std::fmin(x - ip, std::nextafter(F(1), F(0)));
}
template <class F>
PARAS_KERNEL_HD inline int gamma_sign(F x) {
    if (!(x < F(0))) return 1;
    F f = std::floor(x);
    if (f == x) return 1;
    long long k = static_cast<long long>(f);
    return (k % 2 == 0) ? 1 : -1;
}
template <class F>
PARAS_KERNEL_HD inline F mad(F a, F b, F c) { return a * b + c; }
template <class F>
PARAS_KERNEL_HD inline F fdim(F x, F y) { return std::fdim(x, y); }
}

PARAS_KERNEL_HD inline std::uint16_t hbits(half h) { return __builtin_bit_cast(std::uint16_t, h); }
PARAS_KERNEL_HD inline half hfrom(std::uint16_t b) { return __builtin_bit_cast(half, b); }

template <class T>
using uns_t = std::make_unsigned_t<T>;
template <class T>
inline constexpr int bits = static_cast<int>(sizeof(T) * CHAR_BIT);

struct op_abs {
    template <class T>
    PARAS_KERNEL_HD static T f(T x) {
        if constexpr (std::is_signed_v<T>) return x < 0 ? static_cast<T>(-x) : x;
        else return x;
    }
};
struct op_abs_diff {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        using U = uns_t<T>;
        return static_cast<T>(x > y ? static_cast<U>(static_cast<U>(x) - static_cast<U>(y))
                                    : static_cast<U>(static_cast<U>(y) - static_cast<U>(x)));
    }
};
struct op_add_sat {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        T r;
        if (!__builtin_add_overflow(x, y, &r)) return r;
        if constexpr (std::is_signed_v<T>)
            return y > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        else
            return std::numeric_limits<T>::max();
    }
};
struct op_sub_sat {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        T r;
        if (!__builtin_sub_overflow(x, y, &r)) return r;
        if constexpr (std::is_signed_v<T>)
            return y < 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        else
            return std::numeric_limits<T>::min();
    }
};
struct op_hadd {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        return static_cast<T>((x >> 1) + (y >> 1) + (x & y & 1));
    }
};
struct op_rhadd {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        return static_cast<T>((x >> 1) + (y >> 1) + ((x | y) & 1));
    }
};
struct op_clz {
    template <class T>
    PARAS_KERNEL_HD static T f(T x) {
        using U = uns_t<T>;
        U u = static_cast<U>(x);
        if (u == 0) return static_cast<T>(bits<T>);
        return static_cast<T>(__builtin_clzll(static_cast<unsigned long long>(u)) -
                              (64 - bits<T>));
    }
};
struct op_ctz {
    template <class T>
    PARAS_KERNEL_HD static T f(T x) {
        using U = uns_t<T>;
        U u = static_cast<U>(x);
        if (u == 0) return static_cast<T>(bits<T>);
        return static_cast<T>(__builtin_ctzll(static_cast<unsigned long long>(u)));
    }
};
struct op_popcount {
    template <class T>
    PARAS_KERNEL_HD static T f(T x) {
        return static_cast<T>(
            __builtin_popcountll(static_cast<unsigned long long>(static_cast<uns_t<T>>(x))));
    }
};
struct op_mul_hi {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        if constexpr (sizeof(T) < 8) {
            using W = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;
            return static_cast<T>((static_cast<W>(x) * static_cast<W>(y)) >> bits<T>);
        } else {
            using W = std::conditional_t<std::is_signed_v<T>, __int128, unsigned __int128>;
            return static_cast<T>((static_cast<W>(x) * static_cast<W>(y)) >> 64);
        }
    }
};
struct op_mad_hi {
    template <class T>
    PARAS_KERNEL_HD static T f(T a, T b, T c) {
        return static_cast<T>(static_cast<uns_t<T>>(op_mul_hi::f(a, b)) + static_cast<uns_t<T>>(c));
    }
};
struct op_mad_sat {
    template <class T>
    PARAS_KERNEL_HD static T f(T a, T b, T c) {
        using W = std::conditional_t<std::is_signed_v<T>, __int128, unsigned __int128>;
        W r = static_cast<W>(a) * static_cast<W>(b) + static_cast<W>(c);
        if constexpr (sizeof(T) == 8 && !std::is_signed_v<T>) {
            unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
            if (p > std::numeric_limits<T>::max() ||
                p + c > static_cast<unsigned __int128>(std::numeric_limits<T>::max()))
                return std::numeric_limits<T>::max();
            return static_cast<T>(p + c);
        }
        if (r > static_cast<W>(std::numeric_limits<T>::max())) return std::numeric_limits<T>::max();
        if (r < static_cast<W>(std::numeric_limits<T>::min())) return std::numeric_limits<T>::min();
        return static_cast<T>(r);
    }
};
struct op_rotate {
    template <class T>
    PARAS_KERNEL_HD static T f(T v, T count) {
        using U = uns_t<T>;
        const U u = static_cast<U>(v);
        const unsigned c = static_cast<unsigned>(static_cast<U>(count) % bits<T>);
        if (c == 0) return v;
        return static_cast<T>(static_cast<U>((u << c) | (u >> (bits<T> - c))));
    }
};
struct op_mul24 {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        return static_cast<T>(static_cast<std::uint32_t>(x) * static_cast<std::uint32_t>(y));
    }
};
struct op_mad24 {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y, T z) {
        return static_cast<T>(static_cast<std::uint32_t>(x) * static_cast<std::uint32_t>(y) +
                              static_cast<std::uint32_t>(z));
    }
};
struct op_max {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        if constexpr (is_fp<T>) return T(std::fmax(static_cast<std::conditional_t<std::is_same_v<T, half>, float, T>>(x),
                                                   static_cast<std::conditional_t<std::is_same_v<T, half>, float, T>>(y)));
        else return x < y ? y : x;
    }
};
struct op_min {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y) {
        if constexpr (is_fp<T>) return T(std::fmin(static_cast<std::conditional_t<std::is_same_v<T, half>, float, T>>(x),
                                                   static_cast<std::conditional_t<std::is_same_v<T, half>, float, T>>(y)));
        else return y < x ? y : x;
    }
};
struct op_clamp {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T lo, T hi) { return op_min::f(op_max::f(x, lo), hi); }
};

template <class Op, class A>
PARAS_KERNEL_HD inline gret_t<A> map1(const A& a) {
    if constexpr (is_ns<A>) {
        gret_t<A> r{};
        for (std::size_t i = 0; i < ns<A>::n; ++i) put(r, i, Op::f(at(a, i)));
        return r;
    } else {
        return Op::f(a);
    }
}
template <class Op, class A, class B>
PARAS_KERNEL_HD inline gret_t<A> map2(const A& a, const B& b) {
    if constexpr (is_ns<A>) {
        gret_t<A> r{};
        for (std::size_t i = 0; i < ns<A>::n; ++i) put(r, i, Op::f(at(a, i), at(b, i)));
        return r;
    } else {
        return Op::f(a, b);
    }
}
template <class Op, class A, class B, class C>
PARAS_KERNEL_HD inline gret_t<A> map3(const A& a, const B& b, const C& c) {
    if constexpr (is_ns<A>) {
        gret_t<A> r{};
        for (std::size_t i = 0; i < ns<A>::n; ++i)
            put(r, i, Op::f(at(a, i), at(b, i), at(c, i)));
        return r;
    } else {
        return Op::f(a, b, c);
    }
}

template <class E>
using rel_int_t = std::conditional_t<
    sizeof(E) == 1, std::int8_t,
    std::conditional_t<sizeof(E) == 2, std::int16_t,
                       std::conditional_t<sizeof(E) == 4, std::int32_t, std::int64_t>>>;
template <class T>
using rel_ret_t = std::conditional_t<ns<T>::kind == 1, rebind_t<T, bool>,
                                     rebind_t<T, rel_int_t<elem_t<T>>>>;
template <class T>
PARAS_KERNEL_HD inline elem_t<rel_ret_t<T>> rel_val(bool b) {
    if constexpr (ns<T>::kind == 1) return b;
    else return b ? elem_t<rel_ret_t<T>>(-1) : elem_t<rel_ret_t<T>>(0);
}

template <class T>
PARAS_KERNEL_HD inline bool msb(T x) {
    return (static_cast<uns_t<T>>(x) >> (bits<T> - 1)) & 1u;
}

template <class T>
using bits_t = std::conditional_t<
    sizeof(T) == 1, std::uint8_t,
    std::conditional_t<sizeof(T) == 2, std::uint16_t,
                       std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>>;

template <class E>
inline constexpr bool gen_scalar = is_int_scalar<E> || is_fp<E>;
template <class E>
inline constexpr bool vec_elem = is_vec_int<E> || is_fp<E>;
template <class T>
inline constexpr bool gen_type = gen_scalar<T> || (ns<T>::kind == 1 && gen_scalar<elem_t<T>>) ||
                                 (ns<T>::kind == 2 && vec_elem<elem_t<T>>);

}
}

#define PARAS_BI_NS ::sycl::detail::paras_bi

#define PARAS_BI_UNARY(NAME, FIMPL, DIMPL)                                                         \
    PARAS_KERNEL_HD inline float NAME(float x) { return FIMPL(x); }                               \
    PARAS_KERNEL_HD inline double NAME(double x) { return DIMPL(x); }                             \
    PARAS_KERNEL_HD inline half NAME(half x) { return half(FIMPL(static_cast<float>(x))); }        \
    template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar>, int> = 0>        \
    PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> NAME(NonScalar x) {                       \
        PARAS_BI_NS::ret_t<NonScalar> r{};                                                         \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)                            \
            PARAS_BI_NS::put(r, i, NAME(PARAS_BI_NS::at(x, i)));                                   \
        return r;                                                                                  \
    }

#define PARAS_BI_BINARY(NAME, FIMPL, DIMPL)                                                        \
    PARAS_KERNEL_HD inline float NAME(float x, float y) { return FIMPL(x, y); }                   \
    PARAS_KERNEL_HD inline double NAME(double x, double y) { return DIMPL(x, y); }                \
    PARAS_KERNEL_HD inline half NAME(half x, half y) {                                             \
        return half(FIMPL(static_cast<float>(x), static_cast<float>(y)));                          \
    }                                                                                              \
    template <typename NonScalar1, typename NonScalar2,                                            \
              std::enable_if_t<PARAS_BI_NS::fp_pair<NonScalar1, NonScalar2>, int> = 0>             \
    PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar1> NAME(NonScalar1 x, NonScalar2 y) {       \
        PARAS_BI_NS::ret_t<NonScalar1> r{};                                                        \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i)                           \
            PARAS_BI_NS::put(r, i, NAME(PARAS_BI_NS::at(x, i), PARAS_BI_NS::at(y, i)));            \
        return r;                                                                                  \
    }

#define PARAS_BI_TERNARY(NAME, FIMPL, DIMPL)                                                       \
    PARAS_KERNEL_HD inline float NAME(float a, float b, float c) { return FIMPL(a, b, c); }       \
    PARAS_KERNEL_HD inline double NAME(double a, double b, double c) { return DIMPL(a, b, c); }   \
    PARAS_KERNEL_HD inline half NAME(half a, half b, half c) {                                     \
        return half(FIMPL(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)));   \
    }                                                                                              \
    template <typename NonScalar1, typename NonScalar2, typename NonScalar3,                       \
              std::enable_if_t<PARAS_BI_NS::fp_triple<NonScalar1, NonScalar2, NonScalar3>, int> =  \
                  0>                                                                               \
    PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar1> NAME(NonScalar1 a, NonScalar2 b,         \
                                                               NonScalar3 c) {                     \
        PARAS_BI_NS::ret_t<NonScalar1> r{};                                                        \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i)                           \
            PARAS_BI_NS::put(r, i,                                                                 \
                             NAME(PARAS_BI_NS::at(a, i), PARAS_BI_NS::at(b, i),                    \
                                  PARAS_BI_NS::at(c, i)));                                         \
        return r;                                                                                  \
    }

#define PARAS_BI_SCALAR2(NAME)                                                                     \
    template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar>, int> = 0>        \
    PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> NAME(NonScalar x,                         \
                                                              typename NonScalar::value_type y) {  \
        PARAS_BI_NS::ret_t<NonScalar> r{};                                                         \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)                            \
            PARAS_BI_NS::put(r, i, NAME(PARAS_BI_NS::at(x, i), y));                                \
        return r;                                                                                  \
    }

#define PARAS_BI_S(F) PARAS_BI_NS::s::F

PARAS_BI_UNARY(acos, std::acos, std::acos)
PARAS_BI_UNARY(acosh, std::acosh, std::acosh)
PARAS_BI_UNARY(acospi, PARAS_BI_S(acospi), PARAS_BI_S(acospi))
PARAS_BI_UNARY(asin, std::asin, std::asin)
PARAS_BI_UNARY(asinh, std::asinh, std::asinh)
PARAS_BI_UNARY(asinpi, PARAS_BI_S(asinpi), PARAS_BI_S(asinpi))
PARAS_BI_UNARY(atan, std::atan, std::atan)
PARAS_BI_BINARY(atan2, std::atan2, std::atan2)
PARAS_BI_UNARY(atanh, std::atanh, std::atanh)
PARAS_BI_UNARY(atanpi, PARAS_BI_S(atanpi), PARAS_BI_S(atanpi))
PARAS_BI_BINARY(atan2pi, PARAS_BI_S(atan2pi), PARAS_BI_S(atan2pi))
PARAS_BI_UNARY(cbrt, std::cbrt, std::cbrt)
PARAS_BI_UNARY(ceil, std::ceil, std::ceil)
PARAS_BI_BINARY(copysign, std::copysign, std::copysign)
PARAS_BI_UNARY(cos, std::cos, std::cos)
PARAS_BI_UNARY(cosh, std::cosh, std::cosh)
PARAS_BI_UNARY(cospi, PARAS_BI_S(cospi), PARAS_BI_S(cospi_d))
PARAS_BI_UNARY(erfc, std::erfc, std::erfc)
PARAS_BI_UNARY(erf, std::erf, std::erf)
PARAS_BI_UNARY(exp, std::exp, std::exp)
PARAS_BI_UNARY(exp2, std::exp2, std::exp2)
PARAS_BI_UNARY(exp10, PARAS_BI_S(exp10), PARAS_BI_S(exp10))
PARAS_BI_UNARY(expm1, std::expm1, std::expm1)
PARAS_BI_UNARY(fabs, std::fabs, std::fabs)
PARAS_BI_BINARY(fdim, PARAS_BI_S(fdim), PARAS_BI_S(fdim))
PARAS_BI_UNARY(floor, std::floor, std::floor)
PARAS_BI_TERNARY(fma, std::fma, std::fma)
PARAS_BI_BINARY(fmax, std::fmax, std::fmax)
PARAS_BI_SCALAR2(fmax)
PARAS_BI_BINARY(fmin, std::fmin, std::fmin)
PARAS_BI_SCALAR2(fmin)
PARAS_BI_BINARY(fmod, std::fmod, std::fmod)
PARAS_BI_BINARY(hypot, std::hypot, std::hypot)
PARAS_BI_UNARY(lgamma, std::lgamma, std::lgamma)
PARAS_BI_UNARY(log, std::log, std::log)
PARAS_BI_UNARY(log2, std::log2, std::log2)
PARAS_BI_UNARY(log10, std::log10, std::log10)
PARAS_BI_UNARY(log1p, std::log1p, std::log1p)
PARAS_BI_UNARY(logb, std::logb, std::logb)
PARAS_BI_TERNARY(mad, PARAS_BI_S(mad), PARAS_BI_S(mad))
PARAS_BI_BINARY(maxmag, PARAS_BI_S(maxmag), PARAS_BI_S(maxmag))
PARAS_BI_BINARY(minmag, PARAS_BI_S(minmag), PARAS_BI_S(minmag))
namespace detail {
namespace paras_bi {
PARAS_KERNEL_HD inline half half_nextafter(half x, half y) {
    const float fx = static_cast<float>(x), fy = static_cast<float>(y);
    if (std::isnan(fx) || std::isnan(fy)) return half(fx + fy);
    if (fx == fy) return y;
    std::uint16_t b = hbits(x);
    if ((b & 0x7fffu) == 0)
        return hfrom(static_cast<std::uint16_t>((fy < 0.0f ? 0x8000u : 0u) | 1u));
    const bool away = (fy > fx) == ((b & 0x8000u) == 0);
    b = static_cast<std::uint16_t>(away ? b + 1u : b - 1u);
    return hfrom(b);
}
}
}
PARAS_KERNEL_HD inline float nextafter(float x, float y) { return std::nextafter(x, y); }
PARAS_KERNEL_HD inline double nextafter(double x, double y) { return std::nextafter(x, y); }
PARAS_KERNEL_HD inline half nextafter(half x, half y) { return PARAS_BI_NS::half_nextafter(x, y); }
template <typename NonScalar1, typename NonScalar2,
          std::enable_if_t<PARAS_BI_NS::fp_pair<NonScalar1, NonScalar2>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar1> nextafter(NonScalar1 x, NonScalar2 y) {
    PARAS_BI_NS::ret_t<NonScalar1> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i)
        PARAS_BI_NS::put(r, i, nextafter(PARAS_BI_NS::at(x, i), PARAS_BI_NS::at(y, i)));
    return r;
}
PARAS_BI_BINARY(pow, std::pow, std::pow)
PARAS_BI_BINARY(powr, PARAS_BI_S(powr), PARAS_BI_S(powr))
PARAS_BI_BINARY(remainder, std::remainder, std::remainder)
PARAS_BI_UNARY(rint, std::rint, std::rint)
PARAS_BI_UNARY(round, std::round, std::round)
#if PARAS_GPU_BACKEND
PARAS_BI_UNARY(rsqrt, ::rsqrtf, ::rsqrt)
#else
PARAS_BI_UNARY(rsqrt, PARAS_BI_S(rsqrt), PARAS_BI_S(rsqrt))
#endif
PARAS_BI_UNARY(sin, std::sin, std::sin)
PARAS_BI_UNARY(sinh, std::sinh, std::sinh)
PARAS_BI_UNARY(sinpi, PARAS_BI_S(sinpi), PARAS_BI_S(sinpi_d))
PARAS_BI_UNARY(sqrt, std::sqrt, std::sqrt)
PARAS_BI_UNARY(tan, std::tan, std::tan)
PARAS_BI_UNARY(tanh, std::tanh, std::tanh)
PARAS_BI_UNARY(tanpi, PARAS_BI_S(tanpi), PARAS_BI_S(tanpi_d))
PARAS_BI_UNARY(tgamma, std::tgamma, std::tgamma)
PARAS_BI_UNARY(trunc, std::trunc, std::trunc)

PARAS_KERNEL_HD inline int ilogb(float x) { return std::ilogb(x); }
PARAS_KERNEL_HD inline int ilogb(double x) { return std::ilogb(x); }
PARAS_KERNEL_HD inline int ilogb(half x) { return std::ilogb(static_cast<float>(x)); }
template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::rebind_t<NonScalar, int> ilogb(NonScalar x) {
    PARAS_BI_NS::rebind_t<NonScalar, int> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)
        PARAS_BI_NS::put(r, i, ilogb(PARAS_BI_NS::at(x, i)));
    return r;
}

#define PARAS_BI_INT2(NAME, FIMPL, DIMPL)                                                          \
    PARAS_KERNEL_HD inline float NAME(float x, int k) { return FIMPL(x, k); }                     \
    PARAS_KERNEL_HD inline double NAME(double x, int k) { return DIMPL(x, k); }                   \
    PARAS_KERNEL_HD inline half NAME(half x, int k) {                                              \
        return half(FIMPL(static_cast<float>(x), k));                                              \
    }                                                                                              \
    template <typename NonScalar1, typename NonScalar2,                                            \
              std::enable_if_t<PARAS_BI_NS::int_partner<NonScalar1, NonScalar2>, int> = 0>         \
    PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar1> NAME(NonScalar1 x, NonScalar2 k) {       \
        PARAS_BI_NS::ret_t<NonScalar1> r{};                                                        \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i)                           \
            PARAS_BI_NS::put(r, i, NAME(PARAS_BI_NS::at(x, i), PARAS_BI_NS::at(k, i)));            \
        return r;                                                                                  \
    }
PARAS_BI_INT2(ldexp, std::ldexp, std::ldexp)
PARAS_BI_INT2(pown, PARAS_BI_S(pown), PARAS_BI_S(pown))
PARAS_BI_INT2(rootn, PARAS_BI_S(rootn), PARAS_BI_S(rootn))
template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> ldexp(NonScalar x, int k) {
    PARAS_BI_NS::ret_t<NonScalar> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)
        PARAS_BI_NS::put(r, i, ldexp(PARAS_BI_NS::at(x, i), k));
    return r;
}

PARAS_KERNEL_HD inline float nan(std::uint32_t nancode) {
    return __builtin_bit_cast(float, 0x7fc00000u | (nancode & 0x003fffffu));
}
PARAS_KERNEL_HD inline double nan(std::uint64_t nancode) {
    return __builtin_bit_cast(double,
                              0x7ff8000000000000ull | (nancode & 0x0007ffffffffffffull));
}
PARAS_KERNEL_HD inline half nan(std::uint16_t nancode) {
    return PARAS_BI_NS::hfrom(static_cast<std::uint16_t>(0x7e00u | (nancode & 0x01ffu)));
}
namespace detail {
namespace paras_bi {
template <class U>
using nan_t = std::conditional_t<std::is_same_v<U, std::uint32_t>, float,
                                 std::conditional_t<std::is_same_v<U, std::uint64_t>, double,
                                                    half>>;
template <class T>
inline constexpr bool nan_ns =
    is_ns<T> && (std::is_same_v<elem_t<T>, std::uint32_t> ||
                 std::is_same_v<elem_t<T>, std::uint64_t> ||
                 std::is_same_v<elem_t<T>, std::uint16_t>);
}
}
template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::nan_ns<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::rebind_t<NonScalar, PARAS_BI_NS::nan_t<PARAS_BI_NS::elem_t<NonScalar>>>
nan(NonScalar nancode) {
    PARAS_BI_NS::rebind_t<NonScalar, PARAS_BI_NS::nan_t<PARAS_BI_NS::elem_t<NonScalar>>> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)
        PARAS_BI_NS::put(r, i, nan(PARAS_BI_NS::at(nancode, i)));
    return r;
}

#define PARAS_BI_PTR_SCALARS(NAME, PT, BODY)                                                       \
    template <typename Ptr,                                                                        \
              std::enable_if_t<PARAS_BI_NS::ptr_to<Ptr, PT(float)>, int> = 0>                      \
    PARAS_KERNEL_HD inline float NAME(float x, Ptr p) { using F = float; BODY }                    \
    template <typename Ptr,                                                                        \
              std::enable_if_t<PARAS_BI_NS::ptr_to<Ptr, PT(double)>, int> = 0>                     \
    PARAS_KERNEL_HD inline double NAME(double x, Ptr p) { using F = double; BODY }                 \
    template <typename Ptr, std::enable_if_t<PARAS_BI_NS::ptr_to<Ptr, PT(half)>, int> = 0>         \
    PARAS_KERNEL_HD inline half NAME(half hx, Ptr p) {                                             \
        using F = float;                                                                           \
        const float x = static_cast<float>(hx);                                                    \
        PARAS_BI_HALF_##NAME                                                                       \
    }

#define PARAS_BI_SAME(T) T
#define PARAS_BI_INT(T) int

#define PARAS_BI_HALF_fract                                                                        \
    float ip;                                                                                      \
    float r = PARAS_BI_NS::s::fract(x, ip);                                                        \
    half fr(r);                                                                                    \
    if (static_cast<float>(fr) >= 1.0f) fr = PARAS_BI_NS::hfrom(0x3bffu);         \
    PARAS_BI_NS::store(p, half(ip));                                                               \
    return fr;
PARAS_BI_PTR_SCALARS(fract, PARAS_BI_SAME, F ip; F r = PARAS_BI_NS::s::fract(x, ip);
                     PARAS_BI_NS::store(p, ip); return r;)
#define PARAS_BI_HALF_frexp                                                                        \
    int e;                                                                                         \
    float m = std::frexp(x, &e);                                                                   \
    PARAS_BI_NS::store(p, e);                                                                      \
    return half(m);
PARAS_BI_PTR_SCALARS(frexp, PARAS_BI_INT, int e; F m = std::frexp(x, &e);
                     PARAS_BI_NS::store(p, e); return m;)
#define PARAS_BI_HALF_lgamma_r                                                                     \
    PARAS_BI_NS::store(p, PARAS_BI_NS::s::gamma_sign(x));                                          \
    return half(std::lgamma(x));
PARAS_BI_PTR_SCALARS(lgamma_r, PARAS_BI_INT, PARAS_BI_NS::store(p, PARAS_BI_NS::s::gamma_sign(x));
                     return static_cast<F>(std::lgamma(x));)
#define PARAS_BI_HALF_modf                                                                         \
    float ip;                                                                                      \
    float r = std::modf(x, &ip);                                                                   \
    PARAS_BI_NS::store(p, half(ip));                                                               \
    return half(r);
PARAS_BI_PTR_SCALARS(modf, PARAS_BI_SAME, F ip; F r = std::modf(x, &ip);
                     PARAS_BI_NS::store(p, ip); return r;)
#define PARAS_BI_HALF_sincos                                                                       \
    PARAS_BI_NS::store(p, half(std::cos(x)));                                                      \
    return half(std::sin(x));
PARAS_BI_PTR_SCALARS(sincos, PARAS_BI_SAME, PARAS_BI_NS::store(p, static_cast<F>(std::cos(x)));
                     return static_cast<F>(std::sin(x));)

#define PARAS_BI_PTR_NS(NAME, PTARGET)                                                             \
    template <typename NonScalar, typename Ptr,                                                    \
              std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar> &&                                    \
                                   PARAS_BI_NS::ptr_to<Ptr, PTARGET>,                              \
                               int> = 0>                                                           \
    PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> NAME(NonScalar x, Ptr p) {                \
        PARAS_BI_NS::ret_t<NonScalar> r{};                                                         \
        PTARGET out{};                                                                             \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i) {                          \
            PARAS_BI_NS::elem_t<PTARGET> o{};                                                      \
            PARAS_BI_NS::put(r, i, NAME(PARAS_BI_NS::at(x, i), &o));                               \
            PARAS_BI_NS::put(out, i, o);                                                           \
        }                                                                                          \
        PARAS_BI_NS::store(p, out);                                                                \
        return r;                                                                                  \
    }
namespace detail {
namespace paras_bi {
template <class T>
using int_of_t = rebind_t<T, int>;
}
}
PARAS_BI_PTR_NS(fract, PARAS_BI_NS::ret_t<NonScalar>)
PARAS_BI_PTR_NS(modf, PARAS_BI_NS::ret_t<NonScalar>)
PARAS_BI_PTR_NS(sincos, PARAS_BI_NS::ret_t<NonScalar>)
PARAS_BI_PTR_NS(frexp, PARAS_BI_NS::int_of_t<NonScalar>)
PARAS_BI_PTR_NS(lgamma_r, PARAS_BI_NS::int_of_t<NonScalar>)

namespace detail {
namespace paras_bi {
template <class F>
PARAS_KERNEL_HD inline F remquo7(F x, F y, int& q) {
    using W = std::conditional_t<std::is_same_v<F, float>, double, long double>;
    const F r = std::remainder(x, y);
    q = 0;
    if (std::isnan(r) || std::isinf(x) || y == F(0))
        return r;
    const W ax = std::fabs(static_cast<W>(x)), ay = std::fabs(static_cast<W>(y));
    const W big = ay * W(128);
    const W xr = std::isinf(big) ? ax : std::fmod(ax, big);
    const W rem = std::remainder(xr, ay);
    const long long k = static_cast<long long>(std::nearbyint((xr - rem) / ay)) & 127;
    q = static_cast<int>(((x < F(0)) != (y < F(0))) ? -k : k);
    return r;
}
}
}
template <typename Ptr, std::enable_if_t<PARAS_BI_NS::ptr_to<Ptr, int>, int> = 0>
PARAS_KERNEL_HD inline float remquo(float x, float y, Ptr quo) {
    int q;
    const float r = PARAS_BI_NS::remquo7(x, y, q);
    PARAS_BI_NS::store(quo, q);
    return r;
}
template <typename Ptr, std::enable_if_t<PARAS_BI_NS::ptr_to<Ptr, int>, int> = 0>
PARAS_KERNEL_HD inline double remquo(double x, double y, Ptr quo) {
    int q;
    const double r = PARAS_BI_NS::remquo7(x, y, q);
    PARAS_BI_NS::store(quo, q);
    return r;
}
template <typename Ptr, std::enable_if_t<PARAS_BI_NS::ptr_to<Ptr, int>, int> = 0>
PARAS_KERNEL_HD inline half remquo(half x, half y, Ptr quo) {
    int q;
    const float r = PARAS_BI_NS::remquo7(static_cast<float>(x), static_cast<float>(y), q);
    PARAS_BI_NS::store(quo, q);
    return half(r);
}
template <typename NonScalar1, typename NonScalar2, typename Ptr,
          std::enable_if_t<PARAS_BI_NS::fp_pair<NonScalar1, NonScalar2> &&
                               PARAS_BI_NS::ptr_to<Ptr, PARAS_BI_NS::rebind_t<NonScalar1, int>>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar1> remquo(NonScalar1 x, NonScalar2 y, Ptr quo) {
    PARAS_BI_NS::ret_t<NonScalar1> r{};
    PARAS_BI_NS::rebind_t<NonScalar1, int> q{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i) {
        int qi = 0;
        PARAS_BI_NS::put(r, i, remquo(PARAS_BI_NS::at(x, i), PARAS_BI_NS::at(y, i), &qi));
        PARAS_BI_NS::put(q, i, qi);
    }
    PARAS_BI_NS::store(quo, q);
    return r;
}

#define PARAS_BI_INT_1(NAME, OP)                                                                   \
    template <typename GenInt, std::enable_if_t<PARAS_BI_NS::gen_int<GenInt>, int> = 0>            \
    PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenInt> NAME(GenInt x) {                            \
        return PARAS_BI_NS::map1<PARAS_BI_NS::OP>(x);                                              \
    }
#define PARAS_BI_INT_2(NAME, OP)                                                                   \
    template <typename GenInt1, typename GenInt2,                                                  \
              std::enable_if_t<PARAS_BI_NS::gen_int<GenInt1> &&                                    \
                                   PARAS_BI_NS::gen_match<GenInt1, GenInt2>,                       \
                               int> = 0>                                                           \
    PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenInt1> NAME(GenInt1 x, GenInt2 y) {               \
        return PARAS_BI_NS::map2<PARAS_BI_NS::OP>(x, y);                                           \
    }
#define PARAS_BI_INT_3(NAME, OP)                                                                   \
    template <typename GenInt1, typename GenInt2, typename GenInt3,                                \
              std::enable_if_t<PARAS_BI_NS::gen_int<GenInt1> &&                                    \
                                   PARAS_BI_NS::gen_match<GenInt1, GenInt2> &&                     \
                                   PARAS_BI_NS::gen_match<GenInt1, GenInt3>,                       \
                               int> = 0>                                                           \
    PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenInt1> NAME(GenInt1 a, GenInt2 b, GenInt3 c) {    \
        return PARAS_BI_NS::map3<PARAS_BI_NS::OP>(a, b, c);                                        \
    }

PARAS_BI_INT_1(abs, op_abs)
PARAS_BI_INT_2(abs_diff, op_abs_diff)
PARAS_BI_INT_2(add_sat, op_add_sat)
PARAS_BI_INT_2(hadd, op_hadd)
PARAS_BI_INT_2(rhadd, op_rhadd)
PARAS_BI_INT_1(clz, op_clz)
PARAS_BI_INT_1(ctz, op_ctz)
PARAS_BI_INT_3(mad_hi, op_mad_hi)
PARAS_BI_INT_3(mad_sat, op_mad_sat)
PARAS_BI_INT_2(mul_hi, op_mul_hi)
PARAS_BI_INT_2(rotate, op_rotate)
PARAS_BI_INT_2(sub_sat, op_sub_sat)
PARAS_BI_INT_1(popcount, op_popcount)

namespace detail {
namespace paras_bi {
template <class T>
inline constexpr bool int32_gen =
    std::is_same_v<el_t<T>, std::int32_t> || std::is_same_v<el_t<T>, std::uint32_t>;
template <class T>
inline constexpr bool int_or_float = gen_int<T> || gen_float<T>;

template <class H, class L, class EH, class EL>
inline constexpr bool ups_ok =
    (std::is_same_v<H, EH> && std::is_same_v<L, EL>) ||
    (ns<H>::kind == 1 && ns<L>::kind == 1 && ns<H>::n == ns<L>::n &&
     std::is_same_v<elem_t<H>, EH> && std::is_same_v<elem_t<L>, EL>) ||
    (ns<H>::kind == 2 && ns<L>::kind == 2 && ns<H>::n == ns<L>::n &&
     std::is_same_v<elem_t<H>, EH> && std::is_same_v<elem_t<L>, EL>);
template <class H>
using ups_wide_t = std::conditional_t<
    sizeof(el_t<H>) == 1,
    std::conditional_t<std::is_signed_v<el_t<H>>, std::int16_t, std::uint16_t>,
    std::conditional_t<sizeof(el_t<H>) == 2,
                       std::conditional_t<std::is_signed_v<el_t<H>>, std::int32_t, std::uint32_t>,
                       std::conditional_t<std::is_signed_v<el_t<H>>, std::int64_t, std::uint64_t>>>;
template <class H>
using ups_ret_t = std::conditional_t<is_ns<H>, rebind_t<H, ups_wide_t<H>>, ups_wide_t<H>>;
template <class H, class L>
inline constexpr bool ups_any = ups_ok<H, L, std::uint8_t, std::uint8_t> ||
                                ups_ok<H, L, std::int8_t, std::uint8_t> ||
                                ups_ok<H, L, std::uint16_t, std::uint16_t> ||
                                ups_ok<H, L, std::int16_t, std::uint16_t> ||
                                ups_ok<H, L, std::uint32_t, std::uint32_t> ||
                                ups_ok<H, L, std::int32_t, std::uint32_t>;
}
}

template <typename Int32Bit1, typename Int32Bit2, typename Int32Bit3,
          std::enable_if_t<PARAS_BI_NS::gen_int<Int32Bit1> && PARAS_BI_NS::int32_gen<Int32Bit1> &&
                               PARAS_BI_NS::gen_match<Int32Bit1, Int32Bit2> &&
                               PARAS_BI_NS::gen_match<Int32Bit1, Int32Bit3>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<Int32Bit1> mad24(Int32Bit1 x, Int32Bit2 y, Int32Bit3 z) {
    return PARAS_BI_NS::map3<PARAS_BI_NS::op_mad24>(x, y, z);
}
template <typename Int32Bit1, typename Int32Bit2,
          std::enable_if_t<PARAS_BI_NS::gen_int<Int32Bit1> && PARAS_BI_NS::int32_gen<Int32Bit1> &&
                               PARAS_BI_NS::gen_match<Int32Bit1, Int32Bit2>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<Int32Bit1> mul24(Int32Bit1 x, Int32Bit2 y) {
    return PARAS_BI_NS::map2<PARAS_BI_NS::op_mul24>(x, y);
}

template <typename Hi, typename Lo,
          std::enable_if_t<PARAS_BI_NS::ups_any<Hi, Lo>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ups_ret_t<Hi> upsample(Hi hi, Lo lo) {
    using W = PARAS_BI_NS::ups_wide_t<Hi>;
    using UW = std::make_unsigned_t<W>;
    constexpr int sh = PARAS_BI_NS::bits<PARAS_BI_NS::el_t<Hi>>;
    if constexpr (PARAS_BI_NS::is_ns<Hi>) {
        PARAS_BI_NS::ups_ret_t<Hi> r{};
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<Hi>::n; ++i)
            PARAS_BI_NS::put(r, i,
                             static_cast<W>((static_cast<UW>(static_cast<W>(PARAS_BI_NS::at(hi, i))) << sh) |
                                            static_cast<UW>(PARAS_BI_NS::at(lo, i))));
        return r;
    } else {
        return static_cast<W>((static_cast<UW>(static_cast<W>(hi)) << sh) | static_cast<UW>(lo));
    }
}

template <typename T1, typename T2, typename T3,
          std::enable_if_t<PARAS_BI_NS::int_or_float<T1> && PARAS_BI_NS::gen_match<T1, T2> &&
                               PARAS_BI_NS::gen_match<T1, T3>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<T1> clamp(T1 x, T2 minval, T3 maxval) {
    return PARAS_BI_NS::map3<PARAS_BI_NS::op_clamp>(x, minval, maxval);
}
template <typename NonScalar,
          std::enable_if_t<PARAS_BI_NS::is_ns<NonScalar> && PARAS_BI_NS::int_or_float<NonScalar>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> clamp(NonScalar x,
                                                           typename NonScalar::value_type minval,
                                                           typename NonScalar::value_type maxval) {
    PARAS_BI_NS::ret_t<NonScalar> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)
        PARAS_BI_NS::put(r, i, PARAS_BI_NS::op_clamp::f(PARAS_BI_NS::at(x, i), minval, maxval));
    return r;
}
#define PARAS_BI_MINMAX(NAME, OP)                                                                  \
    template <typename T1, typename T2,                                                            \
              std::enable_if_t<PARAS_BI_NS::int_or_float<T1> && PARAS_BI_NS::gen_match<T1, T2>,    \
                               int> = 0>                                                           \
    PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<T1> NAME(T1 x, T2 y) {                              \
        return PARAS_BI_NS::map2<PARAS_BI_NS::OP>(x, y);                                           \
    }                                                                                              \
    template <typename NonScalar,                                                                  \
              std::enable_if_t<PARAS_BI_NS::is_ns<NonScalar> &&                                    \
                                   PARAS_BI_NS::int_or_float<NonScalar>,                           \
                               int> = 0>                                                           \
    PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> NAME(NonScalar x,                         \
                                                              typename NonScalar::value_type y) {  \
        PARAS_BI_NS::ret_t<NonScalar> r{};                                                         \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)                            \
            PARAS_BI_NS::put(r, i, PARAS_BI_NS::OP::f(PARAS_BI_NS::at(x, i), y));                  \
        return r;                                                                                  \
    }
PARAS_BI_MINMAX(max, op_max)
PARAS_BI_MINMAX(min, op_min)

namespace detail {
namespace paras_bi {
template <class T>
using calc_t = std::conditional_t<std::is_same_v<T, half>, float, T>;
struct op_degrees {
    template <class T>
    PARAS_KERNEL_HD static T f(T r) { return T(static_cast<calc_t<T>>(180.0 / pi) * static_cast<calc_t<T>>(r)); }
};
struct op_radians {
    template <class T>
    PARAS_KERNEL_HD static T f(T d) { return T(static_cast<calc_t<T>>(pi / 180.0) * static_cast<calc_t<T>>(d)); }
};
struct op_mix {
    template <class T>
    PARAS_KERNEL_HD static T f(T x, T y, T a) {
        using C = calc_t<T>;
        return T(C(x) + (C(y) - C(x)) * C(a));
    }
};
struct op_step {
    template <class T>
    PARAS_KERNEL_HD static T f(T edge, T x) { return calc_t<T>(x) < calc_t<T>(edge) ? T(0.0f) : T(1.0f); }
};
struct op_smoothstep {
    template <class T>
    PARAS_KERNEL_HD static T f(T e0, T e1, T x) {
        using C = calc_t<T>;
        auto rt = [](C v) { return static_cast<C>(static_cast<T>(v)); };
        C t = rt((C(x) - C(e0)) / (C(e1) - C(e0)));
        t = std::fmin(std::fmax(t, C(0)), C(1));
        return T(t * t * (C(3) - C(2) * t));
    }
};
struct op_sign {
    template <class T>
    PARAS_KERNEL_HD static T f(T x) {
        using C = calc_t<T>;
        const C v = C(x);
        if (std::isnan(v)) return T(0.0f);
        if (v > C(0)) return T(1.0f);
        if (v < C(0)) return T(-1.0f);
        return x;
    }
};
}
}

template <typename GenFloat, std::enable_if_t<PARAS_BI_NS::gen_float<GenFloat>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenFloat> degrees(GenFloat radians) {
    return PARAS_BI_NS::map1<PARAS_BI_NS::op_degrees>(radians);
}
template <typename GenFloat, std::enable_if_t<PARAS_BI_NS::gen_float<GenFloat>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenFloat> radians(GenFloat degrees) {
    return PARAS_BI_NS::map1<PARAS_BI_NS::op_radians>(degrees);
}
template <typename GenFloat, std::enable_if_t<PARAS_BI_NS::gen_float<GenFloat>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenFloat> sign(GenFloat x) {
    return PARAS_BI_NS::map1<PARAS_BI_NS::op_sign>(x);
}
template <typename GenFloat1, typename GenFloat2, typename GenFloat3,
          std::enable_if_t<PARAS_BI_NS::gen_float<GenFloat1> &&
                               PARAS_BI_NS::gen_match<GenFloat1, GenFloat2> &&
                               PARAS_BI_NS::gen_match<GenFloat1, GenFloat3>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenFloat1> mix(GenFloat1 x, GenFloat2 y, GenFloat3 a) {
    return PARAS_BI_NS::map3<PARAS_BI_NS::op_mix>(x, y, a);
}
template <typename NonScalar1, typename NonScalar2,
          std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar1> &&
                               PARAS_BI_NS::gen_match<NonScalar1, NonScalar2>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar1> mix(NonScalar1 x, NonScalar2 y,
                                                          typename NonScalar1::value_type a) {
    PARAS_BI_NS::ret_t<NonScalar1> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i)
        PARAS_BI_NS::put(r, i,
                         PARAS_BI_NS::op_mix::f(PARAS_BI_NS::at(x, i), PARAS_BI_NS::at(y, i),
                                                static_cast<PARAS_BI_NS::elem_t<NonScalar1>>(a)));
    return r;
}
template <typename GenFloat1, typename GenFloat2,
          std::enable_if_t<PARAS_BI_NS::gen_float<GenFloat1> &&
                               PARAS_BI_NS::gen_match<GenFloat1, GenFloat2>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenFloat1> step(GenFloat1 edge, GenFloat2 x) {
    return PARAS_BI_NS::map2<PARAS_BI_NS::op_step>(edge, x);
}
template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> step(typename NonScalar::value_type edge,
                                                          NonScalar x) {
    PARAS_BI_NS::ret_t<NonScalar> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)
        PARAS_BI_NS::put(r, i, PARAS_BI_NS::op_step::f(static_cast<PARAS_BI_NS::elem_t<NonScalar>>(edge),
                                                       PARAS_BI_NS::at(x, i)));
    return r;
}
template <typename GenFloat1, typename GenFloat2, typename GenFloat3,
          std::enable_if_t<PARAS_BI_NS::gen_float<GenFloat1> &&
                               PARAS_BI_NS::gen_match<GenFloat1, GenFloat2> &&
                               PARAS_BI_NS::gen_match<GenFloat1, GenFloat3>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenFloat1> smoothstep(GenFloat1 edge0, GenFloat2 edge1,
                                                                 GenFloat3 x) {
    return PARAS_BI_NS::map3<PARAS_BI_NS::op_smoothstep>(edge0, edge1, x);
}
template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar> smoothstep(
    typename NonScalar::value_type edge0, typename NonScalar::value_type edge1, NonScalar x) {
    using E = PARAS_BI_NS::elem_t<NonScalar>;
    PARAS_BI_NS::ret_t<NonScalar> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)
        PARAS_BI_NS::put(r, i, PARAS_BI_NS::op_smoothstep::f(static_cast<E>(edge0), static_cast<E>(edge1),
                                                             PARAS_BI_NS::at(x, i)));
    return r;
}

namespace detail {
namespace paras_bi {
template <class A, class B>
PARAS_KERNEL_HD inline double dot_acc(const A& a, const B& b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < ns<A>::n; ++i)
        acc += static_cast<double>(static_cast<calc_t<el_t<A>>>(at(a, i))) *
               static_cast<double>(static_cast<calc_t<el_t<A>>>(at(b, i)));
    return acc;
}
}
}

template <typename Geo3or4Float1, typename Geo3or4Float2,
          std::enable_if_t<PARAS_BI_NS::fp_ns<Geo3or4Float1> &&
                               (PARAS_BI_NS::ns<Geo3or4Float1>::n == 3 ||
                                PARAS_BI_NS::ns<Geo3or4Float1>::n == 4) &&
                               PARAS_BI_NS::gen_match<Geo3or4Float1, Geo3or4Float2>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<Geo3or4Float1> cross(Geo3or4Float1 p0, Geo3or4Float2 p1) {
    using E = PARAS_BI_NS::elem_t<Geo3or4Float1>;
    using C = PARAS_BI_NS::calc_t<E>;
    auto a = [&](std::size_t i) { return static_cast<C>(PARAS_BI_NS::at(p0, i)); };
    auto b = [&](std::size_t i) { return static_cast<C>(PARAS_BI_NS::at(p1, i)); };
    PARAS_BI_NS::ret_t<Geo3or4Float1> r{};
    PARAS_BI_NS::put(r, 0, E(a(1) * b(2) - a(2) * b(1)));
    PARAS_BI_NS::put(r, 1, E(a(2) * b(0) - a(0) * b(2)));
    PARAS_BI_NS::put(r, 2, E(a(0) * b(1) - a(1) * b(0)));
    if constexpr (PARAS_BI_NS::ns<Geo3or4Float1>::n == 4) PARAS_BI_NS::put(r, 3, E(0.0f));
    return r;
}
template <typename GeoFloat1, typename GeoFloat2,
          std::enable_if_t<PARAS_BI_NS::geo<GeoFloat1> && PARAS_BI_NS::gen_match<GeoFloat1, GeoFloat2>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::el_t<GeoFloat1> dot(GeoFloat1 p0, GeoFloat2 p1) {
    return static_cast<PARAS_BI_NS::el_t<GeoFloat1>>(
        static_cast<PARAS_BI_NS::calc_t<PARAS_BI_NS::el_t<GeoFloat1>>>(PARAS_BI_NS::dot_acc(p0, p1)));
}
template <typename GeoFloat, std::enable_if_t<PARAS_BI_NS::geo<GeoFloat>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::el_t<GeoFloat> length(GeoFloat p) {
    using E = PARAS_BI_NS::el_t<GeoFloat>;
    return static_cast<E>(static_cast<PARAS_BI_NS::calc_t<E>>(std::sqrt(PARAS_BI_NS::dot_acc(p, p))));
}
template <typename GeoFloat1, typename GeoFloat2,
          std::enable_if_t<PARAS_BI_NS::geo<GeoFloat1> && PARAS_BI_NS::gen_match<GeoFloat1, GeoFloat2>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::el_t<GeoFloat1> distance(GeoFloat1 p0, GeoFloat2 p1) {
    using E = PARAS_BI_NS::el_t<GeoFloat1>;
    double acc = 0.0;
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<GeoFloat1>::n; ++i) {
        const double d = static_cast<double>(static_cast<PARAS_BI_NS::calc_t<E>>(PARAS_BI_NS::at(p0, i))) -
                         static_cast<double>(static_cast<PARAS_BI_NS::calc_t<E>>(PARAS_BI_NS::at(p1, i)));
        acc += d * d;
    }
    return static_cast<E>(static_cast<PARAS_BI_NS::calc_t<E>>(std::sqrt(acc)));
}
template <typename GeoFloat, std::enable_if_t<PARAS_BI_NS::geo<GeoFloat>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GeoFloat> normalize(GeoFloat p) {
    using E = PARAS_BI_NS::el_t<GeoFloat>;
    const double len = std::sqrt(PARAS_BI_NS::dot_acc(p, p));
    if constexpr (PARAS_BI_NS::is_ns<GeoFloat>) {
        PARAS_BI_NS::gret_t<GeoFloat> r{};
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<GeoFloat>::n; ++i) {
            const double v = static_cast<double>(static_cast<PARAS_BI_NS::calc_t<E>>(PARAS_BI_NS::at(p, i)));
            PARAS_BI_NS::put(r, i, static_cast<E>(static_cast<PARAS_BI_NS::calc_t<E>>(len == 0.0 ? v : v / len)));
        }
        return r;
    } else {
        const double v = static_cast<double>(static_cast<PARAS_BI_NS::calc_t<E>>(p));
        return static_cast<E>(static_cast<PARAS_BI_NS::calc_t<E>>(len == 0.0 ? v : v / len));
    }
}
template <typename GeoFloat, std::enable_if_t<PARAS_BI_NS::geo_float<GeoFloat>, int> = 0>
PARAS_KERNEL_HD inline float fast_length(GeoFloat p) {
    return length(p);
}
template <typename GeoFloat1, typename GeoFloat2,
          std::enable_if_t<PARAS_BI_NS::geo_float<GeoFloat1> &&
                               PARAS_BI_NS::gen_match<GeoFloat1, GeoFloat2>,
                           int> = 0>
PARAS_KERNEL_HD inline float fast_distance(GeoFloat1 p0, GeoFloat2 p1) {
    return distance(p0, p1);
}
template <typename GeoFloat, std::enable_if_t<PARAS_BI_NS::geo_float<GeoFloat>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GeoFloat> fast_normalize(GeoFloat p) {
    return normalize(p);
}

namespace detail {
namespace paras_bi {
PARAS_KERNEL_HD inline bool h_isnormal(half h) {
    const unsigned e = (hbits(h) >> 10) & 0x1fu;
    return e != 0u && e != 0x1fu;
}
}
}

#define PARAS_BI_REL2(NAME, EXPR)                                                                  \
    PARAS_KERNEL_HD inline bool NAME(float x, float y) { return EXPR; }                           \
    PARAS_KERNEL_HD inline bool NAME(double x, double y) { return EXPR; }                         \
    PARAS_KERNEL_HD inline bool NAME(half hx, half hy) {                                           \
        const float x = static_cast<float>(hx), y = static_cast<float>(hy);                        \
        return EXPR;                                                                               \
    }                                                                                              \
    template <typename NonScalar1, typename NonScalar2,                                            \
              std::enable_if_t<PARAS_BI_NS::fp_pair<NonScalar1, NonScalar2>, int> = 0>             \
    PARAS_KERNEL_HD inline PARAS_BI_NS::rel_ret_t<NonScalar1> NAME(NonScalar1 x, NonScalar2 y) {   \
        PARAS_BI_NS::rel_ret_t<NonScalar1> r{};                                                    \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i)                           \
            PARAS_BI_NS::put(r, i,                                                                 \
                             PARAS_BI_NS::rel_val<NonScalar1>(                                     \
                                 NAME(PARAS_BI_NS::at(x, i), PARAS_BI_NS::at(y, i))));             \
        return r;                                                                                  \
    }
#define PARAS_BI_REL1(NAME, EXPR, HEXPR)                                                           \
    PARAS_KERNEL_HD inline bool NAME(float x) { return EXPR; }                                    \
    PARAS_KERNEL_HD inline bool NAME(double x) { return EXPR; }                                   \
    PARAS_KERNEL_HD inline bool NAME(half hx) { return HEXPR; }                                    \
    template <typename NonScalar, std::enable_if_t<PARAS_BI_NS::fp_ns<NonScalar>, int> = 0>        \
    PARAS_KERNEL_HD inline PARAS_BI_NS::rel_ret_t<NonScalar> NAME(NonScalar x) {                   \
        PARAS_BI_NS::rel_ret_t<NonScalar> r{};                                                     \
        for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar>::n; ++i)                            \
            PARAS_BI_NS::put(r, i, PARAS_BI_NS::rel_val<NonScalar>(NAME(PARAS_BI_NS::at(x, i))));  \
        return r;                                                                                  \
    }

PARAS_BI_REL2(isequal, x == y)
PARAS_BI_REL2(isnotequal, x != y)
PARAS_BI_REL2(isgreater, x > y)
PARAS_BI_REL2(isgreaterequal, x >= y)
PARAS_BI_REL2(isless, x < y)
PARAS_BI_REL2(islessequal, x <= y)
PARAS_BI_REL2(islessgreater, (x < y) || (x > y))
PARAS_BI_REL2(isordered, (x == x) && (y == y))
PARAS_BI_REL2(isunordered, std::isnan(x) || std::isnan(y))
PARAS_BI_REL1(isfinite, std::isfinite(x), std::isfinite(static_cast<float>(hx)))
PARAS_BI_REL1(isinf, std::isinf(x), std::isinf(static_cast<float>(hx)))
PARAS_BI_REL1(isnan, std::isnan(x), std::isnan(static_cast<float>(hx)))
PARAS_BI_REL1(isnormal, std::isnormal(x), PARAS_BI_NS::h_isnormal(hx))
PARAS_BI_REL1(signbit, std::signbit(x), (PARAS_BI_NS::hbits(hx) & 0x8000u) != 0u)

namespace detail {
namespace paras_bi {
template <class T>
inline constexpr bool anyall1 =
    (ns<T>::kind == 1 && std::is_same_v<elem_t<T>, bool>) ||
    (ns<T>::kind == 2 && (std::is_same_v<elem_t<T>, std::int8_t> ||
                          std::is_same_v<elem_t<T>, std::int16_t> ||
                          std::is_same_v<elem_t<T>, std::int32_t> ||
                          std::is_same_v<elem_t<T>, std::int64_t>));
template <class E>
inline constexpr bool signed_gen = std::is_same_v<E, signed char> || std::is_same_v<E, short> ||
                                   std::is_same_v<E, int> || std::is_same_v<E, long> ||
                                   std::is_same_v<E, long long>;
template <class T>
inline constexpr bool anyall2 = signed_gen<T> || (ns<T>::kind == 1 && signed_gen<elem_t<T>>);
template <class T>
using anyall_ret_t = std::conditional_t<ns<T>::kind == 1, bool, int>;
}
}

template <typename GenInt, std::enable_if_t<PARAS_BI_NS::anyall1<GenInt>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::anyall_ret_t<GenInt> any(GenInt x) {
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<GenInt>::n; ++i) {
        if constexpr (PARAS_BI_NS::ns<GenInt>::kind == 1) {
            if (PARAS_BI_NS::at(x, i)) return true;
        } else if (PARAS_BI_NS::msb(PARAS_BI_NS::at(x, i))) {
            return 1;
        }
    }
    return 0;
}
template <typename GenInt, std::enable_if_t<PARAS_BI_NS::anyall1<GenInt>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::anyall_ret_t<GenInt> all(GenInt x) {
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<GenInt>::n; ++i) {
        if constexpr (PARAS_BI_NS::ns<GenInt>::kind == 1) {
            if (!PARAS_BI_NS::at(x, i)) return false;
        } else if (!PARAS_BI_NS::msb(PARAS_BI_NS::at(x, i))) {
            return 0;
        }
    }
    return 1;
}
template <typename GenInt, std::enable_if_t<PARAS_BI_NS::anyall2<GenInt>, int> = 0, typename = void>
[[deprecated("SYCL 2020: any(scalar/marray of signed integers) is deprecated")]]
PARAS_KERNEL_HD inline bool any(GenInt x) {
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<GenInt>::n; ++i)
        if (PARAS_BI_NS::msb(PARAS_BI_NS::at(x, i))) return true;
    return false;
}
template <typename GenInt, std::enable_if_t<PARAS_BI_NS::anyall2<GenInt>, int> = 0, typename = void>
[[deprecated("SYCL 2020: all(scalar/marray of signed integers) is deprecated")]]
PARAS_KERNEL_HD inline bool all(GenInt x) {
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<GenInt>::n; ++i)
        if (!PARAS_BI_NS::msb(PARAS_BI_NS::at(x, i))) return false;
    return true;
}

namespace detail {
namespace paras_bi {
struct op_bitselect {
    template <class T>
    PARAS_KERNEL_HD static T f(T a, T b, T c) {
        using B = bits_t<T>;
        const B ba = __builtin_bit_cast(B, a), bb = __builtin_bit_cast(B, b),
                bc = __builtin_bit_cast(B, c);
        return __builtin_bit_cast(T, static_cast<B>((ba & ~bc) | (bb & bc)));
    }
};
template <class X, class Y>
constexpr bool same_size_f() {
    if constexpr (std::is_void_v<X> || std::is_void_v<Y>) return false;
    else return sizeof(X) == sizeof(Y);
}
template <class X, class Y>
inline constexpr bool same_size = same_size_f<X, Y>();
template <class A, class B, class C>
inline constexpr bool select_ok =
    (ns<A>::kind == 1 && gen_scalar<elem_t<A>> && std::is_same_v<A, B> && ns<C>::kind == 1 &&
     ns<C>::n == ns<A>::n && std::is_same_v<elem_t<C>, bool>) ||
    (ns<A>::kind == 2 && vec_elem<elem_t<A>> && same_shape<A, B> && ns<C>::kind == 2 &&
     ns<C>::n == ns<A>::n && std::is_integral_v<elem_t<C>> &&
     same_size<elem_t<C>, elem_t<A>>);
}
}

template <typename GenType1, typename GenType2, typename GenType3,
          std::enable_if_t<PARAS_BI_NS::gen_type<GenType1> &&
                               PARAS_BI_NS::gen_match<GenType1, GenType2> &&
                               PARAS_BI_NS::gen_match<GenType1, GenType3>,
                           int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::gret_t<GenType1> bitselect(GenType1 a, GenType2 b, GenType3 c) {
    return PARAS_BI_NS::map3<PARAS_BI_NS::op_bitselect>(a, b, c);
}
template <typename Scalar, std::enable_if_t<PARAS_BI_NS::gen_scalar<Scalar>, int> = 0>
PARAS_KERNEL_HD inline Scalar select(Scalar a, Scalar b, bool c) {
    return c ? b : a;
}
template <typename NonScalar1, typename NonScalar2, typename NonScalar3,
          std::enable_if_t<PARAS_BI_NS::select_ok<NonScalar1, NonScalar2, NonScalar3>, int> = 0>
PARAS_KERNEL_HD inline PARAS_BI_NS::ret_t<NonScalar1> select(NonScalar1 a, NonScalar2 b,
                                                             NonScalar3 c) {
    PARAS_BI_NS::ret_t<NonScalar1> r{};
    for (std::size_t i = 0; i < PARAS_BI_NS::ns<NonScalar1>::n; ++i) {
        bool pick;
        if constexpr (PARAS_BI_NS::ns<NonScalar1>::kind == 1) pick = PARAS_BI_NS::at(c, i);
        else pick = PARAS_BI_NS::msb(PARAS_BI_NS::at(c, i));
        PARAS_BI_NS::put(r, i, pick ? PARAS_BI_NS::at(b, i) : PARAS_BI_NS::at(a, i));
    }
    return r;
}

#undef PARAS_BI_UNARY
#undef PARAS_BI_BINARY
#undef PARAS_BI_TERNARY
#undef PARAS_BI_SCALAR2
#undef PARAS_BI_S
#undef PARAS_BI_INT2
#undef PARAS_BI_PTR_SCALARS
#undef PARAS_BI_SAME
#undef PARAS_BI_INT
#undef PARAS_BI_PTR_NS
#undef PARAS_BI_INT_1
#undef PARAS_BI_INT_2
#undef PARAS_BI_INT_3
#undef PARAS_BI_MINMAX
#undef PARAS_BI_REL2
#undef PARAS_BI_REL1

}

#endif
