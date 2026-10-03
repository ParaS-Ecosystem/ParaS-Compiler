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

#ifndef __PARAS_RANGE_HPP__
#define __PARAS_RANGE_HPP__

#include <cstddef>
#include <type_traits>
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl
{

    template <int Dimensions = 1>
    class range
    {
    public:
        static constexpr int dimensions = Dimensions;

        size_t dims_[Dimensions];

        PARAS_KERNEL_HD
        range()
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] = 0;
        }

        template <int D = Dimensions,
                  typename std::enable_if<D == 1, int>::type = 0>
        PARAS_KERNEL_HD
        range(size_t dim0)
        {
            dims_[0] = dim0;
        }

        template <int D = Dimensions,
                  typename std::enable_if<D == 2, int>::type = 0>
        PARAS_KERNEL_HD
        range(size_t dim0, size_t dim1)
        {
            dims_[0] = dim0;
            dims_[1] = dim1;
        }

        template <int D = Dimensions,
                  typename std::enable_if<D == 3, int>::type = 0>
        PARAS_KERNEL_HD
        range(size_t dim0, size_t dim1, size_t dim2)
        {
            dims_[0] = dim0;
            dims_[1] = dim1;
            dims_[2] = dim2;
        }

        PARAS_KERNEL_HD
        size_t get(int d) const
        {
            return dims_[d];
        }

        PARAS_KERNEL_HD
        size_t &operator[](int d) { return dims_[d]; }

        PARAS_KERNEL_HD
        size_t operator[](int d) const { return dims_[d]; }

        PARAS_KERNEL_HD
        range operator*(const range &rhs) const
        {

            range result;
            for (int i = 0; i < Dimensions; ++i)
            {
                result[i] = dims_[i] * rhs[i];
            }
            return result;
        }

        PARAS_KERNEL_HD
        range operator+(const range &rhs) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] + rhs[i];
            return result;
        }
        PARAS_KERNEL_HD
        range operator-(const range &rhs) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] - rhs[i];
            return result;
        }
        PARAS_KERNEL_HD
        range operator/(const range &rhs) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] / rhs[i];
            return result;
        }
        PARAS_KERNEL_HD
        range operator%(const range &rhs) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] % rhs[i];
            return result;
        }

        PARAS_KERNEL_HD
        range operator+(size_t scalar) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] + scalar;
            return result;
        }
        PARAS_KERNEL_HD
        range operator-(size_t scalar) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] - scalar;
            return result;
        }
        PARAS_KERNEL_HD
        range operator*(size_t scalar) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] * scalar;
            return result;
        }
        PARAS_KERNEL_HD
        range operator/(size_t scalar) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] / scalar;
            return result;
        }
        PARAS_KERNEL_HD
        range operator%(size_t scalar) const
        {
            range result;
            for (int i = 0; i < Dimensions; ++i)
                result[i] = dims_[i] % scalar;
            return result;
        }

        PARAS_KERNEL_HD
        range &operator+=(const range &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] += rhs[i];
            return *this;
        }
        PARAS_KERNEL_HD
        range &operator-=(const range &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] -= rhs[i];
            return *this;
        }
        PARAS_KERNEL_HD
        range &operator*=(const range &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] *= rhs[i];
            return *this;
        }
        PARAS_KERNEL_HD
        range &operator/=(const range &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] /= rhs[i];
            return *this;
        }
        PARAS_KERNEL_HD
        range &operator%=(const range &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] %= rhs[i];
            return *this;
        }

        PARAS_KERNEL_HD
        bool operator==(const range &rhs) const
        {
            for (int i = 0; i < Dimensions; ++i)
                if (dims_[i] != rhs.dims_[i])
                    return false;
            return true;
        }
        PARAS_KERNEL_HD
        bool operator!=(const range &rhs) const { return !(*this == rhs); }

        PARAS_KERNEL_HD
        size_t size() const
        {
            if constexpr (Dimensions == 1)
                return dims_[0];
            else if constexpr (Dimensions == 2)
                return dims_[0] * dims_[1];
            else
                return dims_[0] * dims_[1] * dims_[2];
        }

#define PARAS_RANGE_BINOP(OP)                                                 \
        PARAS_KERNEL_HD friend range operator OP(const range &a,              \
                                                 const range &b) {            \
            range r(a);                                                       \
            for (int d = 0; d < Dimensions; ++d)                              \
                r.dims_[d] = a.dims_[d] OP b.dims_[d];                        \
            return r;                                                         \
        }                                                                     \
        PARAS_KERNEL_HD friend range operator OP(const range &a, size_t s) {  \
            range r(a);                                                       \
            for (int d = 0; d < Dimensions; ++d)                              \
                r.dims_[d] = a.dims_[d] OP s;                                 \
            return r;                                                         \
        }                                                                     \
        PARAS_KERNEL_HD friend range operator OP(size_t s, const range &a) {  \
            range r(a);                                                       \
            for (int d = 0; d < Dimensions; ++d)                              \
                r.dims_[d] = s OP a.dims_[d];                                 \
            return r;                                                         \
        }
        PARAS_RANGE_BINOP(<<)
        PARAS_RANGE_BINOP(>>)
        PARAS_RANGE_BINOP(&)
        PARAS_RANGE_BINOP(|)
        PARAS_RANGE_BINOP(^)
        PARAS_RANGE_BINOP(&&)
        PARAS_RANGE_BINOP(||)
        PARAS_RANGE_BINOP(<)
        PARAS_RANGE_BINOP(>)
        PARAS_RANGE_BINOP(<=)
        PARAS_RANGE_BINOP(>=)
#undef PARAS_RANGE_BINOP

#define PARAS_RANGE_COMPOUND(OP)                                              \
        PARAS_KERNEL_HD friend range &operator OP##=(range & a,               \
                                                     const range &b) {        \
            for (int d = 0; d < Dimensions; ++d)                              \
                a.dims_[d] OP## = b.dims_[d];                                 \
            return a;                                                         \
        }                                                                     \
        PARAS_KERNEL_HD friend range &operator OP##=(range & a, size_t s) {   \
            for (int d = 0; d < Dimensions; ++d)                              \
                a.dims_[d] OP## = s;                                          \
            return a;                                                         \
        }
        PARAS_RANGE_COMPOUND(<<)
        PARAS_RANGE_COMPOUND(>>)
        PARAS_RANGE_COMPOUND(&)
        PARAS_RANGE_COMPOUND(|)
        PARAS_RANGE_COMPOUND(^)
#undef PARAS_RANGE_COMPOUND

#define PARAS_RANGE_SCALAR_COMPOUND(OP)                                       \
        PARAS_KERNEL_HD friend range &operator OP##=(range & a, size_t s) {   \
            for (int d = 0; d < Dimensions; ++d)                              \
                a.dims_[d] OP## = s;                                          \
            return a;                                                         \
        }
        PARAS_RANGE_SCALAR_COMPOUND(+)
        PARAS_RANGE_SCALAR_COMPOUND(-)
        PARAS_RANGE_SCALAR_COMPOUND(*)
        PARAS_RANGE_SCALAR_COMPOUND(/)
        PARAS_RANGE_SCALAR_COMPOUND(%)
#undef PARAS_RANGE_SCALAR_COMPOUND
        PARAS_KERNEL_HD friend range operator%(size_t s, const range &a) {
            range r(a);
            for (int d = 0; d < Dimensions; ++d)
                r.dims_[d] = s % a.dims_[d];
            return r;
        }

        PARAS_KERNEL_HD friend range operator+(const range &a) { return a; }
        PARAS_KERNEL_HD friend range operator-(const range &a) {
            range r(a);
            for (int d = 0; d < Dimensions; ++d)
                r.dims_[d] = -a.dims_[d];
            return r;
        }
        PARAS_KERNEL_HD friend range &operator++(range &a) {
            for (int d = 0; d < Dimensions; ++d)
                ++a.dims_[d];
            return a;
        }
        PARAS_KERNEL_HD friend range &operator--(range &a) {
            for (int d = 0; d < Dimensions; ++d)
                --a.dims_[d];
            return a;
        }
        PARAS_KERNEL_HD friend range operator++(range &a, int) {
            range old(a);
            ++a;
            return old;
        }
        PARAS_KERNEL_HD friend range operator--(range &a, int) {
            range old(a);
            --a;
            return old;
        }
    };

    range(size_t) -> range<1>;
    range(size_t, size_t) -> range<2>;
    range(size_t, size_t, size_t) -> range<3>;

    template <int Dimensions>
    PARAS_KERNEL_HD
    inline range<Dimensions> operator+(size_t scalar, const range<Dimensions> &r)
    {
        range<Dimensions> result;
        for (int i = 0; i < Dimensions; ++i) result[i] = scalar + r[i];
        return result;
    }
    template <int Dimensions>
    PARAS_KERNEL_HD
    inline range<Dimensions> operator-(size_t scalar, const range<Dimensions> &r)
    {
        range<Dimensions> result;
        for (int i = 0; i < Dimensions; ++i) result[i] = scalar - r[i];
        return result;
    }
    template <int Dimensions>
    PARAS_KERNEL_HD
    inline range<Dimensions> operator*(size_t scalar, const range<Dimensions> &r)
    {
        range<Dimensions> result;
        for (int i = 0; i < Dimensions; ++i) result[i] = scalar * r[i];
        return result;
    }
    template <int Dimensions>
    PARAS_KERNEL_HD
    inline range<Dimensions> operator/(size_t scalar, const range<Dimensions> &r)
    {
        range<Dimensions> result;
        for (int i = 0; i < Dimensions; ++i) result[i] = scalar / r[i];
        return result;
    }

}

#endif
