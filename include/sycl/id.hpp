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

#ifndef __PARAS_ID_HPP__
#define __PARAS_ID_HPP__

#include <cstddef>
#include <type_traits>
#include "range.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl
{
    namespace detail {

        template <int Dimensions, typename Derived>
        struct id_size_t_conv_base {};

        template <typename Derived>
        struct id_size_t_conv_base<1, Derived> {
            PARAS_KERNEL_HD
            operator size_t() const {
                return static_cast<const Derived *>(this)->dims_[0];
            }
        };
    }

    template <int Dimensions = 1>
    class id : public detail::id_size_t_conv_base<Dimensions, id<Dimensions>>
    {
    public:
        static constexpr int dimensions = Dimensions;
        size_t dims_[Dimensions];

        PARAS_KERNEL_HD
        id()
        {
            for (int i = 0; i < Dimensions; i++)
                dims_[i] = 0;
        }

        PARAS_KERNEL_HD
        id(size_t i0)
        {
            static_assert(Dimensions == 1,
                          "id(size_t) is only valid for id<1>");
            dims_[0] = i0;
        }

        PARAS_KERNEL_HD
        id(size_t i0, size_t i1)
        {
            static_assert(Dimensions == 2,
                          "id(size_t,size_t) is only valid for id<2>");
            dims_[0] = i0;
            dims_[1] = i1;
        }

        PARAS_KERNEL_HD
        id(size_t i0, size_t i1, size_t i2)
        {
            static_assert(Dimensions == 3,
                          "id(size_t,size_t,size_t) is only valid for id<3>");
            dims_[0] = i0;
            dims_[1] = i1;
            dims_[2] = i2;
        }

        PARAS_KERNEL_HD
        id(const range<Dimensions> &r)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] = r[i];
        }

        PARAS_KERNEL_HD
        id &operator=(const range<Dimensions> &r)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] = r[i];
            return *this;
        }

        PARAS_KERNEL_HD
        size_t &operator[](int dim)
        {
            return dims_[dim];
        }

        PARAS_KERNEL_HD
        size_t operator[](int dim) const
        {
            return dims_[dim];
        }

        PARAS_KERNEL_HD
        size_t get(int dim) const
        {
            return dims_[dim];
        }

        PARAS_KERNEL_HD
        size_t &get(int dim)
        {
            return dims_[dim];
        }

        PARAS_KERNEL_HD
        friend bool operator==(const id &lhs, const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                if (lhs.dims_[i] != rhs.dims_[i]) return false;
            return true;
        }

        PARAS_KERNEL_HD
        friend bool operator!=(const id &lhs, const id &rhs)
        {
            return !(lhs == rhs);
        }

        template <typename ScalarT,
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>>
        PARAS_KERNEL_HD
        friend bool operator==(const id &lhs, ScalarT rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                if (lhs.dims_[i] != static_cast<size_t>(rhs)) return false;
            return true;
        }
        template <typename ScalarT,
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>>
        PARAS_KERNEL_HD
        friend bool operator==(ScalarT lhs, const id &rhs)
        {
            return rhs == lhs;
        }
        template <typename ScalarT,
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>>
        PARAS_KERNEL_HD
        friend bool operator!=(const id &lhs, ScalarT rhs)
        {
            return !(lhs == rhs);
        }
        template <typename ScalarT,
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>>
        PARAS_KERNEL_HD
        friend bool operator!=(ScalarT lhs, const id &rhs)
        {
            return !(rhs == lhs);
        }

#define PARAS_SYCL_ID_ID_RELATIONAL_OP(OP)                     \
        PARAS_KERNEL_HD                                        \
        friend id operator OP(const id &lhs, const id &rhs)    \
        {                                                      \
            id result{};                                       \
            for (int i = 0; i < Dimensions; ++i)                \
                result[i] = (lhs.dims_[i] OP rhs.dims_[i]) ? 1 : 0; \
            return result;                                       \
        }

        PARAS_SYCL_ID_ID_RELATIONAL_OP(<)
        PARAS_SYCL_ID_ID_RELATIONAL_OP(>)
        PARAS_SYCL_ID_ID_RELATIONAL_OP(<=)
        PARAS_SYCL_ID_ID_RELATIONAL_OP(>=)

#undef PARAS_SYCL_ID_ID_RELATIONAL_OP

#define PARAS_SYCL_ID_ID_LOGICAL_OP(OP)                        \
        PARAS_KERNEL_HD                                        \
        friend id operator OP(const id &lhs, const id &rhs)    \
        {                                                      \
            id result{};                                       \
            for (int i = 0; i < Dimensions; ++i)                \
                result[i] = (lhs.dims_[i] OP rhs.dims_[i]) ? 1 : 0; \
            return result;                                       \
        }

        PARAS_SYCL_ID_ID_LOGICAL_OP(&&)
        PARAS_SYCL_ID_ID_LOGICAL_OP(||)

#undef PARAS_SYCL_ID_ID_LOGICAL_OP

#define PARAS_SYCL_ID_SCALAR_OP(OP)                                    \
        template <typename ScalarT,                                    \
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>> \
        PARAS_KERNEL_HD                                                \
        friend id operator OP(const id &lhs, ScalarT rhs)              \
        {                                                               \
            id result{};                                                \
            for (int i = 0; i < Dimensions; ++i)                        \
                result[i] = (lhs.dims_[i] OP static_cast<size_t>(rhs)) ? 1 : 0; \
            return result;                                              \
        }                                                               \
        template <typename ScalarT,                                    \
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>> \
        PARAS_KERNEL_HD                                                \
        friend id operator OP(ScalarT lhs, const id &rhs)              \
        {                                                               \
            id result{};                                                \
            for (int i = 0; i < Dimensions; ++i)                        \
                result[i] = (static_cast<size_t>(lhs) OP rhs.dims_[i]) ? 1 : 0; \
            return result;                                              \
        }

        PARAS_SYCL_ID_SCALAR_OP(&&)
        PARAS_SYCL_ID_SCALAR_OP(||)
        PARAS_SYCL_ID_SCALAR_OP(<)
        PARAS_SYCL_ID_SCALAR_OP(>)
        PARAS_SYCL_ID_SCALAR_OP(<=)
        PARAS_SYCL_ID_SCALAR_OP(>=)

#undef PARAS_SYCL_ID_SCALAR_OP

        PARAS_KERNEL_HD
        id operator+() const { return *this; }
        PARAS_KERNEL_HD
        id operator-() const
        {
            id result{};
            for (int i = 0; i < Dimensions; ++i)
                result[i] = -dims_[i];
            return result;
        }

        PARAS_KERNEL_HD
        id &operator+=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] += rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator-=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] -= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator++()
        {
            for (int i = 0; i < Dimensions; ++i)
                ++dims_[i];
            return *this;
        }
        PARAS_KERNEL_HD
        id operator++(int)
        {
            id result(*this);
            ++(*this);
            return result;
        }
        PARAS_KERNEL_HD
        id &operator--()
        {
            for (int i = 0; i < Dimensions; ++i)
                --dims_[i];
            return *this;
        }
        PARAS_KERNEL_HD
        id operator--(int)
        {
            id result(*this);
            --(*this);
            return result;
        }

        PARAS_KERNEL_HD
        id &operator*=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] *= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator/=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] /= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator%=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] %= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator<<=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] <<= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator>>=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] >>= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator&=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] &= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator|=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] |= rhs.dims_[i];
            return *this;
        }

        PARAS_KERNEL_HD
        id &operator^=(const id &rhs)
        {
            for (int i = 0; i < Dimensions; ++i)
                dims_[i] ^= rhs.dims_[i];
            return *this;
        }

#define PARAS_SYCL_ID_SCALAR_COMPOUND_OP(OP)                           \
        template <typename ScalarT,                                    \
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>> \
        PARAS_KERNEL_HD                                                \
        id &operator OP(ScalarT rhs)                                   \
        {                                                               \
            for (int i = 0; i < Dimensions; ++i)                        \
                dims_[i] OP static_cast<size_t>(rhs);                   \
            return *this;                                               \
        }

        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(+=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(-=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(*=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(/=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(%=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(<<=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(>>=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(&=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(|=)
        PARAS_SYCL_ID_SCALAR_COMPOUND_OP(^=)

#undef PARAS_SYCL_ID_SCALAR_COMPOUND_OP

#define PARAS_SYCL_ID_BINARY_OP(OP)                    \
        PARAS_KERNEL_HD                                \
        id operator OP(const id &rhs) const noexcept { \
            id result(*this);                          \
            result OP##= rhs;                          \
            return result;                              \
        }                                                \
        template <typename ScalarT,                                    \
                  typename = std::enable_if_t<std::is_integral_v<ScalarT>>> \
        PARAS_KERNEL_HD                                                  \
        id operator OP(ScalarT scalar) const noexcept {                   \
            id result(*this);                                              \
            for (int i = 0; i < Dimensions; ++i)                             \
                result.dims_[i] = result.dims_[i] OP static_cast<size_t>(scalar); \
            return result;                                                    \
        }

        PARAS_SYCL_ID_BINARY_OP(+)
        PARAS_SYCL_ID_BINARY_OP(-)
        PARAS_SYCL_ID_BINARY_OP(*)
        PARAS_SYCL_ID_BINARY_OP(/)
        PARAS_SYCL_ID_BINARY_OP(%)
        PARAS_SYCL_ID_BINARY_OP(<<)
        PARAS_SYCL_ID_BINARY_OP(>>)
        PARAS_SYCL_ID_BINARY_OP(&)
        PARAS_SYCL_ID_BINARY_OP(|)
        PARAS_SYCL_ID_BINARY_OP(^)

#undef PARAS_SYCL_ID_BINARY_OP
    };

#define PARAS_SYCL_ID_SCALAR_FIRST_OP(OP)                                 \
    template <typename ScalarT, int Dimensions,                          \
              typename = std::enable_if_t<std::is_integral_v<ScalarT>>>   \
    PARAS_KERNEL_HD                                                      \
    inline id<Dimensions> operator OP(ScalarT scalar, const id<Dimensions> &rhs) { \
        id<Dimensions> result{};                                          \
        for (int i = 0; i < Dimensions; ++i)                               \
            result[i] = static_cast<size_t>(scalar) OP rhs[i];              \
        return result;                                                       \
    }

    PARAS_SYCL_ID_SCALAR_FIRST_OP(+)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(-)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(*)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(/)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(%)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(<<)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(>>)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(&)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(|)
    PARAS_SYCL_ID_SCALAR_FIRST_OP(^)

#undef PARAS_SYCL_ID_SCALAR_FIRST_OP

    id(size_t) -> id<1>;
    id(size_t, size_t) -> id<2>;
    id(size_t, size_t, size_t) -> id<3>;

}

#endif
