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

#ifndef __PARAS_INDEX_OPS_HPP__
#define __PARAS_INDEX_OPS_HPP__

#include <cstddef>
#include <type_traits>

#define PARAS_INDEX_SCALAR template <typename S, std::enable_if_t<std::is_integral_v<S>, int> = 0>

#define PARAS_INDEX_BINARY_OP(T, D, OP)                                                            \
    PARAS_KERNEL_HD friend T operator OP(const T& lhs, const T& rhs) {                             \
        T r;                                                                                       \
        for (int i = 0; i < D; ++i)                                                                \
            r[i] = lhs[i] OP rhs[i];                                                               \
        return r;                                                                                  \
    }                                                                                              \
    PARAS_INDEX_SCALAR                                                                             \
    PARAS_KERNEL_HD friend T operator OP(const T& lhs, const S& rhs) {                             \
        T r;                                                                                       \
        for (int i = 0; i < D; ++i)                                                                \
            r[i] = lhs[i] OP rhs;                                                                  \
        return r;                                                                                  \
    }                                                                                              \
    PARAS_INDEX_SCALAR                                                                             \
    PARAS_KERNEL_HD friend T operator OP(const S& lhs, const T& rhs) {                             \
        T r;                                                                                       \
        for (int i = 0; i < D; ++i)                                                                \
            r[i] = lhs OP rhs[i];                                                                  \
        return r;                                                                                  \
    }

#define PARAS_INDEX_COMPOUND_OP(T, D, OP)                                                          \
    PARAS_KERNEL_HD friend T& operator OP(T& lhs, const T& rhs) {                                  \
        for (int i = 0; i < D; ++i)                                                                \
            lhs[i] OP rhs[i];                                                                      \
        return lhs;                                                                                \
    }                                                                                              \
    PARAS_INDEX_SCALAR                                                                             \
    PARAS_KERNEL_HD friend T& operator OP(T& lhs, const S& rhs) {                                  \
        for (int i = 0; i < D; ++i)                                                                \
            lhs[i] OP rhs;                                                                         \
        return lhs;                                                                                \
    }

#define PARAS_INDEX_OPERATORS(T, D)                                                                \
    PARAS_INDEX_BINARY_OP(T, D, +)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, -)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, *)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, /)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, %)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, <<)                                                                \
    PARAS_INDEX_BINARY_OP(T, D, >>)                                                                \
    PARAS_INDEX_BINARY_OP(T, D, &)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, |)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, ^)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, &&)                                                                \
    PARAS_INDEX_BINARY_OP(T, D, ||)                                                                \
    PARAS_INDEX_BINARY_OP(T, D, <)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, >)                                                                 \
    PARAS_INDEX_BINARY_OP(T, D, <=)                                                                \
    PARAS_INDEX_BINARY_OP(T, D, >=)                                                                \
    PARAS_INDEX_COMPOUND_OP(T, D, +=)                                                              \
    PARAS_INDEX_COMPOUND_OP(T, D, -=)                                                              \
    PARAS_INDEX_COMPOUND_OP(T, D, *=)                                                              \
    PARAS_INDEX_COMPOUND_OP(T, D, /=)                                                              \
    PARAS_INDEX_COMPOUND_OP(T, D, %=)                                                              \
    PARAS_INDEX_COMPOUND_OP(T, D, <<=)                                                             \
    PARAS_INDEX_COMPOUND_OP(T, D, >>=)                                                             \
    PARAS_INDEX_COMPOUND_OP(T, D, &=)                                                              \
    PARAS_INDEX_COMPOUND_OP(T, D, |=)                                                              \
    PARAS_INDEX_COMPOUND_OP(T, D, ^=)                                                              \
    PARAS_KERNEL_HD friend T operator+(const T& v) {                                               \
        return v;                                                                                  \
    }                                                                                              \
    PARAS_KERNEL_HD friend T operator-(const T& v) {                                               \
        T r;                                                                                       \
        for (int i = 0; i < D; ++i)                                                                \
            r[i] = -v[i];                                                                          \
        return r;                                                                                  \
    }                                                                                              \
    PARAS_KERNEL_HD friend T& operator++(T& v) {                                                   \
        for (int i = 0; i < D; ++i)                                                                \
            ++v[i];                                                                                \
        return v;                                                                                  \
    }                                                                                              \
    PARAS_KERNEL_HD friend T operator++(T& v, int) {                                               \
        T old = v;                                                                                 \
        ++v;                                                                                       \
        return old;                                                                                \
    }                                                                                              \
    PARAS_KERNEL_HD friend T& operator--(T& v) {                                                   \
        for (int i = 0; i < D; ++i)                                                                \
            --v[i];                                                                                \
        return v;                                                                                  \
    }                                                                                              \
    PARAS_KERNEL_HD friend T operator--(T& v, int) {                                               \
        T old = v;                                                                                 \
        --v;                                                                                       \
        return old;                                                                                \
    }

#endif
