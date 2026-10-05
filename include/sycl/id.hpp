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
#include "kem_gpu/gpu_utilities.hpp"
#include "range.hpp"

namespace sycl {

template <int Dimensions = 1>
class id {
public:
    static constexpr int dimensions = Dimensions;
    size_t dims_[Dimensions];
    PARAS_KERNEL_HD
    id() {
        for (int i = 0; i < Dimensions; i++)
            dims_[i] = 0;
    }

    PARAS_KERNEL_HD
    id(size_t i0) {
        static_assert(Dimensions == 1, "id(size_t) is only valid for id<1>");
        dims_[0] = i0;
    }

    PARAS_KERNEL_HD
    id(size_t i0, size_t i1) {
        static_assert(Dimensions == 2, "id(size_t,size_t) is only valid for id<2>");
        dims_[0] = i0;
        dims_[1] = i1;
    }

    PARAS_KERNEL_HD
    id(size_t i0, size_t i1, size_t i2) {
        static_assert(Dimensions == 3, "id(size_t,size_t,size_t) is only valid for id<3>");
        dims_[0] = i0;
        dims_[1] = i1;
        dims_[2] = i2;
    }

    PARAS_KERNEL_HD
    explicit id(const range<Dimensions>& r) {
        for (int i = 0; i < Dimensions; i++)
            dims_[i] = r[i];
    }

    PARAS_KERNEL_HD
    size_t& operator[](int dim) { return dims_[dim]; }

    PARAS_KERNEL_HD
    size_t operator[](int dim) const { return dims_[dim]; }

    PARAS_KERNEL_HD
    size_t get(int dim) const { return dims_[dim]; }

    PARAS_KERNEL_HD
    size_t& get(int dim) { return dims_[dim]; }

    /** Implicit id<1> -> size_t (SYCL 2020). Kept as a plain (non-template)
     *  conversion so `ptr[id]` keeps working; the operator== below is an
     *  exact match, so `id<2> == id<2>` no longer falls back to this. */
    PARAS_KERNEL_HD
    operator size_t() const {
        static_assert(Dimensions == 1, "Implicit conversion only valid for id<1>");
        return dims_[0];
    }

    PARAS_KERNEL_HD
    friend bool operator==(const id& a, const id& b) {
        for (int d = 0; d < Dimensions; ++d)
            if (a.dims_[d] != b.dims_[d])
                return false;
        return true;
    }

    PARAS_KERNEL_HD
    friend bool operator!=(const id& a, const id& b) { return !(a == b); }

    PARAS_INDEX_OPERATORS(id, Dimensions)
};

id(size_t) -> id<1>;
id(size_t, size_t) -> id<2>;
id(size_t, size_t, size_t) -> id<3>;

} // namespace sycl

#endif
