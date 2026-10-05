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

#ifndef __PARAS_ITEM_HPP__
#define __PARAS_ITEM_HPP__

#include <type_traits>
#include "id.hpp"
#include "range.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <int Dims = 1, bool WithOffset = true>
class item {
    static_assert(Dims >= 1 && Dims <= 3, "item supports 1, 2 or 3 dimensions");
    id<Dims> idx_;
    range<Dims> range_;

public:
    static constexpr int dimensions = Dims;

    PARAS_KERNEL_HD
    item(const id<Dims>& i, const range<Dims>& r) : idx_(i), range_(r) {}

    PARAS_KERNEL_HD
    item(const id<Dims>& i) : idx_(i) {
        for (int d = 0; d < Dims; ++d)
            range_[d] = i[d] + 1;
    }
    template <int D = Dims, typename std::enable_if_t<D == 1, int> = 0>
    PARAS_KERNEL_HD item(size_t i) : item(id<1>(i)) {}
    template <int D = Dims, typename std::enable_if_t<D == 2, int> = 0>
    PARAS_KERNEL_HD item(size_t i, size_t j) : item(id<2>(i, j)) {}
    template <int D = Dims, typename std::enable_if_t<D == 3, int> = 0>
    PARAS_KERNEL_HD item(size_t i, size_t j, size_t k) : item(id<3>(i, j, k)) {}

    PARAS_KERNEL_HD id<Dims> get_id() const { return idx_; }
    PARAS_KERNEL_HD size_t get_id(int d) const { return idx_[d]; }
    PARAS_KERNEL_HD size_t operator[](int d) const { return idx_[d]; }
    PARAS_KERNEL_HD size_t& operator[](int d) { return idx_[d]; }

    PARAS_KERNEL_HD range<Dims> get_range() const { return range_; }
    PARAS_KERNEL_HD size_t get_range(int d) const { return range_[d]; }

    PARAS_KERNEL_HD id<Dims> get_offset() const { return id<Dims>(); }

    PARAS_KERNEL_HD
    size_t get_linear_id() const {
        size_t lin = 0;
        for (int d = 0; d < Dims; ++d)
            lin = lin * range_[d] + idx_[d];
        return lin;
    }

    PARAS_KERNEL_HD operator id<Dims>() const { return idx_; }

    template <int D = Dims, typename std::enable_if_t<D == 1, int> = 0>
    PARAS_KERNEL_HD operator size_t() const {
        return idx_[0];
    }

    PARAS_KERNEL_HD
    friend bool operator==(const item& a, const item& b) {
        for (int d = 0; d < Dims; ++d)
            if (a.idx_[d] != b.idx_[d] || a.range_[d] != b.range_[d])
                return false;
        return true;
    }
    PARAS_KERNEL_HD
    friend bool operator!=(const item& a, const item& b) { return !(a == b); }
};

namespace detail {
template <int D, typename Func>
PARAS_KERNEL_HD inline void invoke_range_kernel(const Func& f, const id<D>& idx,
                                                const range<D>& r) {
    if constexpr (std::is_invocable_v<const Func&, item<D>>)
        f(item<D>(idx, r));
    else
        f(idx);
}
} // namespace detail

} // namespace sycl

#endif
