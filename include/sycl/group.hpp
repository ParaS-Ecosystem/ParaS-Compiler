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

#ifndef __PARAS_GROUP_HPP__
#define __PARAS_GROUP_HPP__

#include "id.hpp"
#include "range.hpp"
#include "h_item.hpp"
#include "atomic_ref.hpp"
#include "work_group_algorithms_detail.hpp"
#include "decorated_ptr_and_device_event.hpp"

#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <int Dimensions = 1>
class group {
public:
    using id_type = id<Dimensions>;
    using range_type = range<Dimensions>;
    using linear_id_type = size_t;

    static constexpr int dimensions = Dimensions;
    static constexpr memory_scope fence_scope = memory_scope::work_group;

private:
    id_type group_id_;
    range_type group_range_;
    range_type local_range_;

public:
    PARAS_KERNEL_HD
    group(id_type gid, range_type gr, range_type lr)
        : group_id_(gid), group_range_(gr), local_range_(lr) {}

    PARAS_KERNEL_HD
    id_type get_group_id() const { return group_id_; }
    PARAS_KERNEL_HD
    size_t get_group_id(int dim) const { return group_id_[dim]; }

    PARAS_KERNEL_HD
    id_type get_local_id() const {
        size_t lin = get_local_linear_id();
        id_type out;
        for (int d = Dimensions - 1; d >= 0; d--) {
            out[d] = lin % local_range_[d];
            lin /= local_range_[d];
        }
        return out;
    }
    PARAS_KERNEL_HD
    size_t get_local_id(int dim) const { return get_local_id()[dim]; }

    PARAS_KERNEL_HD
    range_type get_local_range() const { return local_range_; }
    PARAS_KERNEL_HD
    size_t get_local_range(int dim) const { return local_range_[dim]; }

    PARAS_KERNEL_HD
    range_type get_group_range() const { return group_range_; }
    PARAS_KERNEL_HD
    size_t get_group_range(int dim) const { return group_range_[dim]; }

    PARAS_KERNEL_HD
    range_type get_max_local_range() const { return local_range_; }

    PARAS_KERNEL_HD
    size_t operator[](int dim) const { return group_id_[dim]; }

    PARAS_KERNEL_HD
    size_t get_group_linear_id() const {
        size_t linear = 0, mul = 1;
        for (int d = Dimensions - 1; d >= 0; d--) {
            linear += group_id_[d] * mul;
            mul *= group_range_[d];
        }
        return linear;
    }

    PARAS_KERNEL_HD
    size_t get_local_linear_id() const { return detail::wg_local_linear_id(); }

    PARAS_KERNEL_HD
    size_t get_group_linear_range() const {
        size_t r = 1;
        for (int d = 0; d < Dimensions; d++)
            r *= group_range_[d];
        return r;
    }

    PARAS_KERNEL_HD
    size_t get_local_linear_range() const {
        size_t r = 1;
        for (int d = 0; d < Dimensions; d++)
            r *= local_range_[d];
        return r;
    }

    PARAS_KERNEL_HD
    bool leader() const { return get_local_linear_id() == 0; }

    template <typename Fn>
    PARAS_KERNEL_HD void parallel_for_work_item(const Fn& fn) const {
        parallel_for_work_item(local_range_, fn);
    }

    template <typename Fn>
    PARAS_KERNEL_HD void parallel_for_work_item(range<Dimensions> flexibleRange,
                                                const Fn& fn) const {
        range_type globalRange;
        std::size_t count = 1;
        for (int d = 0; d < Dimensions; ++d) {
            globalRange[d] = group_range_[d] * flexibleRange[d];
            count *= flexibleRange[d];
        }
        for (std::size_t lin = 0; lin < count; ++lin) {
            id_type localId;
            std::size_t rem = lin;
            for (int d = Dimensions - 1; d >= 0; --d) {
                localId[d] = rem % flexibleRange[d];
                rem /= flexibleRange[d];
            }
            id_type globalId;
            for (int d = 0; d < Dimensions; ++d)
                globalId[d] = group_id_[d] * flexibleRange[d] + localId[d];
            h_item<Dimensions> item(globalId, globalRange, localId, flexibleRange);
            fn(item);
        }
    }

    bool operator==(const group& rhs) const {
        for (int d = 0; d < Dimensions; ++d) {
            if (group_id_[d] != rhs.group_id_[d])
                return false;
            if (group_range_[d] != rhs.group_range_[d])
                return false;
            if (local_range_[d] != rhs.local_range_[d])
                return false;
        }
        return true;
    }

    bool operator!=(const group& rhs) const { return !(*this == rhs); }

    template <typename dataT>
    PARAS_KERNEL_HD device_event async_work_group_copy(decorated_local_ptr<dataT> dest,
                                                       decorated_global_ptr<dataT> src,
                                                       std::size_t numElements,
                                                       std::size_t srcStride = 1) const {
        for (std::size_t i = 0; i < numElements; ++i) {
            dest[i] = src[i * srcStride];
        }
        return device_event{};
    }

    template <typename dataT>
    PARAS_KERNEL_HD device_event async_work_group_copy(decorated_global_ptr<dataT> dest,
                                                       decorated_local_ptr<dataT> src,
                                                       std::size_t numElements,
                                                       std::size_t destStride = 1) const {
        for (std::size_t i = 0; i < numElements; ++i) {
            dest[i * destStride] = src[i];
        }
        return device_event{};
    }

    template <typename... EventTN>
    PARAS_KERNEL_HD void wait_for(EventTN... events) const {
        (events.wait(), ...);
    }
};

} // namespace sycl

#endif
