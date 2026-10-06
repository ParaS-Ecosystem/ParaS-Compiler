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

#ifndef __PARAS_SUB_GROUP_HPP__
#define __PARAS_SUB_GROUP_HPP__

#include "kem_gpu/gpu_utilities.hpp"

#include <cstdint>
#include <type_traits>
#include "math.hpp"
#include "atomic_ref.hpp"
#include "sycl/id.hpp"
#include "sycl/range.hpp"
#include "work_group_algorithms_detail.hpp"

namespace sycl {

/* ================= sub_group ================= */

class sub_group {
public:
    using id_type = id<1>;
    using range_type = range<1>;
    using linear_id_type = uint32_t;

    static constexpr int dimensions = 1;
    static constexpr memory_scope fence_scope = memory_scope::sub_group;

#if PARAS_GPU_BACKEND

    static constexpr linear_id_type warp_size = PARAS_WARP_SIZE;

    PARAS_KERNEL_D
    static linear_id_type paras_block_linear_id() {
        return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    }

    PARAS_KERNEL_D
    static linear_id_type paras_block_size() { return blockDim.x * blockDim.y * blockDim.z; }

    PARAS_KERNEL_D
    linear_id_type get_local_linear_id() const { return paras_block_linear_id() & (warp_size - 1); }

    PARAS_KERNEL_D
    linear_id_type get_local_linear_range() const {
        const linear_id_type base = paras_block_linear_id() & ~(warp_size - 1);
        const linear_id_type left = paras_block_size() - base;
        return left < warp_size ? left : warp_size;
    }

    PARAS_KERNEL_D
    linear_id_type get_group_linear_id() const { return paras_block_linear_id() / warp_size; }

    PARAS_KERNEL_D
    linear_id_type get_group_linear_range() const {
        return (paras_block_size() + warp_size - 1) / warp_size;
    }
#else
    PARAS_KERNEL_HD
    linear_id_type get_local_linear_id() const { return 0; }

    PARAS_KERNEL_HD
    linear_id_type get_local_linear_range() const { return 1; }

    PARAS_KERNEL_HD
    linear_id_type get_group_linear_id() const { return detail::wg_local_linear_id(); }

    PARAS_KERNEL_HD
    linear_id_type get_group_linear_range() const { return paras_host_detail::tl_local_size; }
#endif

    PARAS_KERNEL_HD
    id_type get_local_id() const { return id_type(get_local_linear_id()); }

    PARAS_KERNEL_HD
    range_type get_local_range() const { return range_type(get_local_linear_range()); }

    PARAS_KERNEL_HD
    id_type get_group_id() const { return id_type(get_group_linear_id()); }

    PARAS_KERNEL_HD
    range_type get_group_range() const { return range_type(get_group_linear_range()); }

    PARAS_KERNEL_HD
    range_type get_max_local_range() const {
#if PARAS_GPU_BACKEND
        return range_type(warp_size);
#else
        return range_type(1);
#endif
    }

    PARAS_KERNEL_HD
    bool leader() const { return get_local_linear_id() == 0; }
};

/* ================= subgroup algorithms ================= */

#if PARAS_GPU_BACKEND

namespace detail {

template <typename T, typename BinaryOperation>
PARAS_KERNEL_D inline T sub_group_reduce(const sub_group& g, T value, BinaryOperation binary_op) {
    const std::uint64_t mask = paras_active_mask();
    const auto lane = g.get_local_linear_id();
    const auto size = g.get_local_linear_range();
    for (unsigned offset = sub_group::warp_size / 2; offset > 0; offset >>= 1) {
        T other = paras_shfl_down(mask, value, static_cast<int>(offset));
        if (lane + offset < size)
            value = binary_op(value, other);
    }
    return paras_shfl(mask, value, 0);
}
} // namespace detail

template <typename Group>
PARAS_KERNEL_HD inline bool any_of_group(Group g, bool pred) {
    if constexpr (std::is_same_v<Group, sub_group>)
        return paras_any_sync(paras_active_mask(), pred);
    else
        return detail::wg_any(g, pred);
}

template <typename Group, typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T reduce_over_group(Group g, T value, BinaryOperation binary_op) {
    if constexpr (std::is_same_v<Group, sub_group>) {
        return detail::sub_group_reduce(g, value, binary_op);
    } else {
        return detail::wg_reduce(g, value, binary_op);
    }
}

template <typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T reduce_over_group(sub_group g, T value, BinaryOperation binary_op) {
#if PARAS_GPU_BACKEND
    return detail::sub_group_reduce(g, value, binary_op);
#else
    return value;
#endif
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T select_from_group(Group g, T value,
                                           typename Group::id_type remote_local_id) {
    if constexpr (std::is_same_v<Group, sub_group>) {
        return paras_shfl(paras_active_mask(), value, remote_local_id[0]);
    } else {
        return value;
    }
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T select_from_group(Group g, T value, uint32_t remote_lane) {
    return select_from_group(g, value, typename Group::id_type(remote_lane));
}

#else

template <typename Group>
PARAS_KERNEL_HD inline bool any_of_group(Group g, bool pred) {
    if constexpr (std::is_same_v<Group, sub_group>)
        return pred;
    else
        return detail::wg_any(g, pred);
}

template <typename Group, typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T reduce_over_group(Group g, T value, BinaryOperation binary_op) {
    if constexpr (std::is_same_v<Group, sub_group>)
        return value;
    else
        return detail::wg_reduce(g, value, binary_op);
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T select_from_group(Group, T value, typename Group::id_type) {
    return value;
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T select_from_group(Group, T value, uint32_t) {
    return value;
}

#endif

template <typename Group, typename V, typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T reduce_over_group(Group g, V x, T init, BinaryOperation binary_op) {
    return binary_op(init, reduce_over_group(g, T(x), binary_op));
}

} // namespace sycl

#endif
