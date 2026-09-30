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

#ifndef __PARAS_GROUP_ALGORITHMS_HPP__
#define __PARAS_GROUP_ALGORITHMS_HPP__

#include "sub_group.hpp"
#include "kem_gpu/gpu_utilities.hpp"
#include <type_traits>
#include "work_group_algorithms_detail.hpp"

namespace sycl {

template <typename Group, typename T>
PARAS_KERNEL_HD
inline T shift_group_left(Group g,
                          T x,
                          typename Group::linear_id_type delta = 1)
{
#if PARAS_GPU_BACKEND

    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
    {

        return paras_shfl_down(paras_active_mask(), x, delta);
    }
    else
    {
        return x;
    }

#else
    return x;
#endif
}

template <typename Group, typename T>
PARAS_KERNEL_HD
inline T shift_group_right(Group g,
                           T x,
                           typename Group::linear_id_type delta = 1)
{
#if PARAS_GPU_BACKEND

    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
    {

        return paras_shfl(paras_active_mask(), x, g.get_local_linear_id() - delta);
    }
    else
    {
        return x;
    }

#else
    return x;
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          typename T, typename BinaryOperation>
PARAS_KERNEL_HD
OutPtr joint_exclusive_scan(Group,
                            InPtr first,
                            InPtr last,
                            OutPtr result,
                            T init,
                            BinaryOperation binary_op)
{
    #if PARAS_GPU_BACKEND

        const int tid = threadIdx.x;
        const int N   = last - first;

        if (tid == 0){
            T running = init;
            for (int i = 0; i < N; ++i){
                result[i] = running;
                running   = binary_op(running, first[i]);
            }
        }

        paras_syncthreads();

        return result + N;

    #else

        T running = init;

        for (InPtr it = first; it != last; ++it)
        {
            *result++ = running;
            running   = binary_op(running, *it);
        }

        return result;

    #endif
}

template <typename Group, typename T>
PARAS_KERNEL_HD
inline T group_broadcast(Group g, T x, typename Group::id_type local_id)
{
#if PARAS_GPU_BACKEND
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return paras_shfl(paras_active_mask(), x, static_cast<int>(local_id[0]));
    else
        return detail::wg_broadcast(g, x, detail::wg_linearize(g, local_id));
#else
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return x;
    else
        return detail::wg_broadcast(g, x, detail::wg_linearize(g, local_id));
#endif
}

template <typename Group, typename T>
PARAS_KERNEL_HD
inline T group_broadcast(Group g, T x, typename Group::linear_id_type linear_local_id)
{
#if PARAS_GPU_BACKEND
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return paras_shfl(paras_active_mask(), x, static_cast<int>(linear_local_id));
    else
        return detail::wg_broadcast(g, x, linear_local_id);
#else
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return x;
    else
        return detail::wg_broadcast(g, x, linear_local_id);
#endif
}

template <typename Group, typename T>
PARAS_KERNEL_HD
inline T group_broadcast(Group g, T x)
{
    return group_broadcast(g, x, typename Group::linear_id_type(0));
}

template <typename Group>
PARAS_KERNEL_HD
inline bool all_of_group(Group g, bool pred)
{
#if PARAS_GPU_BACKEND
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return !paras_any_sync(paras_active_mask(), !pred);
    else
        return detail::wg_all(g, pred);
#else
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return pred;
    else
        return detail::wg_all(g, pred);
#endif
}

template <typename Group, typename T, typename Predicate>
PARAS_KERNEL_HD
inline bool all_of_group(Group g, T x, Predicate pred)
{
    return all_of_group(g, pred(x));
}

template <typename Group>
PARAS_KERNEL_HD
inline bool none_of_group(Group g, bool pred)
{
    return !any_of_group(g, pred);
}

template <typename Group, typename T, typename Predicate>
PARAS_KERNEL_HD
inline bool none_of_group(Group g, T x, Predicate pred)
{
    return !any_of_group(g, pred(x));
}

template <typename Group, typename V, typename BinaryOperation>
PARAS_KERNEL_HD
inline V inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op)
{
#if PARAS_GPU_BACKEND
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
    {
        const std::uint64_t mask = paras_active_mask();
        const auto lane = g.get_local_linear_id();
        for (typename Group::linear_id_type offset = 1;
             offset < g.get_local_linear_range();
             offset <<= 1)
        {
            V other = paras_shfl_up(mask, x, static_cast<int>(offset));
            if (lane >= offset) x = binary_op(x, other);
        }
        return x;
    }
    else
    {
        return detail::wg_inclusive_scan(g, x, binary_op);
    }
#else
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return x;
    else
        return detail::wg_inclusive_scan(g, x, binary_op);
#endif
}

template <typename Group, typename T, typename BinaryOperation>
PARAS_KERNEL_HD
inline T exclusive_scan_over_group(Group g, T x, T init, BinaryOperation binary_op)
{
#if PARAS_GPU_BACKEND
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
    {
        T inclusive = inclusive_scan_over_group(g, x, binary_op);
        T shifted = paras_shfl_up(paras_active_mask(), inclusive, 1);

        return (g.get_local_linear_id() == 0) ? init : binary_op(init, shifted);
    }
    else
    {
        return detail::wg_exclusive_scan(g, x, init, binary_op);
    }
#else
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return init;
    else
        return detail::wg_exclusive_scan(g, x, init, binary_op);
#endif
}

template <typename Group, typename T, typename BinaryOperation>
PARAS_KERNEL_HD
inline T exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op)
{
    return exclusive_scan_over_group(g, x, T{}, binary_op);
}

template <typename Group, typename T>
PARAS_KERNEL_HD
inline T permute_group_by_xor(Group g, T x, typename Group::linear_id_type mask_xor)
{
#if PARAS_GPU_BACKEND
    if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>)
        return paras_shfl_xor(paras_active_mask(), x, static_cast<int>(mask_xor));
    else
        return x;
#else
    (void)g; (void)mask_xor;
    return x;
#endif
}

} // namespace sycl

#endif
