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
#include "known_identity.hpp"

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

namespace detail {

template <typename Group>
PARAS_KERNEL_HD inline std::size_t wg_team_size(const Group &g) {
#if PARAS_DEVICE_PASS
    const std::size_t launched = static_cast<std::size_t>(blockDim.x) *
                                 static_cast<std::size_t>(blockDim.y) *
                                 static_cast<std::size_t>(blockDim.z);
    return launched == g.get_local_linear_range() ? launched : 1;
#else
    return paras_host_detail::tl_barrier ? g.get_local_linear_range() : 1;
#endif
}

}

template <typename Group, typename InPtr, typename OutPtr,
          typename T, typename BinaryOperation>
PARAS_KERNEL_HD
OutPtr joint_exclusive_scan(Group g,
                            InPtr first,
                            InPtr last,
                            OutPtr result,
                            T init,
                            BinaryOperation binary_op)
{
    const std::size_t n = static_cast<std::size_t>(last - first);

    if constexpr (!std::is_same_v<std::decay_t<Group>, sub_group> &&
                  sizeof(T) * detail::kMaxWorkGroupSize <= 32768)
    {
        const std::size_t team = detail::wg_team_size(g);
        if (team > 1)
        {
            const std::size_t lid   = detail::wg_local_linear_id();
            const std::size_t piece = (n + team - 1) / team;
            const std::size_t begin = (lid * piece < n) ? lid * piece : n;
            const std::size_t end   = (begin + piece < n) ? begin + piece : n;

            T total = init;
            for (std::size_t i = begin; i < end; ++i)
            {
                const T v = first[i];
                total = (i == begin) ? v : binary_op(total, v);
            }

            T running = detail::wg_exclusive_scan(g, total, init, binary_op);

            for (std::size_t i = begin; i < end; ++i)
            {
                const T v = first[i];
                result[i] = running;
                running   = binary_op(running, v);
            }

            paras_syncthreads();
            return result + n;
        }
    }

    #if PARAS_GPU_BACKEND

        if (detail::wg_local_linear_id() == 0){
            T running = init;
            for (std::size_t i = 0; i < n; ++i){
                const T v = first[i];
                result[i] = running;
                running   = binary_op(running, v);
            }
        }

        paras_syncthreads();

        return result + n;

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

    if constexpr (has_known_identity_v<BinaryOperation, T>)
        return exclusive_scan_over_group(g, x, known_identity_v<BinaryOperation, T>, binary_op);
    else
        return exclusive_scan_over_group(g, x, T{}, binary_op);
}

template <typename Group, typename Ptr, typename BinaryOperation>
PARAS_KERNEL_HD inline auto joint_reduce(Group, Ptr first, Ptr last,
                                         BinaryOperation binary_op)
{
    using T = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
    T acc = *first;
    for (Ptr it = first + 1; it < last; ++it)
        acc = binary_op(acc, *it);
    return acc;
}

template <typename Group, typename Ptr, typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T joint_reduce(Group, Ptr first, Ptr last, T init,
                                      BinaryOperation binary_op)
{
    T acc = init;
    for (Ptr it = first; it < last; ++it)
        acc = binary_op(acc, *it);
    return acc;
}

template <typename Group, typename Ptr, typename Predicate>
PARAS_KERNEL_HD inline bool joint_any_of(Group, Ptr first, Ptr last, Predicate pred)
{
    for (Ptr it = first; it < last; ++it)
        if (pred(*it)) return true;
    return false;
}
template <typename Group, typename Ptr, typename Predicate>
PARAS_KERNEL_HD inline bool joint_all_of(Group, Ptr first, Ptr last, Predicate pred)
{
    for (Ptr it = first; it < last; ++it)
        if (!pred(*it)) return false;
    return true;
}
template <typename Group, typename Ptr, typename Predicate>
PARAS_KERNEL_HD inline bool joint_none_of(Group g, Ptr first, Ptr last, Predicate pred)
{
    return !joint_any_of(g, first, last, pred);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation>
PARAS_KERNEL_HD inline OutPtr joint_inclusive_scan(Group, InPtr first, InPtr last,
                                                   OutPtr result,
                                                   BinaryOperation binary_op, T init)
{
    T acc = init;
    for (InPtr it = first; it < last; ++it, ++result) {
        acc = binary_op(acc, *it);
        *result = acc;
    }
    return result;
}

template <typename Group, typename InPtr, typename OutPtr, typename BinaryOperation>
PARAS_KERNEL_HD inline OutPtr joint_inclusive_scan(Group, InPtr first, InPtr last,
                                                   OutPtr result,
                                                   BinaryOperation binary_op)
{
    using T = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
    if (!(first < last)) return result;
    T acc = *first;
    *result = acc;
    ++result;
    for (InPtr it = first + 1; it < last; ++it, ++result) {
        acc = binary_op(acc, *it);
        *result = acc;
    }
    return result;
}

template <typename Group, typename InPtr, typename OutPtr, typename BinaryOperation,
          typename = std::enable_if_t<!std::is_pointer_v<BinaryOperation> &&
                                      std::is_invocable_v<BinaryOperation,
                                                          decltype(*std::declval<InPtr>()),
                                                          decltype(*std::declval<InPtr>())>>>
PARAS_KERNEL_HD inline OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last,
                                                   OutPtr result,
                                                   BinaryOperation binary_op)
{
    using In = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
    using T = detail::op_value_t<BinaryOperation, In>;
    return joint_exclusive_scan(g, first, last, result,
                                known_identity_v<BinaryOperation, T>, binary_op);
}

template <typename Group, typename V, typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op,
                                                   T init)
{
    return binary_op(init, static_cast<T>(inclusive_scan_over_group(g, static_cast<T>(x), binary_op)));
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

}

#endif
