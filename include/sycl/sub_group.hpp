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

#include "atomic_ref.hpp"
#include "math.hpp"
#include "sycl/id.hpp"
#include "sycl/range.hpp"
#include <cstdint>
#include <type_traits>

namespace sycl {

class sub_group {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = uint32_t;

  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

#if PARAS_GPU_BACKEND
  static constexpr linear_id_type warp_size = 32;

  PARAS_KERNEL_D
  linear_id_type get_local_linear_id() const {
    unsigned linear_id = threadIdx.x + threadIdx.y * blockDim.x +
                         threadIdx.z * blockDim.x * blockDim.y;

    return linear_id & (warp_size - 1);
  }

  PARAS_KERNEL_D
  linear_id_type get_local_linear_range() const { return warp_size; }
#else
  PARAS_KERNEL_HD
  linear_id_type get_local_linear_id() const { return 0; }

  PARAS_KERNEL_HD
  linear_id_type get_local_linear_range() const { return 1; }
#endif

  PARAS_KERNEL_HD
  id_type get_local_id() const { return id_type(get_local_linear_id()); }

  PARAS_KERNEL_HD
  range_type get_local_range() const {
    return range_type(get_local_linear_range());
  }

  PARAS_KERNEL_HD
  range_type get_max_local_range() const { return get_local_range(); }

  PARAS_KERNEL_HD
  bool leader() const { return get_local_linear_id() == 0; }
};

#if PARAS_GPU_BACKEND

template <typename Group>
PARAS_KERNEL_HD inline bool any_of_group(Group g, bool pred) {
  if constexpr (std::is_same_v<Group, sub_group>)
    return paras_any_sync(paras_active_mask(), pred);
  else
    return pred;
}

template <typename Group, typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T reduce_over_group(Group g, T value,
                                           BinaryOperation binary_op) {
  if constexpr (std::is_same_v<Group, sub_group>) {
    const unsigned activeMask = paras_active_mask();

    for (int offset = g.get_local_linear_range() / 2; offset > 0;
         offset >>= 1) {
      T other = paras_shfl_down(activeMask, value, offset);
      value = binary_op(value, other);
    }

    return paras_shfl(activeMask, value, 0);
  } else {
    return value;
  }
}

template <typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T reduce_over_group(sub_group g, T value,
                                           BinaryOperation binary_op) {
#if PARAS_GPU_BACKEND
  const unsigned activeMask = paras_active_mask();

  for (int offset = g.get_local_linear_range() / 2; offset > 0; offset >>= 1) {
    T other = paras_shfl_down(activeMask, value, offset);
    value = binary_op(value, other);
  }

  return paras_shfl(activeMask, value, 0);
#else
  return value;
#endif
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T
select_from_group(Group g, T value, typename Group::id_type remote_local_id) {
  if constexpr (std::is_same_v<Group, sub_group>) {
    return paras_shfl(paras_active_mask(), value, remote_local_id[0]);
  } else {
    return value;
  }
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T select_from_group(Group g, T value,
                                           uint32_t remote_lane) {
  return select_from_group(g, value, typename Group::id_type(remote_lane));
}

#else

template <typename Group>
PARAS_KERNEL_HD inline bool any_of_group(Group, bool pred) {
  return pred;
}

template <typename Group, typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline T reduce_over_group(Group, T value, BinaryOperation) {
  return value;
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T select_from_group(Group, T value,
                                           typename Group::id_type) {
  return value;
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T select_from_group(Group, T value, uint32_t) {
  return value;
}

#endif

} // namespace sycl

#endif
