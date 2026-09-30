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
#include <type_traits>
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename Group, typename T>
PARAS_KERNEL_HD inline T
shift_group_left(Group g, T x, typename Group::linear_id_type delta = 1) {
#if PARAS_GPU_BACKEND

  if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>) {
    return paras_shfl_down(0xffffffff, x, delta);
  } else {
    return x;
  }

#else
  return x;
#endif
}

template <typename Group, typename T>
PARAS_KERNEL_HD inline T
shift_group_right(Group g, T x, typename Group::linear_id_type delta = 1) {
#if PARAS_GPU_BACKEND

  if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>) {
    return paras_shfl(0xffffffff, x, g.get_local_linear_id() - delta);
  } else {
    return x;
  }

#else
  return x;
#endif
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation>
PARAS_KERNEL_HD OutPtr joint_exclusive_scan(Group, InPtr first, InPtr last,
                                            OutPtr result, T init,
                                            BinaryOperation binary_op) {
#if PARAS_GPU_BACKEND

  const int tid = threadIdx.x;
  const int N = last - first;

  if (tid == 0) {
    T running = init;

    for (int i = 0; i < N; ++i) {
      result[i] = running;
      running = binary_op(running, first[i]);
    }
  }

  paras_syncthreads();

  return result + N;

#else

  T running = init;

  for (InPtr it = first; it != last; ++it) {
    *result++ = running;
    running = binary_op(running, *it);
  }

  return result;

#endif
}

} // namespace sycl

#endif
