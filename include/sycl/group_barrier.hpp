/**<
 * Copyright 2025 The ParaS-Compiler Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_GROUP_BARRIER_HPP__
#define __PARAS_GROUP_BARRIER_HPP__

#include "sub_group.hpp"

#define PARAS_KERNEL_D
#define PARAS_KERNEL_HD

namespace sycl {

template <typename Group>
PARAS_KERNEL_HD inline void
group_barrier(Group, memory_scope scope = Group::fence_scope) {
#if PARAS_GPU_BACKEND

  if constexpr (std::is_same_v<std::decay_t<Group>, sub_group>) {
    paras_syncwarp();
  } else {
    paras_syncthreads();
  }

#else

#endif
}

} // namespace sycl

#endif
