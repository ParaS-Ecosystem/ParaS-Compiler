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

#ifndef __PARAS_THREADPOOL_EXECUTE_ND_HPP__
#define __PARAS_THREADPOOL_EXECUTE_ND_HPP__

#include <algorithm>
#include <thread>
#include <vector>
#include "kem/host_pool.hpp"

#include "sycl/id.hpp"
#include "sycl/item.hpp"
#include "sycl/range.hpp"

template <typename Func>
void threadpool::execute_2D(const sycl::range<2> &r, Func f) {
  const size_t C = r[1];
  paras_host_detail::host_parallel_chunks(
      r[0] * C, [&](std::size_t start, std::size_t end) {
        for (std::size_t idx = start; idx < end; idx++)
          sycl::detail::invoke_range_kernel<2>(
              f, sycl::id<2>(idx / C, idx % C), r);
      });
}
#endif
