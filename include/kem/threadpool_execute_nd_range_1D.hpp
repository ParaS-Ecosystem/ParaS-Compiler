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

#ifndef __PARAS_THREADPOOL_EXECUTE_ND_RANGE_1D_HPP__
#define __PARAS_THREADPOOL_EXECUTE_ND_RANGE_1D_HPP__

#include <algorithm>
#include <thread>
#include <vector>

#include "sycl/nd_item.hpp"
#include "sycl/nd_range.hpp"

template <typename Func>
void threadpool::execute_nd_range_1D(const sycl::nd_range<1> &r, Func f) {

  auto GR = r.get_global_range();
  auto LR = r.get_local_range();
  auto GRR = r.get_group_range();

  const size_t global_size = GR[0];
  const size_t local_size = LR[0];

  const size_t num_threads =
      std::max<size_t>(1, std::thread::hardware_concurrency());

  const size_t work_per_thread = (global_size + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t t = 0; t < num_threads; ++t) {

    const size_t start = t * work_per_thread;
    const size_t end = std::min(start + work_per_thread, global_size);

    if (start >= end)
      break;

    threads.emplace_back([=, &f]() {
      for (size_t gid = start; gid < end; ++gid) {

        const size_t group_id = gid / local_size;
        const size_t local_id = gid % local_size;

        sycl::nd_item<1> item(sycl::id<1>(gid), sycl::id<1>(local_id),
                              sycl::id<1>(group_id), GR, LR, GRR);

        f(item);
      }
    });
  }

  for (auto &th : threads)
    th.join();
}

#endif
