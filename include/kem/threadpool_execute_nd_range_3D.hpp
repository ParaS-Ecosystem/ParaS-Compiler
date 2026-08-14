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

#ifndef __PARAS_THREADPOOL_EXECUTE_ND_RANGE_3D_HPP__
#define __PARAS_THREADPOOL_EXECUTE_ND_RANGE_3D_HPP__

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

template <typename Func>
void threadpool::execute_nd_range_3D(const sycl::nd_range<3> &r, Func f) {
  auto global = r.get_global_range();
  auto local = r.get_local_range();
  auto groups = r.get_group_range();

  const size_t G0 = global[0];
  const size_t G1 = global[1];
  const size_t G2 = global[2];

  const size_t L0 = local[0];
  const size_t L1 = local[1];
  const size_t L2 = local[2];

  if (G0 == 0 || G1 == 0 || G2 == 0)
    return;

  if (L0 == 0 || L1 == 0 || L2 == 0)
    throw std::runtime_error("Invalid local range");

  if ((G0 % L0) != 0 || (G1 % L1) != 0 || (G2 % L2) != 0) {
    throw std::runtime_error("Global range must be divisible by local range");
  }

  const size_t total = G0 * G1 * G2;

  const size_t nthreads =
      std::max<size_t>(1, std::thread::hardware_concurrency());

  const size_t chunk = (total + nthreads - 1) / nthreads;

  std::vector<std::thread> threads;

  for (size_t t = 0; t < nthreads; ++t) {
    size_t start = t * chunk;
    size_t end = std::min(start + chunk, total);

    if (start >= end)
      break;

    threads.emplace_back([=, &f]() {
      for (size_t linear = start; linear < end; ++linear) {
        size_t gid0 = linear / (G1 * G2);
        size_t rem = linear % (G1 * G2);
        size_t gid1 = rem / G2;
        size_t gid2 = rem % G2;

        size_t lid0 = gid0 % L0;
        size_t lid1 = gid1 % L1;
        size_t lid2 = gid2 % L2;

        size_t grp0 = gid0 / L0;
        size_t grp1 = gid1 / L1;
        size_t grp2 = gid2 / L2;

        sycl::nd_item<3> item(
            sycl::id<3>(gid0, gid1, gid2), sycl::id<3>(lid0, lid1, lid2),
            sycl::id<3>(grp0, grp1, grp2), global, local, groups);

        f(item);
      }
    });
  }

  for (auto &th : threads)
    th.join();
}

#endif
