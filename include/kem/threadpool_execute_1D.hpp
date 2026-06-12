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

#ifndef __PARAS_THREADPOOL_EXECUTE_1D_HPP__
#define __PARAS_THREADPOOL_EXECUTE_1D_HPP__

#include "sycl/id.hpp"
#include "sycl/range.hpp"
#include <algorithm>
#include <thread>
#include <vector>

template <typename Func>
void threadpool::execute_1D(const sycl::range<1> &r, Func f) {
  size_t N = r[0];
  unsigned NT = get_num_threads();

  std::vector<std::thread> threads;
  threads.reserve(NT);

  size_t chunk = (N + NT - 1) / NT;

  for (unsigned t = 0; t < NT; t++) {
    size_t start = t * chunk;
    size_t end = std::min(start + chunk, N);
    if (start >= end)
      break;

    threads.emplace_back([=]() {
      for (size_t i = start; i < end; i++)
        f(sycl::id<1>(i));
    });
  }

  for (auto &th : threads)
    th.join();
}
#endif
