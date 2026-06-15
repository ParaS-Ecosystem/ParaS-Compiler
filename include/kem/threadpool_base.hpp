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

#ifndef __PARAS_THREADPOOL_BASE_HPP__
#define __PARAS_THREADPOOL_BASE_HPP__

#include "sycl/context.hpp"
#include "sycl/device.hpp"
#include <thread>

class threadpool_base {
public:
  sycl::device get_device() const { return sycl::device(); }
  sycl::context get_context() const { return sycl::context(); }

  static unsigned get_num_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 4 : n;
  }

  void wait() {}
};
#endif
