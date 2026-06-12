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

#ifndef __PARAS_THREADPOOL_SPWAN_HPP__
#define __PARAS_THREADPOOL_SPWAN_HPP__

#include "sycl/handler.hpp"

template <typename Func> void threadpool::spawn_1D(Func f) {
  sycl::handler cgh(*this);
  f(cgh);
}

template <typename Func> sycl::event threadpool::spawn_1D_event(Func f) {
  spawn_1D(f);
  return sycl::event{};
}

template <typename Func> void threadpool::spawn_ND(Func f) {
  sycl::handler cgh(*this);
  f(cgh);
}
#endif
