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
