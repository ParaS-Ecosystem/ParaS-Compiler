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

#include "threadpool_execute_nd_range_common.hpp"

template <typename Func>
void threadpool::execute_nd_range_3D(const sycl::nd_range<3>& r, Func f) {
    paras_host_detail::execute_nd_range<3>(r, f);
}

#endif
