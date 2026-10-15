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

#ifndef __PARAS_DEVICE_PRELUDE_HPP__
#define __PARAS_DEVICE_PRELUDE_HPP__

#if (defined(__HIP__) || defined(__CUDA__)) && defined(__cplusplus)
#include <cstddef>
#if defined(__GLIBCXX__) &&                                                    \
    (defined(_GLIBCXX_ASSERTIONS) || defined(_GLIBCXX_PARALLEL) ||             \
     defined(_GLIBCXX_PARALLEL_ASSERTIONS))
namespace std {
__attribute__((device)) __attribute__((noreturn)) inline void
__glibcxx_assert_fail(const char *, int, const char *, const char *) noexcept {
  __builtin_trap();
}
}
#endif

#endif

#endif
