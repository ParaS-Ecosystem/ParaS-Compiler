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

#ifndef __PARAS_NATIVE_HPP__
#define __PARAS_NATIVE_HPP__

#include "math.hpp"

namespace sycl {
namespace native {

inline float rsqrt(float x) { return 1.0f / sycl::sqrt(x); }

} // namespace native
} // namespace sycl

#endif
