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

#ifndef __PARAS_GET_PTR_HPP__
#define __PARAS_GET_PTR_HPP__

#include "context.hpp"
#include "usm_alloc.hpp"

namespace sycl {
namespace usm {

inline alloc get_pointer_type(const void *ptr,
                              const sycl::context &syclContext) {
  return ptr ? alloc::unknown : alloc::unknown;
}

} // namespace usm
} // namespace sycl

#endif
