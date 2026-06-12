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

#ifndef __PARAS_USM_HPP__
#define __PARAS_USM_HPP__

#include "queue.hpp"
#include <cstdlib>
#include <iostream>

namespace sycl {

template <typename T>
T *malloc_shared(size_t n, const queue &, const property_list &propList = {}) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T>
T *malloc_host(size_t n, const queue &, const property_list &propList = {}) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

template <typename T>
T *malloc_device(size_t n, const queue &, const property_list &propList = {}) {
  return static_cast<T *>(std::malloc(sizeof(T) * n));
}

inline void free(void *ptr, const queue &) { std::free(ptr); }

} // namespace sycl

#endif
