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

#ifndef __PARAS_ID_HPP__
#define __PARAS_ID_HPP__

#include <cstddef>

#define PARAS_KERNEL_D
#define PARAS_KERNEL_HD

namespace sycl {

template <int Dimensions = 1> class id {
public:
  static constexpr int dimensions = Dimensions;
  size_t dims_[Dimensions];

  PARAS_KERNEL_HD
  id() {
    for (int i = 0; i < Dimensions; i++)
      dims_[i] = 0;
  }

  PARAS_KERNEL_HD
  explicit id(size_t i0) {
    static_assert(Dimensions == 1, "id(size_t) is only valid for id<1>");
    dims_[0] = i0;
  }

  PARAS_KERNEL_HD
  id(size_t i0, size_t i1) {
    static_assert(Dimensions == 2, "id(size_t,size_t) is only valid for id<2>");
    dims_[0] = i0;
    dims_[1] = i1;
  }

  PARAS_KERNEL_HD
  id(size_t i0, size_t i1, size_t i2) {
    static_assert(Dimensions == 3,
                  "id(size_t,size_t,size_t) is only valid for id<3>");
    dims_[0] = i0;
    dims_[1] = i1;
    dims_[2] = i2;
  }

  PARAS_KERNEL_HD
  size_t &operator[](int dim) { return dims_[dim]; }

  PARAS_KERNEL_HD
  size_t operator[](int dim) const { return dims_[dim]; }

  PARAS_KERNEL_HD
  size_t get(int dim) const { return dims_[dim]; }

  PARAS_KERNEL_HD
  size_t &get(int dim) { return dims_[dim]; }

  PARAS_KERNEL_HD
  operator size_t() const {
    static_assert(Dimensions == 1, "Implicit conversion only valid for id<1>");
    return dims_[0];
  }
};

id(size_t) -> id<1>;
id(size_t, size_t) -> id<2>;
id(size_t, size_t, size_t) -> id<3>;

} // namespace sycl

#endif
