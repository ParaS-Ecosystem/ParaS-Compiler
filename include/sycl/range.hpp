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

#ifndef __PARAS_RANGE_HPP__
#define __PARAS_RANGE_HPP__

#include <cstddef>

#define PARAS_KERNEL_D
#define PARAS_KERNEL_HD

namespace sycl {

template <int Dimensions = 1> class range {
public:
  static constexpr int dimensions = Dimensions;

  size_t dims_[Dimensions];

  PARAS_KERNEL_HD
  range() {
    for (int i = 0; i < Dimensions; i++)
      dims_[i] = 0;
  }

  PARAS_KERNEL_HD
  range(size_t dim0) {
    static_assert(Dimensions == 1, "range(size_t) only valid for range<1>");
    dims_[0] = dim0;
  }

  PARAS_KERNEL_HD
  range(size_t dim0, size_t dim1) {
    static_assert(Dimensions == 2,
                  "range(size_t,size_t) only valid for range<2>");
    dims_[0] = dim0;
    dims_[1] = dim1;
  }

  PARAS_KERNEL_HD
  range(size_t dim0, size_t dim1, size_t dim2) {
    static_assert(Dimensions == 3,
                  "range(size_t,size_t,size_t) only valid for range<3>");
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
  }

  size_t get(int d) const { return dims_[d]; }

  PARAS_KERNEL_HD
  size_t &operator[](int d) { return dims_[d]; }

  PARAS_KERNEL_HD
  size_t operator[](int d) const { return dims_[d]; }

  PARAS_KERNEL_HD
  range operator*(const range &rhs) const {

    range result;
    for (int i = 0; i < Dimensions; ++i) {
      result[i] = dims_[i] * rhs[i];
    }
    return result;
  }

  size_t size() const {
    if constexpr (Dimensions == 1)
      return dims_[0];
    else if constexpr (Dimensions == 2)
      return dims_[0] * dims_[1];
    else
      return dims_[0] * dims_[1] * dims_[2];
  }
};

range(size_t) -> range<1>;
range(size_t, size_t) -> range<2>;
range(size_t, size_t, size_t) -> range<3>;

} // namespace sycl

#endif
