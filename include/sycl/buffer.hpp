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

#ifndef __PARAS_BUFFER_HPP__
#define __PARAS_BUFFER_HPP__

#include <cstdlib>
#include <memory>

#include "access.hpp"
#include "accessor.hpp"
#include "handler.hpp"
#include "property_list.hpp"
#include "range.hpp"

namespace sycl {

template <typename T, int Dimensions = 1,
          typename AllocatorT = std::allocator<T>>
class buffer {
  T *data_;
  range<Dimensions> r_;

public:
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using allocator_type = AllocatorT;

  buffer(const range<Dimensions> &rr, const property_list & = {}) : r_(rr) {
    size_t N = rr.size();
    data_ = new T[N]();
  }

  buffer(T *hostPtr, const range<Dimensions> &rr, const property_list & = {})
      : data_(hostPtr), r_(rr) {}

  buffer(const T *hostPtr, const range<Dimensions> &rr,
         const property_list & = {})
      : data_(const_cast<T *>(hostPtr)), r_(rr) {}

  template <access::mode Mode = access::mode::read_write,
            target AccessTarget = target::device>
  accessor<T, Dimensions, Mode, AccessTarget> get_access(handler &cgh) {
    return accessor<T, Dimensions, Mode, AccessTarget>(data_, r_, Mode);
  }

  template <access::mode Mode>
  accessor<T, Dimensions, Mode, target::host_buffer> get_access() {
    return accessor<T, Dimensions, Mode, target::host_buffer>(data_, r_, Mode);
  }

  T *data() const { return data_; }
};

} // namespace sycl

#endif
