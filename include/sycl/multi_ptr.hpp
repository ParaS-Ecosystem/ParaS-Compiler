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

#ifndef __PARAS_MULTI_PTR_HPP__
#define __PARAS_MULTI_PTR_HPP__

#include "access.hpp"
#include <cstddef>
#include <type_traits>
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename ElementType, access::address_space Space,
          access::decorated IsDecorated>
class multi_ptr {
public:
  using element_type = ElementType;
  using pointer = ElementType *;
  using difference_type = std::ptrdiff_t;

  PARAS_KERNEL_HD
  constexpr multi_ptr() noexcept : ptr_(nullptr) {}

  PARAS_KERNEL_HD
  constexpr multi_ptr(const pointer p) noexcept
      : ptr_(const_cast<pointer>(p)) {}

  template <typename U>
  PARAS_KERNEL_HD multi_ptr(U *const p) noexcept
      : ptr_(const_cast<pointer>(p)) {}

  PARAS_KERNEL_HD
  multi_ptr(const multi_ptr &) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr(multi_ptr &&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr &operator=(const multi_ptr &) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr &operator=(multi_ptr &&) noexcept = default;

  template <typename U = ElementType,
            typename = std::enable_if_t<!std::is_const_v<U>>>
  PARAS_KERNEL_HD
  operator multi_ptr<const ElementType, Space, IsDecorated>() const noexcept {
    return multi_ptr<const ElementType, Space, IsDecorated>(ptr_);
  }

  PARAS_KERNEL_HD
  element_type &operator*() noexcept { return *ptr_; }

  PARAS_KERNEL_HD
  const element_type &operator*() const noexcept { return *ptr_; }

  PARAS_KERNEL_HD
  pointer operator->() noexcept { return ptr_; }

  PARAS_KERNEL_HD
  const pointer operator->() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  pointer get() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  operator pointer() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  multi_ptr &operator=(pointer p) noexcept {
    ptr_ = p;
    return *this;
  }

  PARAS_KERNEL_HD
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  PARAS_KERNEL_HD
  element_type &operator[](std::size_t idx) noexcept { return ptr_[idx]; }

  PARAS_KERNEL_HD
  const element_type &operator[](std::size_t idx) const noexcept {
    return ptr_[idx];
  }

  PARAS_KERNEL_HD
  multi_ptr operator+(std::size_t offset) const noexcept {
    return multi_ptr(ptr_ + offset);
  }

  PARAS_KERNEL_HD
  multi_ptr operator-(std::size_t offset) const noexcept {
    return multi_ptr(ptr_ - offset);
  }

private:
  pointer ptr_;
};

} // namespace sycl

#endif
