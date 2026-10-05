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

#include <cstddef>
#include <type_traits>
#include <utility>
#include <iterator>
#include "access.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename ElementType, access::address_space Space, access::decorated IsDecorated = access::decorated::legacy>
class multi_ptr {
public:
  using value_type      = ElementType;
  using element_type    = ElementType;
  using pointer         = ElementType*;
  using reference       = ElementType&;
  using difference_type = std::ptrdiff_t;

  using iterator_category = std::random_access_iterator_tag;

  static constexpr bool is_decorated = (IsDecorated == access::decorated::yes);

  static constexpr access::address_space address_space = Space;

  PARAS_KERNEL_HD
  constexpr multi_ptr() noexcept : ptr_(nullptr) {}

  PARAS_KERNEL_HD
  constexpr multi_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

  PARAS_KERNEL_HD
  constexpr multi_ptr(const pointer p) noexcept : ptr_(const_cast<pointer>(p)) {}

  template <typename U>
  PARAS_KERNEL_HD
  multi_ptr(U* const p) noexcept : ptr_(const_cast<pointer>(p)) {}

  template <typename Ptr>
  static PARAS_KERNEL_HD pointer to_element_pointer(Ptr p) noexcept
  {
      if constexpr (std::is_pointer_v<Ptr>) {
          return const_cast<pointer>(p);
      } else {
          return static_cast<pointer>(p);
      }
  }

  template <typename AccessorT,
            typename = decltype(std::declval<const AccessorT &>().get_pointer())>
  PARAS_KERNEL_HD
  multi_ptr(const AccessorT &acc) noexcept
      : ptr_(to_element_pointer(acc.get_pointer())) {}

  PARAS_KERNEL_HD
  multi_ptr(const multi_ptr&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr(multi_ptr&&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr& operator=(const multi_ptr&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr& operator=(multi_ptr&&) noexcept = default;

  template <typename U = ElementType, typename = std::enable_if_t<!std::is_const_v<U>>>
  PARAS_KERNEL_HD
  operator multi_ptr<const ElementType, Space, IsDecorated>() const noexcept {
    return multi_ptr<const ElementType, Space, IsDecorated>(ptr_);
  }

  template <access::decorated OtherDecorated,
            typename = std::enable_if_t<OtherDecorated != IsDecorated>>
  PARAS_KERNEL_HD
  operator multi_ptr<ElementType, Space, OtherDecorated>() const noexcept {
    return multi_ptr<ElementType, Space, OtherDecorated>(ptr_);
  }

  template <access::decorated OtherDecorated, typename U = ElementType,
            typename = std::enable_if_t<!std::is_const_v<U> && OtherDecorated != IsDecorated>>
  PARAS_KERNEL_HD
  operator multi_ptr<void, Space, OtherDecorated>() const noexcept {
    return multi_ptr<void, Space, OtherDecorated>(static_cast<void *>(ptr_));
  }

  template <access::decorated OtherDecorated, typename U = ElementType,
            typename = std::enable_if_t<std::is_const_v<U> && OtherDecorated != IsDecorated>>
  PARAS_KERNEL_HD
  operator multi_ptr<const void, Space, OtherDecorated>() const noexcept {
    return multi_ptr<const void, Space, OtherDecorated>(static_cast<const void *>(ptr_));
  }

  template <access::decorated OtherDecorated, typename U = ElementType,
            typename = std::enable_if_t<!std::is_const_v<U> && OtherDecorated != IsDecorated>>
  PARAS_KERNEL_HD
  operator multi_ptr<const ElementType, Space, OtherDecorated>() const noexcept {
    return multi_ptr<const ElementType, Space, OtherDecorated>(ptr_);
  }

  PARAS_KERNEL_HD
  element_type& operator*() noexcept { return *ptr_; }

  PARAS_KERNEL_HD
  const element_type& operator*() const noexcept { return *ptr_; }

  PARAS_KERNEL_HD
  pointer operator->() noexcept { return ptr_; }

  PARAS_KERNEL_HD
  pointer operator->() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  pointer get() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  pointer get_decorated() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  void prefetch(std::size_t) const noexcept {}

  PARAS_KERNEL_HD
  operator pointer() const noexcept {
    return ptr_;
  }

  PARAS_KERNEL_HD
  multi_ptr& operator=(pointer p) noexcept {
    ptr_ = p;
    return *this;
  }

  PARAS_KERNEL_HD
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  PARAS_KERNEL_HD
  element_type& operator[](std::size_t idx) noexcept { return ptr_[idx]; }

  PARAS_KERNEL_HD
  const element_type& operator[](std::size_t idx) const noexcept { return ptr_[idx]; }

   PARAS_KERNEL_HD
  multi_ptr operator+(difference_type offset) const noexcept {
    return multi_ptr(ptr_ + offset);
  }

  PARAS_KERNEL_HD
  multi_ptr operator-(difference_type offset) const noexcept {
    return multi_ptr(ptr_ - offset);
  }

  PARAS_KERNEL_HD
	pointer get_raw() const noexcept { return ptr_; }

	PARAS_KERNEL_HD
	multi_ptr& operator+=(difference_type offset) noexcept {
	  ptr_ += offset;
	  return *this;
	}

	PARAS_KERNEL_HD
	multi_ptr& operator-=(difference_type offset) noexcept {
	  ptr_ -= offset;
	  return *this;
	}

	PARAS_KERNEL_HD
	multi_ptr& operator++() noexcept { ++ptr_; return *this; }

	PARAS_KERNEL_HD
	multi_ptr operator++(int) noexcept {
	  multi_ptr tmp(*this);
	  ++ptr_;
	  return tmp;
	}

	PARAS_KERNEL_HD
	multi_ptr& operator--() noexcept { --ptr_; return *this; }

	PARAS_KERNEL_HD
	multi_ptr operator--(int) noexcept {
	  multi_ptr tmp(*this);
	  --ptr_;
	  return tmp;
	}

private:
  pointer ptr_;
};

template <access::address_space Space, access::decorated IsDecorated>
class multi_ptr<void, Space, IsDecorated> {
public:
  using value_type      = void;
  using element_type    = void;
  using pointer         = void*;
  using difference_type = std::ptrdiff_t;

  PARAS_KERNEL_HD
  constexpr multi_ptr() noexcept : ptr_(nullptr) {}

  PARAS_KERNEL_HD
  constexpr multi_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

  PARAS_KERNEL_HD
  constexpr multi_ptr(pointer p) noexcept : ptr_(p) {}

  template <typename U>
  PARAS_KERNEL_HD
  multi_ptr(U* const p) noexcept : ptr_(const_cast<void*>(static_cast<const void*>(p))) {}

  template <typename AccessorT,
            typename = decltype(std::declval<const AccessorT &>().get_pointer())>
  PARAS_KERNEL_HD
  multi_ptr(const AccessorT &acc) noexcept
      : ptr_(const_cast<void*>(static_cast<const void*>(acc.get_pointer()))) {}

  PARAS_KERNEL_HD
  multi_ptr(const multi_ptr&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr(multi_ptr&&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr& operator=(const multi_ptr&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr& operator=(multi_ptr&&) noexcept = default;

  template <typename T>
  PARAS_KERNEL_HD
  multi_ptr(const multi_ptr<T, Space, IsDecorated>& other) noexcept
      : ptr_(const_cast<void*>(static_cast<const void*>(other.get()))) {}

  PARAS_KERNEL_HD
  pointer get() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  operator pointer() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  multi_ptr& operator=(pointer p) noexcept {
    ptr_ = p;
    return *this;
  }

  PARAS_KERNEL_HD
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  PARAS_KERNEL_HD
  pointer get_raw() const noexcept { return ptr_; }

 private:
  pointer ptr_;
};

template <access::address_space Space, access::decorated IsDecorated>
class multi_ptr<const void, Space, IsDecorated> {
public:
  using value_type      = const void;
  using element_type    = const void;
  using pointer         = const void*;
  using difference_type = std::ptrdiff_t;

  PARAS_KERNEL_HD
  constexpr multi_ptr() noexcept : ptr_(nullptr) {}

  PARAS_KERNEL_HD
  constexpr multi_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

  PARAS_KERNEL_HD
  constexpr multi_ptr(pointer p) noexcept : ptr_(p) {}

  template <typename U>
  PARAS_KERNEL_HD
  multi_ptr(const U* const p) noexcept : ptr_(static_cast<const void*>(p)) {}

  template <typename AccessorT,
            typename = decltype(std::declval<const AccessorT &>().get_pointer())>
  PARAS_KERNEL_HD
  multi_ptr(const AccessorT &acc) noexcept
      : ptr_(static_cast<const void*>(acc.get_pointer())) {}

  PARAS_KERNEL_HD
  multi_ptr(const multi_ptr&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr(multi_ptr&&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr& operator=(const multi_ptr&) noexcept = default;
  PARAS_KERNEL_HD
  multi_ptr& operator=(multi_ptr&&) noexcept = default;

  template <typename T>
  PARAS_KERNEL_HD
  multi_ptr(const multi_ptr<T, Space, IsDecorated>& other) noexcept
      : ptr_(static_cast<const void*>(other.get())) {}

  PARAS_KERNEL_HD
  pointer get() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  operator pointer() const noexcept { return ptr_; }

  PARAS_KERNEL_HD
  multi_ptr& operator=(pointer p) noexcept {
    ptr_ = p;
    return *this;
  }

  PARAS_KERNEL_HD
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  PARAS_KERNEL_HD
  pointer get_raw() const noexcept { return ptr_; }

 private:
  pointer ptr_;
};

template <typename ElementType, access::address_space Space,
          access::decorated IsDecorated = access::decorated::legacy>
PARAS_KERNEL_HD
inline multi_ptr<ElementType, Space, IsDecorated> make_ptr(ElementType *ptr) noexcept {
  return multi_ptr<ElementType, Space, IsDecorated>(ptr);
}

template <typename ElementType, access::address_space Space, access::decorated IsDecorated>
PARAS_KERNEL_HD
inline bool operator==(const multi_ptr<ElementType, Space, IsDecorated> &mp, std::nullptr_t) noexcept {
  return mp.get() == nullptr;
}

template <typename ElementType, access::address_space Space, access::decorated IsDecorated>
PARAS_KERNEL_HD
inline bool operator==(std::nullptr_t, const multi_ptr<ElementType, Space, IsDecorated> &mp) noexcept {
  return mp.get() == nullptr;
}

template <typename ElementType, access::address_space Space, access::decorated IsDecorated>
PARAS_KERNEL_HD
inline bool operator!=(const multi_ptr<ElementType, Space, IsDecorated> &mp, std::nullptr_t) noexcept {
  return mp.get() != nullptr;
}

template <typename ElementType, access::address_space Space, access::decorated IsDecorated>
PARAS_KERNEL_HD
inline bool operator!=(std::nullptr_t, const multi_ptr<ElementType, Space, IsDecorated> &mp) noexcept {
  return mp.get() != nullptr;
}

#define PARAS_MULTI_PTR_NULLPTR_RELOP(OP)                                  \
  template <typename ElementType, access::address_space Space, access::decorated IsDecorated> \
  PARAS_KERNEL_HD                                                          \
  inline bool operator OP(const multi_ptr<ElementType, Space, IsDecorated> &mp, std::nullptr_t) noexcept { \
    return mp.get() OP static_cast<typename multi_ptr<ElementType, Space, IsDecorated>::pointer>(nullptr); \
  }                                                                         \
  template <typename ElementType, access::address_space Space, access::decorated IsDecorated> \
  PARAS_KERNEL_HD                                                          \
  inline bool operator OP(std::nullptr_t, const multi_ptr<ElementType, Space, IsDecorated> &mp) noexcept { \
    return static_cast<typename multi_ptr<ElementType, Space, IsDecorated>::pointer>(nullptr) OP mp.get(); \
  }

PARAS_MULTI_PTR_NULLPTR_RELOP(<)
PARAS_MULTI_PTR_NULLPTR_RELOP(>)
PARAS_MULTI_PTR_NULLPTR_RELOP(<=)
PARAS_MULTI_PTR_NULLPTR_RELOP(>=)

#undef PARAS_MULTI_PTR_NULLPTR_RELOP

template <typename ElementType>
using raw_local_ptr = multi_ptr<ElementType, access::address_space::local_space, access::decorated::no>;
template <typename ElementType>
using raw_private_ptr = multi_ptr<ElementType, access::address_space::private_space, access::decorated::no>;
template <typename ElementType>
using raw_global_ptr = multi_ptr<ElementType, access::address_space::global_space, access::decorated::no>;
template <typename ElementType>
using raw_generic_ptr = multi_ptr<ElementType, access::address_space::generic_space, access::decorated::no>;

template <access::address_space Space, access::decorated IsDecorated, typename ElementType>
PARAS_KERNEL_HD
inline multi_ptr<ElementType, Space, IsDecorated> address_space_cast(ElementType *ptr) noexcept {
  return multi_ptr<ElementType, Space, IsDecorated>(ptr);
}

class stream;
template <typename ElementType, access::address_space Space, access::decorated IsDecorated>
const stream &operator<<(const stream &os, const multi_ptr<ElementType, Space, IsDecorated> &mp) {
  return os << mp.get();
}

}

#endif
