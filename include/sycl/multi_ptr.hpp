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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include "access.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget,
          access::placeholder IsPlaceholder>
class accessor;
template <typename DataT, int Dimensions>
class local_accessor;

template <typename T>
struct remove_decoration {
    using type = T;
};
template <typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

namespace detail {
template <access::address_space Space>
inline constexpr bool mp_global_or_generic_v = Space == access::address_space::global_space ||
                                               Space == access::address_space::generic_space;
template <access::address_space Space>
inline constexpr bool mp_local_or_generic_v = Space == access::address_space::local_space ||
                                              Space == access::address_space::generic_space;
template <typename ElementType, typename AccDataT, typename AccValueT>
inline constexpr bool mp_from_acc_v =
    (std::is_void_v<ElementType> ||
     std::is_same_v<std::remove_const_t<ElementType>, std::remove_const_t<AccDataT>>) &&
    (std::is_const_v<ElementType> || !std::is_const_v<AccValueT>);

template <typename AccT>
PARAS_KERNEL_HD auto mp_acc_pointer(const AccT& acc) noexcept;
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy>
class multi_ptr {
public:
    static constexpr bool is_decorated = DecorateAddress == access::decorated::yes;
    static constexpr access::address_space address_space = Space;

    using value_type = ElementType;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference_t<value_type>;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;

    static_assert(std::is_same_v<remove_decoration_t<pointer>, std::add_pointer_t<value_type>>);
    static_assert(
        std::is_same_v<remove_decoration_t<reference>, std::add_lvalue_reference_t<value_type>>);

    PARAS_KERNEL_HD multi_ptr() noexcept : ptr_(nullptr) {}
    multi_ptr(const multi_ptr&) = default;
    multi_ptr(multi_ptr&&) = default;
    PARAS_KERNEL_HD explicit multi_ptr(pointer p) noexcept : ptr_(p) {}
    PARAS_KERNEL_HD multi_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

    template <typename AccDataT, int Dimensions, access_mode Mode,
              access::placeholder IsPlaceholder, access::address_space S = Space,
              typename = std::enable_if_t<
                  detail::mp_global_or_generic_v<S> &&
                  detail::mp_from_acc_v<
                      ElementType, AccDataT,
                      typename accessor<AccDataT, Dimensions, Mode, target::device,
                                        IsPlaceholder>::value_type>>>
    PARAS_KERNEL_HD multi_ptr(accessor<AccDataT, Dimensions, Mode, target::device, IsPlaceholder> a)
        : ptr_(const_cast<pointer>(static_cast<const ElementType*>(detail::mp_acc_pointer(a)))) {}

    template <typename AccDataT, int Dimensions, access::address_space S = Space,
              typename = std::enable_if_t<detail::mp_local_or_generic_v<S> &&
                                          detail::mp_from_acc_v<ElementType, AccDataT, AccDataT>>>
    PARAS_KERNEL_HD multi_ptr(local_accessor<AccDataT, Dimensions> a)
        : ptr_(const_cast<pointer>(static_cast<const ElementType*>(detail::mp_acc_pointer(a)))) {}

    template <typename AccDataT, int Dimensions, access_mode Mode,
              access::placeholder IsPlaceholder, access::address_space S = Space,
              typename = std::enable_if_t<detail::mp_local_or_generic_v<S> &&
                                          detail::mp_from_acc_v<ElementType, AccDataT, AccDataT>>>
    PARAS_KERNEL_HD multi_ptr(accessor<AccDataT, Dimensions, Mode, target::local, IsPlaceholder> a)
        : ptr_(const_cast<pointer>(static_cast<const ElementType*>(detail::mp_acc_pointer(a)))) {}

    template <typename AccDataT, int Dimensions, access::placeholder IsPlaceholder,
              access::address_space S = Space,
              typename = std::enable_if_t<S == access::address_space::constant_space &&
                                          detail::mp_from_acc_v<ElementType, AccDataT, AccDataT>>>
    PARAS_KERNEL_HD multi_ptr(
        accessor<AccDataT, Dimensions, access_mode::read, target::constant_buffer, IsPlaceholder> a)
        : ptr_(const_cast<pointer>(static_cast<const ElementType*>(detail::mp_acc_pointer(a)))) {}

    multi_ptr& operator=(const multi_ptr&) = default;
    multi_ptr& operator=(multi_ptr&&) = default;
    PARAS_KERNEL_HD multi_ptr& operator=(std::nullptr_t) noexcept {
        ptr_ = nullptr;
        return *this;
    }
    template <access::address_space AS, access::decorated IsDecorated,
              access::address_space S = Space,
              typename = std::enable_if_t<S == access::address_space::generic_space &&
                                          AS != access::address_space::constant_space>>
    PARAS_KERNEL_HD multi_ptr& operator=(const multi_ptr<value_type, AS, IsDecorated>& other) {
        ptr_ = other.get_raw();
        return *this;
    }

    PARAS_KERNEL_HD reference operator[](std::ptrdiff_t i) const { return ptr_[i]; }
    PARAS_KERNEL_HD reference operator*() const { return *ptr_; }
    PARAS_KERNEL_HD pointer operator->() const { return ptr_; }
    PARAS_KERNEL_HD pointer get() const { return ptr_; }
    PARAS_KERNEL_HD std::add_pointer_t<value_type> get_raw() const { return ptr_; }
    PARAS_KERNEL_HD pointer get_decorated() const { return ptr_; }
    PARAS_KERNEL_HD operator pointer() const { return ptr_; }

#define PARAS_MP_GENERIC_CAST(TARGET_SPACE)                                                         \
    template <access::decorated IsDecorated, access::address_space S = Space,                      \
              typename = std::enable_if_t<S == access::address_space::generic_space>>              \
    PARAS_KERNEL_HD explicit operator multi_ptr<value_type, access::address_space::TARGET_SPACE,   \
                                                IsDecorated>() const {                             \
        return multi_ptr<value_type, access::address_space::TARGET_SPACE, IsDecorated>(ptr_);      \
    }                                                                                              \
    template <access::decorated IsDecorated, access::address_space S = Space,                      \
              typename V = value_type,                                                             \
              typename = std::enable_if_t<S == access::address_space::generic_space &&             \
                                          !std::is_const_v<V>>>                                    \
    PARAS_KERNEL_HD explicit operator multi_ptr<const value_type,                                  \
                                                access::address_space::TARGET_SPACE,               \
                                                IsDecorated>() const {                             \
        return multi_ptr<const value_type, access::address_space::TARGET_SPACE, IsDecorated>(      \
            ptr_);                                                                                 \
    }
    PARAS_MP_GENERIC_CAST(private_space)
    PARAS_MP_GENERIC_CAST(global_space)
    PARAS_MP_GENERIC_CAST(local_space)
#undef PARAS_MP_GENERIC_CAST

    template <access::decorated IsDecorated, typename V = value_type,
              typename = std::enable_if_t<!std::is_const_v<V>>>
    PARAS_KERNEL_HD operator multi_ptr<void, Space, IsDecorated>() const {
        return multi_ptr<void, Space, IsDecorated>(static_cast<void*>(ptr_));
    }
    template <access::decorated IsDecorated, typename V = value_type,
              typename = std::enable_if_t<std::is_const_v<V>>>
    PARAS_KERNEL_HD operator multi_ptr<const void, Space, IsDecorated>() const {
        return multi_ptr<const void, Space, IsDecorated>(static_cast<const void*>(ptr_));
    }
    template <access::decorated IsDecorated, typename V = value_type,
              typename = std::enable_if_t<!std::is_const_v<V>>>
    PARAS_KERNEL_HD operator multi_ptr<const value_type, Space, IsDecorated>() const {
        return multi_ptr<const value_type, Space, IsDecorated>(ptr_);
    }
    template <bool D = is_decorated, typename = std::enable_if_t<D>>
    PARAS_KERNEL_HD operator multi_ptr<value_type, Space, access::decorated::no>() const {
        return multi_ptr<value_type, Space, access::decorated::no>(ptr_);
    }
    template <bool D = is_decorated, typename = std::enable_if_t<!D>>
    PARAS_KERNEL_HD operator multi_ptr<value_type, Space, access::decorated::yes>() const {
        return multi_ptr<value_type, Space, access::decorated::yes>(ptr_);
    }

    template <access::address_space S = Space,
              typename = std::enable_if_t<S == access::address_space::global_space>>
    PARAS_KERNEL_HD void prefetch(std::size_t) const {}

    PARAS_KERNEL_HD friend multi_ptr& operator++(multi_ptr& mp) {
        ++mp.ptr_;
        return mp;
    }
    PARAS_KERNEL_HD friend multi_ptr operator++(multi_ptr& mp, int) {
        multi_ptr t(mp);
        ++mp.ptr_;
        return t;
    }
    PARAS_KERNEL_HD friend multi_ptr& operator--(multi_ptr& mp) {
        --mp.ptr_;
        return mp;
    }
    PARAS_KERNEL_HD friend multi_ptr operator--(multi_ptr& mp, int) {
        multi_ptr t(mp);
        --mp.ptr_;
        return t;
    }
    PARAS_KERNEL_HD friend multi_ptr& operator+=(multi_ptr& lhs, difference_type r) {
        lhs.ptr_ += r;
        return lhs;
    }
    PARAS_KERNEL_HD friend multi_ptr& operator-=(multi_ptr& lhs, difference_type r) {
        lhs.ptr_ -= r;
        return lhs;
    }
    PARAS_KERNEL_HD friend multi_ptr operator+(const multi_ptr& lhs, difference_type r) {
        return multi_ptr(lhs.ptr_ + r);
    }
    PARAS_KERNEL_HD friend multi_ptr operator-(const multi_ptr& lhs, difference_type r) {
        return multi_ptr(lhs.ptr_ - r);
    }
#define PARAS_MP_CMP(OP)                                                                           \
    PARAS_KERNEL_HD friend bool operator OP(const multi_ptr& lhs, const multi_ptr& rhs) {          \
        return lhs.ptr_ OP rhs.ptr_;                                                               \
    }                                                                                              \
    PARAS_KERNEL_HD friend bool operator OP(const multi_ptr& lhs, std::nullptr_t) {                \
        return lhs.ptr_ OP static_cast<pointer>(nullptr);                                          \
    }                                                                                              \
    PARAS_KERNEL_HD friend bool operator OP(std::nullptr_t, const multi_ptr& rhs) {                \
        return static_cast<pointer>(nullptr) OP rhs.ptr_;                                          \
    }
    PARAS_MP_CMP(==)
    PARAS_MP_CMP(!=)
    PARAS_MP_CMP(<)
    PARAS_MP_CMP(>)
    PARAS_MP_CMP(<=)
    PARAS_MP_CMP(>=)
#undef PARAS_MP_CMP

private:
    pointer ptr_;
};

#define PARAS_VOID_MULTI_PTR(CV_VOID)                                                              \
    template <access::address_space Space, access::decorated DecorateAddress>                      \
    class multi_ptr<CV_VOID, Space, DecorateAddress> {                                             \
    public:                                                                                        \
        static constexpr bool is_decorated = DecorateAddress == access::decorated::yes;            \
        static constexpr access::address_space address_space = Space;                              \
        using value_type = CV_VOID;                                                                \
        using pointer = std::add_pointer_t<value_type>;                                            \
        using difference_type = std::ptrdiff_t;                                                    \
                                                                                                   \
        PARAS_KERNEL_HD multi_ptr() noexcept : ptr_(nullptr) {}                                    \
        multi_ptr(const multi_ptr&) = default;                                                     \
        multi_ptr(multi_ptr&&) = default;                                                          \
        PARAS_KERNEL_HD explicit multi_ptr(pointer p) noexcept : ptr_(p) {}                        \
        PARAS_KERNEL_HD multi_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}                      \
        template <typename ElementType, int Dimensions, access_mode Mode,                          \
                  access::placeholder IsPlaceholder, access::address_space S = Space,              \
                  typename = std::enable_if_t<                                                     \
                      detail::mp_global_or_generic_v<S> &&                                         \
                      (std::is_const_v<CV_VOID> ||                                                 \
                       !std::is_const_v<typename accessor<ElementType, Dimensions, Mode,           \
                                                          target::device,                          \
                                                          IsPlaceholder>::value_type>)>>           \
        PARAS_KERNEL_HD multi_ptr(                                                                 \
            accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder> a)              \
            : ptr_(const_cast<pointer>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}    \
        template <typename ElementType, int Dimensions, access::address_space S = Space,           \
                  typename = std::enable_if_t<detail::mp_local_or_generic_v<S> &&                  \
                                              (std::is_const_v<CV_VOID> ||                         \
                                               !std::is_const_v<ElementType>)>>                    \
        PARAS_KERNEL_HD multi_ptr(local_accessor<ElementType, Dimensions> a)                       \
            : ptr_(const_cast<pointer>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}    \
        template <typename ElementType, int Dimensions, access_mode Mode,                          \
                  access::placeholder IsPlaceholder, access::address_space S = Space,              \
                  typename = std::enable_if_t<detail::mp_local_or_generic_v<S> &&                  \
                                              (std::is_const_v<CV_VOID> ||                         \
                                               !std::is_const_v<ElementType>)>>                    \
        PARAS_KERNEL_HD multi_ptr(                                                                 \
            accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder> a)               \
            : ptr_(const_cast<pointer>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}    \
        template <typename ElementType, int Dimensions, access::placeholder IsPlaceholder,         \
                  access::address_space S = Space,                                                 \
                  typename = std::enable_if_t<S == access::address_space::constant_space &&        \
                                              (std::is_const_v<CV_VOID> ||                         \
                                               !std::is_const_v<ElementType>)>>                    \
        PARAS_KERNEL_HD multi_ptr(accessor<ElementType, Dimensions, access_mode::read,             \
                                           target::constant_buffer, IsPlaceholder> a)              \
            : ptr_(const_cast<pointer>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}    \
                                                                                                   \
        multi_ptr& operator=(const multi_ptr&) = default;                                          \
        multi_ptr& operator=(multi_ptr&&) = default;                                               \
        PARAS_KERNEL_HD multi_ptr& operator=(std::nullptr_t) noexcept {                            \
            ptr_ = nullptr;                                                                        \
            return *this;                                                                          \
        }                                                                                          \
        PARAS_KERNEL_HD pointer get() const { return ptr_; }                                       \
        PARAS_KERNEL_HD operator pointer() const { return ptr_; }                                  \
        template <typename ElementType,                                                            \
                  typename = std::enable_if_t<std::is_const_v<ElementType> ||                      \
                                              !std::is_const_v<CV_VOID>>>                          \
        PARAS_KERNEL_HD explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const { \
            return multi_ptr<ElementType, Space, DecorateAddress>(                                 \
                static_cast<ElementType*>(ptr_));                                                  \
        }                                                                                          \
        template <bool D = is_decorated, typename = std::enable_if_t<D>>                           \
        PARAS_KERNEL_HD operator multi_ptr<value_type, Space, access::decorated::no>() const {     \
            return multi_ptr<value_type, Space, access::decorated::no>(ptr_);                      \
        }                                                                                          \
        template <bool D = is_decorated, typename = std::enable_if_t<!D>>                          \
        PARAS_KERNEL_HD operator multi_ptr<value_type, Space, access::decorated::yes>() const {    \
            return multi_ptr<value_type, Space, access::decorated::yes>(ptr_);                     \
        }                                                                                          \
        template <typename V = CV_VOID, typename = std::enable_if_t<!std::is_const_v<V>>>          \
        PARAS_KERNEL_HD operator multi_ptr<const void, Space, DecorateAddress>() const {           \
            return multi_ptr<const void, Space, DecorateAddress>(ptr_);                            \
        }                                                                                          \
        PARAS_VOID_MP_CMP(==)                                                                      \
        PARAS_VOID_MP_CMP(!=)                                                                      \
        PARAS_VOID_MP_CMP(<)                                                                       \
        PARAS_VOID_MP_CMP(>)                                                                       \
        PARAS_VOID_MP_CMP(<=)                                                                      \
        PARAS_VOID_MP_CMP(>=)                                                                      \
                                                                                                   \
    private:                                                                                       \
        pointer ptr_;                                                                              \
    };
#define PARAS_VOID_MP_CMP(OP)                                                                      \
    PARAS_KERNEL_HD friend bool operator OP(const multi_ptr& lhs, const multi_ptr& rhs) {          \
        return lhs.ptr_ OP rhs.ptr_;                                                               \
    }                                                                                              \
    PARAS_KERNEL_HD friend bool operator OP(const multi_ptr& lhs, std::nullptr_t) {                \
        return lhs.ptr_ OP static_cast<pointer>(nullptr);                                          \
    }                                                                                              \
    PARAS_KERNEL_HD friend bool operator OP(std::nullptr_t, const multi_ptr& rhs) {                \
        return static_cast<pointer>(nullptr) OP rhs.ptr_;                                          \
    }
PARAS_VOID_MULTI_PTR(void)
PARAS_VOID_MULTI_PTR(const void)
#undef PARAS_VOID_MULTI_PTR

template <typename ElementType, access::address_space Space>
class [[deprecated]] multi_ptr<ElementType, Space, access::decorated::legacy> {
public:
    using value_type = ElementType;
    using element_type = ElementType;
    using difference_type = std::ptrdiff_t;
    using pointer_t = typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
    using const_pointer_t =
        typename multi_ptr<const ElementType, Space, access::decorated::yes>::pointer;
    using reference_t = typename multi_ptr<ElementType, Space, access::decorated::yes>::reference;
    using const_reference_t =
        typename multi_ptr<const ElementType, Space, access::decorated::yes>::reference;
    static constexpr access::address_space address_space = Space;

    PARAS_KERNEL_HD multi_ptr() noexcept : ptr_(nullptr) {}
    multi_ptr(const multi_ptr&) = default;
    multi_ptr(multi_ptr&&) = default;
    PARAS_KERNEL_HD multi_ptr(ElementType* p) noexcept : ptr_(p) {}
    PARAS_KERNEL_HD multi_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}
    ~multi_ptr() = default;

    multi_ptr& operator=(const multi_ptr&) = default;
    multi_ptr& operator=(multi_ptr&&) = default;
    PARAS_KERNEL_HD multi_ptr& operator=(ElementType* p) noexcept {
        ptr_ = p;
        return *this;
    }
    PARAS_KERNEL_HD multi_ptr& operator=(std::nullptr_t) noexcept {
        ptr_ = nullptr;
        return *this;
    }
    PARAS_KERNEL_HD friend ElementType& operator*(const multi_ptr& mp) { return *mp.ptr_; }
    PARAS_KERNEL_HD ElementType* operator->() const { return ptr_; }

    template <int Dimensions, access_mode Mode, access::placeholder IsPlaceholder,
              access::address_space S = Space,
              typename = std::enable_if_t<
                  detail::mp_global_or_generic_v<S> &&
                  detail::mp_from_acc_v<ElementType, ElementType,
                                        typename accessor<ElementType, Dimensions, Mode,
                                                          target::device, IsPlaceholder>::value_type>>>
    PARAS_KERNEL_HD multi_ptr(accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder> a)
        : ptr_(const_cast<ElementType*>(detail::mp_acc_pointer(a))) {}
    template <int Dimensions, access_mode Mode, access::placeholder IsPlaceholder,
              access::address_space S = Space,
              typename = std::enable_if_t<detail::mp_local_or_generic_v<S>>>
    PARAS_KERNEL_HD multi_ptr(accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder> a)
        : ptr_(const_cast<ElementType*>(detail::mp_acc_pointer(a))) {}
    template <typename AccDataT, int Dimensions, access::address_space S = Space,
              typename = std::enable_if_t<detail::mp_local_or_generic_v<S> &&
                                          detail::mp_from_acc_v<ElementType, AccDataT, AccDataT>>>
    PARAS_KERNEL_HD multi_ptr(local_accessor<AccDataT, Dimensions> a)
        : ptr_(const_cast<ElementType*>(
              static_cast<const ElementType*>(detail::mp_acc_pointer(a)))) {}
    template <int Dimensions, access_mode Mode, access::placeholder IsPlaceholder,
              access::address_space S = Space,
              typename = std::enable_if_t<S == access::address_space::constant_space>>
    PARAS_KERNEL_HD multi_ptr(
        accessor<ElementType, Dimensions, Mode, target::constant_buffer, IsPlaceholder> a)
        : ptr_(const_cast<ElementType*>(detail::mp_acc_pointer(a))) {}

    PARAS_KERNEL_HD pointer_t get() const { return ptr_; }
    PARAS_KERNEL_HD std::add_pointer_t<value_type> get_raw() const { return ptr_; }
    PARAS_KERNEL_HD pointer_t get_decorated() const { return ptr_; }
    PARAS_KERNEL_HD operator ElementType*() const { return ptr_; }

    template <typename V = ElementType, typename = std::enable_if_t<!std::is_const_v<V>>>
    PARAS_KERNEL_HD operator multi_ptr<void, Space, access::decorated::legacy>() const {
        return multi_ptr<void, Space, access::decorated::legacy>(static_cast<void*>(ptr_));
    }
    template <typename V = ElementType, typename = std::enable_if_t<std::is_const_v<V>>>
    PARAS_KERNEL_HD operator multi_ptr<const void, Space, access::decorated::legacy>() const {
        return multi_ptr<const void, Space, access::decorated::legacy>(
            static_cast<const void*>(ptr_));
    }
    template <typename V = ElementType, typename = std::enable_if_t<!std::is_const_v<V>>>
    PARAS_KERNEL_HD operator multi_ptr<const ElementType, Space, access::decorated::legacy>() const {
        return multi_ptr<const ElementType, Space, access::decorated::legacy>(ptr_);
    }

    PARAS_KERNEL_HD friend multi_ptr& operator++(multi_ptr& mp) {
        ++mp.ptr_;
        return mp;
    }
    PARAS_KERNEL_HD friend multi_ptr operator++(multi_ptr& mp, int) {
        multi_ptr t(mp);
        ++mp.ptr_;
        return t;
    }
    PARAS_KERNEL_HD friend multi_ptr& operator--(multi_ptr& mp) {
        --mp.ptr_;
        return mp;
    }
    PARAS_KERNEL_HD friend multi_ptr operator--(multi_ptr& mp, int) {
        multi_ptr t(mp);
        --mp.ptr_;
        return t;
    }
    PARAS_KERNEL_HD friend multi_ptr& operator+=(multi_ptr& lhs, difference_type r) {
        lhs.ptr_ += r;
        return lhs;
    }
    PARAS_KERNEL_HD friend multi_ptr& operator-=(multi_ptr& lhs, difference_type r) {
        lhs.ptr_ -= r;
        return lhs;
    }
    PARAS_KERNEL_HD friend multi_ptr operator+(const multi_ptr& lhs, difference_type r) {
        return multi_ptr(lhs.ptr_ + r);
    }
    PARAS_KERNEL_HD friend multi_ptr operator-(const multi_ptr& lhs, difference_type r) {
        return multi_ptr(lhs.ptr_ - r);
    }
    PARAS_KERNEL_HD void prefetch(std::size_t) const {}

private:
    using pointer = ElementType*;

public:
    PARAS_VOID_MP_CMP(==)
    PARAS_VOID_MP_CMP(!=)
    PARAS_VOID_MP_CMP(<)
    PARAS_VOID_MP_CMP(>)
    PARAS_VOID_MP_CMP(<=)
    PARAS_VOID_MP_CMP(>=)

private:
    ElementType* ptr_;
};

#define PARAS_LEGACY_VOID_MULTI_PTR(CV_VOID)                                                       \
    template <access::address_space Space>                                                         \
    class [[deprecated]] multi_ptr<CV_VOID, Space, access::decorated::legacy> {                    \
    public:                                                                                        \
        using value_type = CV_VOID;                                                                \
        using element_type = CV_VOID;                                                              \
        using difference_type = std::ptrdiff_t;                                                    \
        using pointer_t = CV_VOID*;                                                                \
        using const_pointer_t = const void*;                                                       \
        static constexpr access::address_space address_space = Space;                              \
                                                                                                   \
        PARAS_KERNEL_HD multi_ptr() noexcept : ptr_(nullptr) {}                                    \
        multi_ptr(const multi_ptr&) = default;                                                     \
        multi_ptr(multi_ptr&&) = default;                                                          \
        PARAS_KERNEL_HD multi_ptr(CV_VOID* p) noexcept : ptr_(p) {}                                \
        PARAS_KERNEL_HD multi_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}                      \
        ~multi_ptr() = default;                                                                    \
        multi_ptr& operator=(const multi_ptr&) = default;                                          \
        multi_ptr& operator=(multi_ptr&&) = default;                                               \
        PARAS_KERNEL_HD multi_ptr& operator=(CV_VOID* p) noexcept {                                \
            ptr_ = p;                                                                              \
            return *this;                                                                          \
        }                                                                                          \
        PARAS_KERNEL_HD multi_ptr& operator=(std::nullptr_t) noexcept {                            \
            ptr_ = nullptr;                                                                        \
            return *this;                                                                          \
        }                                                                                          \
        template <typename ElementType, int Dimensions, access_mode Mode,                          \
                  access::address_space S = Space,                                                 \
                  typename = std::enable_if_t<detail::mp_global_or_generic_v<S>>>                  \
        PARAS_KERNEL_HD multi_ptr(accessor<ElementType, Dimensions, Mode, target::device,          \
                                           access::placeholder::false_t> a)                        \
            : ptr_(const_cast<CV_VOID*>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}   \
        template <typename AccDataT, int Dimensions, access::address_space S = Space,              \
                  typename = std::enable_if_t<detail::mp_local_or_generic_v<S>>>                   \
        PARAS_KERNEL_HD multi_ptr(local_accessor<AccDataT, Dimensions> a)                          \
            : ptr_(const_cast<CV_VOID*>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}   \
                       \
                     \
                                \
        template <typename ElementType, int Dimensions, access_mode Mode,                          \
                  access::address_space S = Space,                                                 \
                  typename = std::enable_if_t<detail::mp_local_or_generic_v<S>>>                   \
        PARAS_KERNEL_HD multi_ptr(accessor<ElementType, Dimensions, Mode, target::local,           \
                                           access::placeholder::false_t> a)                        \
            : ptr_(const_cast<CV_VOID*>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}   \
        template <typename ElementType, int Dimensions, access_mode Mode,                          \
                  access::address_space S = Space,                                                 \
                  typename = std::enable_if_t<S == access::address_space::constant_space>>         \
        PARAS_KERNEL_HD multi_ptr(accessor<ElementType, Dimensions, Mode,                          \
                                           target::constant_buffer, access::placeholder::false_t> a) \
            : ptr_(const_cast<CV_VOID*>(static_cast<const void*>(detail::mp_acc_pointer(a)))) {}   \
        PARAS_KERNEL_HD pointer_t get() const { return ptr_; }                                     \
        PARAS_KERNEL_HD std::add_pointer_t<value_type> get_raw() const { return ptr_; }            \
        PARAS_KERNEL_HD pointer_t get_decorated() const { return ptr_; }                           \
        PARAS_KERNEL_HD operator CV_VOID*() const { return ptr_; }                                 \
        template <typename ElementType>                                                            \
        PARAS_KERNEL_HD explicit                                                                   \
        operator multi_ptr<ElementType, Space, access::decorated::legacy>() const {                \
            return multi_ptr<ElementType, Space, access::decorated::legacy>(                       \
                static_cast<ElementType*>(ptr_));                                                  \
        }                                                                                          \
        template <typename V = CV_VOID, typename = std::enable_if_t<!std::is_const_v<V>>>          \
        PARAS_KERNEL_HD operator multi_ptr<const void, Space, access::decorated::legacy>() const { \
            return multi_ptr<const void, Space, access::decorated::legacy>(ptr_);                  \
        }                                                                                          \
    private:                                                                                       \
        using pointer = CV_VOID*;                                                                  \
                                                                                                   \
    public:                                                                                        \
        PARAS_VOID_MP_CMP(==)                                                                      \
        PARAS_VOID_MP_CMP(!=)                                                                      \
        PARAS_VOID_MP_CMP(<)                                                                       \
        PARAS_VOID_MP_CMP(>)                                                                       \
        PARAS_VOID_MP_CMP(<=)                                                                      \
        PARAS_VOID_MP_CMP(>=)                                                                      \
                                                                                                   \
    private:                                                                                       \
        CV_VOID* ptr_;                                                                             \
    };
PARAS_LEGACY_VOID_MULTI_PTR(void)
PARAS_LEGACY_VOID_MULTI_PTR(const void)
#undef PARAS_LEGACY_VOID_MULTI_PTR
#undef PARAS_VOID_MP_CMP

template <typename ElementType, access::address_space Space, access::decorated DecorateAddress>
PARAS_KERNEL_HD multi_ptr<ElementType, Space, DecorateAddress> make_ptr(ElementType* pointer) {
    return multi_ptr<ElementType, Space, DecorateAddress>(pointer);
}

template <access::address_space Space, access::decorated DecorateAddress, typename ElementType>
PARAS_KERNEL_HD multi_ptr<ElementType, Space, DecorateAddress>
address_space_cast(ElementType* pointer) noexcept {
    return multi_ptr<ElementType, Space, DecorateAddress>(pointer);
}

template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read, target::device, IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::global_space, access::decorated::no>;
template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::write, target::device, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;
template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read_write, target::device, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;
template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read, target::constant_buffer, IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::constant_space, access::decorated::no>;
template <typename T, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, Mode, target::local, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;
template <typename T, int Dimensions>
multi_ptr(local_accessor<T, Dimensions>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

template <typename ElementType>
using constant_ptr =
    multi_ptr<ElementType, access::address_space::constant_space, access::decorated::legacy>;
template <typename ElementType, access::decorated IsDecorated = access::decorated::legacy>
using generic_ptr = multi_ptr<ElementType, access::address_space::generic_space, IsDecorated>;
template <typename ElementType>
using raw_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space, access::decorated::no>;
template <typename ElementType>
using raw_local_ptr =
    multi_ptr<ElementType, access::address_space::local_space, access::decorated::no>;
template <typename ElementType>
using raw_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space, access::decorated::no>;
template <typename ElementType>
using raw_generic_ptr =
    multi_ptr<ElementType, access::address_space::generic_space, access::decorated::no>;

} // namespace sycl

#pragma clang diagnostic pop

#endif
