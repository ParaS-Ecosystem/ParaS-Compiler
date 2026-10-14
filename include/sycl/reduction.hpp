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

#ifndef __PARAS_REDUCTION_HPP__
#define __PARAS_REDUCTION_HPP__

#include <cstddef>
#include <mutex>
#include <type_traits>
#include <utility>
#if __cplusplus >= 202002L
#include <span>
#endif
#include "kem_gpu/gpu_utilities.hpp"
#include "known_identity.hpp"
#include "property_list.hpp"

namespace sycl {

#if __cplusplus >= 202002L
using std::dynamic_extent;
using std::span;
#endif

template <typename T, int D, typename A>
class buffer;
class handler;

template <typename T, typename BinaryOperation, int Dims = 0, std::size_t Extent = 1,
          bool UserIdentity = false>
class reduction_impl {
public:
    static constexpr int dimensions = Dims;
    static constexpr std::size_t extent = Extent;
    static constexpr bool user_identity = UserIdentity;

    T* var_;
    BinaryOperation op_;
    T identity_;
    bool identity_known_;
    bool initialize_to_identity_;

    reduction_impl(T* varPtr, BinaryOperation op, T identity, bool identity_known,
                   bool initialize_to_identity)
        : var_(varPtr), op_(op), identity_(identity), identity_known_(identity_known),
          initialize_to_identity_(initialize_to_identity) {}
};

template <typename R>
struct paras_is_reduction : std::false_type {};
template <typename T, typename Op, int D, std::size_t E, bool U>
struct paras_is_reduction<reduction_impl<T, Op, D, E, U>> : std::true_type {};

namespace paras_reduction_detail {
template <typename BinaryOperation, typename T>
constexpr T identity_or_default() {
    if constexpr (has_known_identity_v<BinaryOperation, T>) {
        return known_identity_v<BinaryOperation, T>;
    } else {
        return T{};
    }
}
template <template <typename> class Op, typename BinaryOperation, typename T>
inline constexpr bool is_op_v =
    std::is_same_v<BinaryOperation, Op<T>> || std::is_same_v<BinaryOperation, Op<void>>;
} // namespace paras_reduction_detail

template <typename T, typename BinaryOperation>
reduction_impl<T, BinaryOperation> reduction(T* var, BinaryOperation combiner,
                                             const property_list& propList = {}) {
    return reduction_impl<T, BinaryOperation>(
        var, combiner, paras_reduction_detail::identity_or_default<BinaryOperation, T>(),
        has_known_identity_v<BinaryOperation, T>, propList.has_initialize_to_identity());
}

template <typename T, typename BinaryOperation>
reduction_impl<T, BinaryOperation, 0, 1, true> reduction(T* var, const T& identity,
                                                         BinaryOperation combiner,
                                                         const property_list& propList = {}) {
    return reduction_impl<T, BinaryOperation, 0, 1, true>(var, combiner, identity, true,
                                                          propList.has_initialize_to_identity());
}

template <typename T, int D, typename A, typename BinaryOperation>
reduction_impl<T, BinaryOperation> reduction(buffer<T, D, A> vars, handler&,
                                             BinaryOperation combiner,
                                             const property_list& propList = {}) {
    if (vars.size() != 1)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "reduction: the buffer range must be 1");
    return reduction_impl<T, BinaryOperation>(
        vars.data(), combiner, paras_reduction_detail::identity_or_default<BinaryOperation, T>(),
        has_known_identity_v<BinaryOperation, T>, propList.has_initialize_to_identity());
}
template <typename T, int D, typename A, typename BinaryOperation>
reduction_impl<T, BinaryOperation, 0, 1, true>
reduction(buffer<T, D, A> vars, handler&, const typename buffer<T, D, A>::value_type& identity,
          BinaryOperation combiner, const property_list& propList = {}) {
    if (vars.size() != 1)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "reduction: the buffer range must be 1");
    return reduction_impl<T, BinaryOperation, 0, 1, true>(vars.data(), combiner, identity, true,
                                                          propList.has_initialize_to_identity());
}

#if __cplusplus >= 202002L
template <typename T, std::size_t Extent, typename BinaryOperation,
          typename = std::enable_if_t<Extent != dynamic_extent>>
reduction_impl<T, BinaryOperation, 1, Extent> reduction(span<T, Extent> vars,
                                                        BinaryOperation combiner,
                                                        const property_list& propList = {}) {
    return reduction_impl<T, BinaryOperation, 1, Extent>(
        vars.data(), combiner, paras_reduction_detail::identity_or_default<BinaryOperation, T>(),
        has_known_identity_v<BinaryOperation, T>, propList.has_initialize_to_identity());
}
template <typename T, std::size_t Extent, typename BinaryOperation,
          typename = std::enable_if_t<Extent != dynamic_extent>>
reduction_impl<T, BinaryOperation, 1, Extent, true>
reduction(span<T, Extent> vars, const T& identity, BinaryOperation combiner,
          const property_list& propList = {}) {
    return reduction_impl<T, BinaryOperation, 1, Extent, true>(
        vars.data(), combiner, identity, true, propList.has_initialize_to_identity());
}
#endif

namespace paras_reduction_detail {
template <typename T>
struct slot {
    T value;
    bool has;
};

template <typename Derived, typename T, typename BinaryOperation, bool HasIdentity>
class reducer_ops {
public:
    PARAS_KERNEL_HD Derived& combine(const T& partial) {
        auto& d = static_cast<Derived&>(*this);
        slot<T>& s = *d.paras_slot();
        s.value = s.has ? d.paras_op()(s.value, partial) : partial;
        s.has = true;
        return d;
    }
    template <bool H = HasIdentity, typename = std::enable_if_t<H>>
    PARAS_KERNEL_HD T identity() const {
        return static_cast<const Derived&>(*this).paras_identity();
    }

    template <typename Op = BinaryOperation, typename = std::enable_if_t<is_op_v<plus, Op, T>>>
    PARAS_KERNEL_HD friend Derived& operator+=(Derived& r, const T& v) {
        return r.combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<is_op_v<multiplies, Op, T>>>
    PARAS_KERNEL_HD friend Derived& operator*=(Derived& r, const T& v) {
        return r.combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && is_op_v<bit_and, Op, T>>>
    PARAS_KERNEL_HD friend Derived& operator&=(Derived& r, const T& v) {
        return r.combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && is_op_v<bit_or, Op, T>>>
    PARAS_KERNEL_HD friend Derived& operator|=(Derived& r, const T& v) {
        return r.combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && is_op_v<bit_xor, Op, T>>>
    PARAS_KERNEL_HD friend Derived& operator^=(Derived& r, const T& v) {
        return r.combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool> &&
                                          is_op_v<plus, Op, T>>>
    PARAS_KERNEL_HD friend Derived& operator++(Derived& r) {
        return r.combine(T{1});
    }
};
} // namespace paras_reduction_detail

template <typename T, typename BinaryOperation, int Dims = 0,
          bool HasIdentity = has_known_identity_v<BinaryOperation, T>>
class reducer;

template <typename T, typename BinaryOperation, bool HasIdentity>
class reducer<T, BinaryOperation, 0, HasIdentity>
    : public paras_reduction_detail::reducer_ops<reducer<T, BinaryOperation, 0, HasIdentity>, T,
                                                 BinaryOperation, HasIdentity> {
public:
    using value_type = T;
    using binary_operation = BinaryOperation;
    static constexpr int dimensions = 0;

    reducer(const reducer&) = delete;
    reducer(reducer&&) = delete;
    reducer& operator=(const reducer&) = delete;
    reducer& operator=(reducer&&) = delete;
    ~reducer() = default;

    PARAS_KERNEL_HD reducer(paras_reduction_detail::slot<T>* s, const T& identity,
                            BinaryOperation op)
        : slot_(s), identity_(identity), op_(op) {}

    PARAS_KERNEL_HD paras_reduction_detail::slot<T>* paras_slot() const { return slot_; }
    PARAS_KERNEL_HD const BinaryOperation& paras_op() const { return op_; }
    PARAS_KERNEL_HD T paras_identity() const { return identity_; }

private:
    paras_reduction_detail::slot<T>* slot_;
    T identity_;
    BinaryOperation op_;
};

namespace paras_reduction_detail {
template <typename T, typename BinaryOperation, bool HasIdentity>
class reducer_element {
public:
    using value_type = T;
    using binary_operation = BinaryOperation;
    static constexpr int dimensions = 0;
    PARAS_KERNEL_HD reducer_element(slot<T>* s, const T& identity, const BinaryOperation& op)
        : slot_(s), identity_(identity), op_(op) {}

    PARAS_KERNEL_HD reducer_element& combine(const T& partial) {
        slot_->value = slot_->has ? op_(slot_->value, partial) : partial;
        slot_->has = true;
        return *this;
    }
    template <bool H = HasIdentity, typename = std::enable_if_t<H>>
    PARAS_KERNEL_HD T identity() const {
        return identity_;
    }
    template <typename Op = BinaryOperation, typename = std::enable_if_t<is_op_v<plus, Op, T>>>
    PARAS_KERNEL_HD reducer_element& operator+=(const T& v) {
        return combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<is_op_v<multiplies, Op, T>>>
    PARAS_KERNEL_HD reducer_element& operator*=(const T& v) {
        return combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && is_op_v<bit_and, Op, T>>>
    PARAS_KERNEL_HD reducer_element& operator&=(const T& v) {
        return combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && is_op_v<bit_or, Op, T>>>
    PARAS_KERNEL_HD reducer_element& operator|=(const T& v) {
        return combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && is_op_v<bit_xor, Op, T>>>
    PARAS_KERNEL_HD reducer_element& operator^=(const T& v) {
        return combine(v);
    }
    template <typename Op = BinaryOperation,
              typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool> &&
                                          is_op_v<plus, Op, T>>>
    PARAS_KERNEL_HD reducer_element& operator++() {
        return combine(T{1});
    }

private:
    slot<T>* slot_;
    T identity_;
    BinaryOperation op_;
};
} // namespace paras_reduction_detail

template <typename T, typename BinaryOperation, bool HasIdentity>
class reducer<T, BinaryOperation, 1, HasIdentity> {
public:
    using value_type = T;
    using binary_operation = BinaryOperation;
    static constexpr int dimensions = 1;

    reducer(const reducer&) = delete;
    reducer(reducer&&) = delete;
    reducer& operator=(const reducer&) = delete;
    reducer& operator=(reducer&&) = delete;
    ~reducer() = default;

    PARAS_KERNEL_HD reducer(paras_reduction_detail::slot<T>* slots, const T& identity,
                            BinaryOperation op)
        : slots_(slots), identity_(identity), op_(op) {}

    PARAS_KERNEL_HD paras_reduction_detail::reducer_element<T, BinaryOperation, HasIdentity>
    operator[](std::size_t index) {
        return paras_reduction_detail::reducer_element<T, BinaryOperation, HasIdentity>(
            slots_ + index, identity_, op_);
    }
    template <bool H = HasIdentity, typename = std::enable_if_t<H>>
    PARAS_KERNEL_HD T identity() const {
        return identity_;
    }

private:
    paras_reduction_detail::slot<T>* slots_;
    T identity_;
    BinaryOperation op_;
};

template <typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline void paras_reduction_merge(T* var, const T& v, BinaryOperation op) {
    if constexpr (std::is_trivially_copyable_v<T> &&
                  (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)) {
        T expected;
        __atomic_load(var, &expected, __ATOMIC_RELAXED);
        T desired = op(expected, v);
        while (!__atomic_compare_exchange(var, &expected, &desired, true, __ATOMIC_ACQ_REL,
                                          __ATOMIC_RELAXED)) {
            desired = op(expected, v);
        }
    } else {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        *var = op(*var, v);
#else
        static_assert(sizeof(T) == 0, "unsupported reduction type on the GPU backend");
#endif
    }
}

namespace paras_reduction_detail {
template <typename R>
struct runner;
template <typename T, typename Op, std::size_t E, bool U>
struct runner<reduction_impl<T, Op, 0, E, U>> {
    static constexpr bool has_identity = U || has_known_identity_v<Op, T>;
    using reducer_t = reducer<T, Op, 0, has_identity>;
    slot<T> s;
    reducer_t red;
    PARAS_KERNEL_HD explicit runner(const reduction_impl<T, Op, 0, E, U>& r)
        : s{r.identity_, r.identity_known_}, red(&s, r.identity_, r.op_) {}
    PARAS_KERNEL_HD void merge(const reduction_impl<T, Op, 0, E, U>& r) {
        if (s.has)
            paras_reduction_merge(r.var_, s.value, r.op_);
    }
};
template <typename T, typename Op, std::size_t E, bool U>
struct runner<reduction_impl<T, Op, 1, E, U>> {
    static constexpr bool has_identity = U || has_known_identity_v<Op, T>;
    using reducer_t = reducer<T, Op, 1, has_identity>;
    slot<T> s[E];
    reducer_t red;
    PARAS_KERNEL_HD explicit runner(const reduction_impl<T, Op, 1, E, U>& r)
        : red(s, r.identity_, r.op_) {
        for (std::size_t i = 0; i < E; ++i)
            s[i] = slot<T>{r.identity_, r.identity_known_};
    }
    PARAS_KERNEL_HD void merge(const reduction_impl<T, Op, 1, E, U>& r) {
        for (std::size_t i = 0; i < E; ++i)
            if (s[i].has)
                paras_reduction_merge(r.var_ + i, s[i].value, r.op_);
    }
};
} // namespace paras_reduction_detail

} // namespace sycl

#endif
