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

#ifndef __PARAS_REDUCTION_IMPL_HPP__
#define __PARAS_REDUCTION_IMPL_HPP__

#include <cstring>
#include <tuple>
#include <utility>

#include "kem/usm_core.hpp"
#include "sycl/reduction.hpp"

namespace sycl {
namespace detail {

template <int D>
PARAS_KERNEL_HD inline std::size_t red_linear(const id<D> &i, const range<D> &r) {
  std::size_t lin = 0;
  for (int d = 0; d < D; ++d)
    lin = lin * r[d] + i[d];
  return lin;
}

template <std::size_t I, typename Func, typename Item, typename Alt, typename Descs,
          typename... Rs>
PARAS_KERNEL_HD inline void red_call(const Func &f, const Item &it, const Alt &alt,
                                     const Descs &ds, std::size_t lin, Rs &...rs) {
  if constexpr (I == std::tuple_size_v<Descs>) {
    if constexpr (std::is_invocable_v<const Func &, Item, Rs &...>)
      f(it, rs...);
    else
      f(alt, rs...);
  } else {
    auto r = std::get<I>(ds).make_reducer(lin);
    red_call<I + 1>(f, it, alt, ds, lin, rs..., r);
  }
}

template <int D, typename Func, typename... Descs> struct red_range_kernel {
  Func f;
  range<D> r;
  std::tuple<Descs...> ds;
  PARAS_KERNEL_HD void operator()(item<D> it) const {
    red_call<0>(f, it, it.get_id(), ds, red_linear(it.get_id(), r));
  }
};

template <int D, typename Func, typename... Descs> struct red_nd_kernel {
  Func f;
  std::tuple<Descs...> ds;
  PARAS_KERNEL_HD void operator()(nd_item<D> it) const {
    id<D> g;
    range<D> gr;
    for (int d = 0; d < D; ++d) {
      g[d] = it.get_global_id(d);
      gr[d] = it.get_global_range(d);
    }
    red_call<0>(f, it, it, ds, red_linear(g, gr));
  }
};

template <typename T, typename Op, bool S>
void red_prepare(reduction_desc<T, Op, S> &d, std::size_t items, const device &dev) {
  const std::size_t n = items * d.count;
  d.slots = static_cast<T *>(
      usm_raw_alloc(usm::alloc::shared, n * sizeof(T), alignof(T), dev));
  d.touched = static_cast<bool *>(
      usm_raw_alloc(usm::alloc::shared, n * sizeof(bool), alignof(bool), dev));
  if (d.slots == nullptr || d.touched == nullptr)
    throw exception(make_error_code(errc::memory_allocation),
                    "reduction: cannot allocate scratch memory");
  std::memset(static_cast<void *>(d.touched), 0, n * sizeof(bool));
}

template <typename T, typename Op, bool S>
void red_finish(reduction_desc<T, Op, S> &d, std::size_t items) {
  for (std::size_t e = 0; e < d.count; ++e) {
    T acc{};
    bool have = false;
    if (!d.init_to_identity) {
      acc = d.target[e];
      have = true;
    } else if (d.has_identity) {
      acc = d.identity;
      have = true;
    }
    for (std::size_t w = 0; w < items; ++w) {
      const std::size_t k = w * d.count + e;
      if (!d.touched[k])
        continue;
      acc = have ? d.op(acc, d.slots[k]) : d.slots[k];
      have = true;
    }
    if (have)
      d.target[e] = acc;
  }
  usm_raw_free(d.slots);
  usm_raw_free(d.touched);
  d.slots = nullptr;
  d.touched = nullptr;
}

template <typename Tuple, std::size_t... I>
auto red_front(Tuple &&t, std::index_sequence<I...>) {
  return std::make_tuple(std::decay_t<std::tuple_element_t<I, std::decay_t<Tuple>>>(
      std::get<I>(t))...);
}

}

template <typename KernelName, int dim, typename First, typename... Rest, typename>
void handler::parallel_for(range<dim> r, First &&first, Rest &&...rest) {
  auto all = std::forward_as_tuple(first, rest...);
  constexpr std::size_t N = 1 + sizeof...(Rest);
  auto f = std::get<N - 1>(all);
  auto ds = detail::red_front(all, std::make_index_sequence<N - 1>{});
  const std::size_t items = r.size();
  device dev = paras_reduction_device();
  std::apply([&](auto &...d) { (detail::red_prepare(d, items, dev), ...); }, ds);
  std::apply(
      [&](auto &...d) {
        parallel_for<KernelName>(
            r, detail::red_range_kernel<dim, decltype(f), std::decay_t<decltype(d)>...>{
                   f, r, std::make_tuple(d...)});
      },
      ds);
  paras_reduction_wait();
  std::apply([&](auto &...d) { (detail::red_finish(d, items), ...); }, ds);
}

template <typename KernelName, int dim, typename First, typename... Rest, typename>
void handler::parallel_for(const nd_range<dim> &r, First &&first, Rest &&...rest) {
  auto all = std::forward_as_tuple(first, rest...);
  constexpr std::size_t N = 1 + sizeof...(Rest);
  auto f = std::get<N - 1>(all);
  auto ds = detail::red_front(all, std::make_index_sequence<N - 1>{});
  const std::size_t items = r.get_global_range().size();
  device dev = paras_reduction_device();
  std::apply([&](auto &...d) { (detail::red_prepare(d, items, dev), ...); }, ds);
  std::apply(
      [&](auto &...d) {
        parallel_for<KernelName>(
            r, detail::red_nd_kernel<dim, decltype(f), std::decay_t<decltype(d)>...>{
                   f, std::make_tuple(d...)});
      },
      ds);
  paras_reduction_wait();
  std::apply([&](auto &...d) { (detail::red_finish(d, items), ...); }, ds);
}

}

#endif
