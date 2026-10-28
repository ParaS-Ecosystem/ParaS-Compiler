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
#include <memory>
#include <vector>
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
    std::get<I>(ds).store(lin, r);
  }
}

template <int D, typename Func, typename... Descs> struct red_range_kernel {
  using paras_inner = Func;
  Func f;
  range<D> r;
  std::tuple<Descs...> ds;
  PARAS_KERNEL_HD void operator()(item<D> it) const {
    red_call<0>(f, it, it.get_id(), ds, red_linear(it.get_id(), r));
  }
};

template <int D, typename Func, typename... Descs> struct red_nd_kernel {
  using paras_inner = Func;
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

template <int D, typename Func, typename Tuple> struct red_range_kernel_of;
template <int D, typename Func, typename... Descs>
struct red_range_kernel_of<D, Func, std::tuple<Descs...>> {
  using type = red_range_kernel<D, Func, Descs...>;
};

template <typename T> struct red_init_kernel {
  T *slots;
  bool *touched;
  T identity;
  PARAS_KERNEL_HD void operator()(item<1> it) const {
    const std::size_t i = it.get_id(0);
    if (touched != nullptr)
      touched[i] = false;
    else
      slots[i] = identity;
  }
};

template <typename T, typename Op> struct red_fold_kernel {
  const T *src;
  const bool *stouch;
  T *dst;
  bool *dtouch;
  std::size_t n, m, count;
  Op op;
  T identity;
  PARAS_KERNEL_HD void operator()(item<1> it) const {
    const std::size_t j = it.get_id(0);
    const std::size_t t = j / count, e = j - t * count;
    if (stouch == nullptr) {
      T acc = identity;
      for (std::size_t w = t; w < n; w += m)
        acc = op(acc, src[w * count + e]);
      dst[j] = acc;
    } else {
      T acc = identity;
      bool have = false;
      for (std::size_t w = t; w < n; w += m) {
        const std::size_t k = w * count + e;
        if (!stouch[k])
          continue;
        acc = have ? op(acc, src[k]) : src[k];
        have = true;
      }
      if (have)
        dst[j] = acc;
      dtouch[j] = have;
    }
  }
};

struct red_scratch_block {
  void *ptr = nullptr;
  std::size_t bytes = 0;
  usm::alloc kind = usm::alloc::device;
  device dev;
  bool valid = false;
};
inline void *red_scratch(red_scratch_block &b, usm::alloc kind, std::size_t bytes,
                         const device &dev) {
  if (b.valid && b.bytes >= bytes && b.kind == kind && b.dev == dev)
    return b.ptr;
  if (b.valid)
    usm_raw_free(b.ptr);
  b.valid = false;
  std::size_t want = bytes < 4096 ? 4096 : bytes;
  if (b.bytes > want && b.dev == dev)
    want = b.bytes;
  b.ptr = usm_raw_alloc(kind, want, 256, dev);
  if (b.ptr == nullptr)
    throw exception(make_error_code(errc::memory_allocation),
                    "reduction: cannot allocate scratch memory");
  b.bytes = want;
  b.kind = kind;
  b.dev = dev;
  b.valid = true;
  return b.ptr;
}

inline red_scratch_block &red_scratch_slot(std::size_t which) {
  static thread_local red_scratch_block *blocks = nullptr;
  constexpr std::size_t kMax = 3 * 8;
  if (blocks == nullptr)
    blocks = new red_scratch_block[kMax];
  if (which >= kMax)
    throw exception(make_error_code(errc::feature_not_supported),
                    "reduction: at most 8 reductions per parallel_for");
  return blocks[which];
}

constexpr std::size_t red_align_up(std::size_t v) { return (v + 255) & ~std::size_t(255); }

inline bool red_never() {
  static volatile bool never = false;
  return never;
}

template <typename Handler, typename T, typename Op, bool S>
void red_prepare(Handler &h, reduction_desc<T, Op, S> &d, std::size_t idx,
                 std::size_t items, const device &dev) {
  const std::size_t n = items * d.count;
  const bool need_touched = !d.has_identity;
  const std::size_t vbytes = red_align_up(n * sizeof(T));
  char *base = static_cast<char *>(red_scratch(
      red_scratch_slot(3 * idx), usm::alloc::device,
      vbytes + (need_touched ? red_align_up(n) : 0), dev));
  d.slots = reinterpret_cast<T *>(base);
  d.touched = need_touched ? reinterpret_cast<bool *>(base + vbytes) : nullptr;
  if constexpr (S) {
    if (red_never())
      h.template parallel_for<void>(range<1>(1),
                                    red_init_kernel<T>{d.slots, d.touched, d.identity});
    h.paras_internal_range(n, red_init_kernel<T>{d.slots, d.touched, d.identity});
  }
}

template <typename T, typename Op, bool S>
void red_apply(reduction_desc<T, Op, S> &d, const T *res, const bool *got) {
  for (std::size_t e = 0; e < d.count; ++e) {
    const bool have = got ? got[e] : true;
    if (d.init_to_identity) {
      if (have)
        d.target[e] = res[e];
      else if (d.has_identity)
        d.target[e] = d.identity;
    } else if (have) {
      d.target[e] = d.op(d.target[e], res[e]);
    }
  }
}

template <typename Handler, typename T, typename Op, bool S>
void red_finish(Handler &h, reduction_desc<T, Op, S> &d, std::size_t idx,
                std::size_t items, const device &dev) {
  const std::size_t count = d.count;
  const std::size_t fan = dev.is_gpu() ? 32 : 256;
  const bool tch = d.touched != nullptr;

  const std::size_t m1 = (items + fan - 1) / fan;
  T *a = d.slots;
  bool *at = d.touched;
  T *b = nullptr;
  bool *bt = nullptr;
  if (m1 > 1) {
    const std::size_t nb = m1 * count;
    const std::size_t vbytes = red_align_up(nb * sizeof(T));
    char *base = static_cast<char *>(red_scratch(red_scratch_slot(3 * idx + 1),
                                                 usm::alloc::device,
                                                 vbytes + (tch ? red_align_up(nb) : 0), dev));
    b = reinterpret_cast<T *>(base);
    bt = tch ? reinterpret_cast<bool *>(base + vbytes) : nullptr;
  }
  const std::size_t rbytes = red_align_up(count * sizeof(T));
  char *rbase = static_cast<char *>(red_scratch(red_scratch_slot(3 * idx + 2),
                                                usm::alloc::shared,
                                                rbytes + red_align_up(count), dev));
  T *res = reinterpret_cast<T *>(rbase);
  bool *rest = reinterpret_cast<bool *>(rbase + rbytes);
  if (red_never())
    h.template parallel_for<void>(range<1>(1),
                                  red_fold_kernel<T, Op>{a, at, res, rest, 1, 1, count, d.op,
                                                         d.identity});
  std::size_t n = items;
  bool in_a = true;
  for (;;) {
    const std::size_t m = (n + fan - 1) / fan;
    T *src = in_a ? a : b;
    bool *stc = in_a ? at : bt;
    T *dst = m == 1 ? res : (in_a ? b : a);
    bool *dtc = tch ? (m == 1 ? rest : (in_a ? bt : at)) : nullptr;
    h.paras_internal_range(m * count,
                           red_fold_kernel<T, Op>{src, stc, dst, dtc, n, m, count, d.op,
                                                  d.identity});
    n = m;
    in_a = !in_a;
    if (m == 1)
      break;
  }
  h.paras_reduction_wait();
  red_apply(d, res, tch ? rest : nullptr);
  d.slots = nullptr;
  d.touched = nullptr;
}

template <typename T, typename Op, bool S> void red_empty(reduction_desc<T, Op, S> &d) {
  if (d.init_to_identity && d.has_identity)
    for (std::size_t e = 0; e < d.count; ++e)
      d.target[e] = d.identity;
}

#if !PARAS_GPU_BACKEND

template <int D> inline id<D> red_delinear(std::size_t lin, const range<D> &r) {
  id<D> i;
  for (int d = D - 1; d >= 0; --d) {
    i[d] = lin % r[d];
    lin /= r[d];
  }
  return i;
}

template <typename T> struct red_host_part {
  std::unique_ptr<T[]> vals;
  std::unique_ptr<bool[]> touched;
};
template <typename T, typename Op, bool S>
red_host_part<T> red_host_part_for(const reduction_desc<T, Op, S> &d, std::size_t chunks) {
  red_host_part<T> p;
  p.vals.reset(new T[chunks * d.count]);
  for (std::size_t k = 0; k < chunks * d.count; ++k)
    p.vals[k] = d.identity;
  p.touched.reset(new bool[chunks * d.count]());
  return p;
}

template <std::size_t I, typename Func, int D, typename Descs, typename Parts,
          typename... Rs>
void red_host_chunk(const Func &f, const range<D> &r, const Descs &ds, Parts &ps,
                    std::size_t c, std::size_t s, std::size_t e, Rs &...rs) {
  if constexpr (I == std::tuple_size_v<Descs>) {
    for (std::size_t lin = s; lin < e; ++lin) {
      const id<D> i = red_delinear(lin, r);
      if constexpr (std::is_invocable_v<const Func &, item<D>, Rs &...>)
        f(item<D>(i, r), rs...);
      else
        f(i, rs...);
    }
  } else {
    const auto &d = std::get<I>(ds);
    auto &p = std::get<I>(ps);
    using DT = std::decay_t<decltype(d)>;
    using T = typename DT::value_type;
    using Op = typename DT::binary_operation;
    if constexpr (std::is_same_v<decltype(d.make_reducer(0)), reducer<T, Op, 1>>) {
      reducer<T, Op, 1> rr(p.vals.get() + c * d.count,
                           d.has_identity ? nullptr : p.touched.get() + c * d.count, d.op,
                           d.identity);
      red_host_chunk<I + 1>(f, r, ds, ps, c, s, e, rs..., rr);
      if (d.has_identity)
        for (std::size_t k = 0; k < d.count; ++k)
          p.touched[c * d.count + k] = true;
    } else {
      reducer<T, Op, 0> rr(d.op, d.identity);
      red_host_chunk<I + 1>(f, r, ds, ps, c, s, e, rs..., rr);
      p.vals[c] = rr.paras_value();
      p.touched[c] = rr.paras_touched();
    }
  }
}

template <typename T, typename Op, bool S>
void red_host_finish(reduction_desc<T, Op, S> &d, red_host_part<T> &p, std::size_t chunks) {
  std::unique_ptr<T[]> res(new T[d.count]);
  for (std::size_t e = 0; e < d.count; ++e)
    res[e] = d.identity;
  std::unique_ptr<bool[]> got(new bool[d.count]());
  for (std::size_t c = 0; c < chunks; ++c)
    for (std::size_t e = 0; e < d.count; ++e) {
      const std::size_t k = c * d.count + e;
      if (!p.touched[k])
        continue;
      res[e] = got[e] ? d.op(res[e], p.vals[k]) : p.vals[k];
      got[e] = true;
    }
  red_apply(d, res.get(), got.get());
}
template <typename Descs, typename Parts, std::size_t... I>
void red_host_finish_all(Descs &ds, Parts &ps, std::size_t chunks, std::index_sequence<I...>) {
  (red_host_finish(std::get<I>(ds), std::get<I>(ps), chunks), ...);
}
#endif

template <typename Tuple, std::size_t... I>
auto red_front(Tuple &&t, std::index_sequence<I...>) {
  return std::make_tuple(std::decay_t<std::tuple_element_t<I, std::decay_t<Tuple>>>(
      std::get<I>(t))...);
}

template <typename Handler, typename Descs, std::size_t... I>
void red_prepare_all(Handler &h, Descs &ds, std::size_t items, const device &dev,
                     std::index_sequence<I...>) {
  (red_prepare(h, std::get<I>(ds), I, items, dev), ...);
}
template <typename Handler, typename Descs, std::size_t... I>
void red_finish_all(Handler &h, Descs &ds, std::size_t items, const device &dev,
                    std::index_sequence<I...>) {
  (red_finish(h, std::get<I>(ds), I, items, dev), ...);
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
  if (items == 0) {
    std::apply([](auto &...d) { (detail::red_empty(d), ...); }, ds);
    return;
  }
#if !PARAS_GPU_BACKEND
  {
    using K = typename detail::red_range_kernel_of<dim, decltype(f), decltype(ds)>::type;
    paras_launch_kernel<KernelName, K>();
    paras_check_range_kernel<K>();
    wait_for_dependencies();
    std::size_t chunks = paras_host_detail::host_core_count();
    if (chunks > items)
      chunks = items;
    if (chunks > paras_host_detail::host_pool::kMaxChunks)
      chunks = paras_host_detail::host_pool::kMaxChunks;
    auto parts = std::apply(
        [&](auto &...d) { return std::make_tuple(detail::red_host_part_for(d, chunks)...); },
        ds);
    const std::size_t base = items / chunks, extra = items % chunks;
    paras_host_detail::host_pool::instance().run_chunks(chunks, [&](std::size_t c) {
      const std::size_t s = c * base + (c < extra ? c : extra);
      detail::red_host_chunk<0>(f, r, ds, parts, c, s, s + base + (c < extra ? 1 : 0));
    });
    detail::red_host_finish_all(ds, parts, chunks, std::make_index_sequence<N - 1>{});
    return;
  }
#endif
  detail::red_prepare_all(*this, ds, items, dev, std::make_index_sequence<N - 1>{});
  std::apply(
      [&](auto &...d) {
        parallel_for<KernelName>(
            r, detail::red_range_kernel<dim, decltype(f), std::decay_t<decltype(d)>...>{
                   f, r, std::make_tuple(d...)});
      },
      ds);
  detail::red_finish_all(*this, ds, items, dev, std::make_index_sequence<N - 1>{});
}

template <typename KernelName, int dim, typename First, typename... Rest, typename>
void handler::parallel_for(const nd_range<dim> &r, First &&first, Rest &&...rest) {
  auto all = std::forward_as_tuple(first, rest...);
  constexpr std::size_t N = 1 + sizeof...(Rest);
  auto f = std::get<N - 1>(all);
  auto ds = detail::red_front(all, std::make_index_sequence<N - 1>{});
  const std::size_t items = r.get_global_range().size();
  device dev = paras_reduction_device();
  if (items == 0) {
    std::apply([](auto &...d) { (detail::red_empty(d), ...); }, ds);
    return;
  }
  detail::red_prepare_all(*this, ds, items, dev, std::make_index_sequence<N - 1>{});
  std::apply(
      [&](auto &...d) {
        parallel_for<KernelName>(
            r, detail::red_nd_kernel<dim, decltype(f), std::decay_t<decltype(d)>...>{
                   f, std::make_tuple(d...)});
      },
      ds);
  detail::red_finish_all(*this, ds, items, dev, std::make_index_sequence<N - 1>{});
}

}

#endif
