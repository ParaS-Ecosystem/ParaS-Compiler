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

#ifndef __PARAS_KERNEL_REGISTRY_HPP__
#define __PARAS_KERNEL_REGISTRY_HPP__

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "aspect.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {
namespace detail {

struct kernel_props {
  std::vector<aspect> aspects;
  int wg_dims = 0;
  std::size_t wg[3] = {0, 0, 0};
  std::size_t sg = 0;
};

template <aspect... As> struct kp_device_has {};
template <std::size_t... Ns> struct kp_work_group_size {};
template <std::size_t N> struct kp_sub_group_size {};
template <typename... Ps> struct kp_list {};

template <typename P> struct kp_apply;
template <aspect... As> struct kp_apply<kp_device_has<As...>> {
  static void to(kernel_props &p) { (p.aspects.push_back(As), ...); }
};
template <std::size_t... Ns> struct kp_apply<kp_work_group_size<Ns...>> {
  static void to(kernel_props &p) {
    static_assert(sizeof...(Ns) >= 1 && sizeof...(Ns) <= 3,
                  "reqd_work_group_size takes 1 to 3 values");
    p.wg_dims = static_cast<int>(sizeof...(Ns));
    const std::size_t v[] = {Ns...};
    for (std::size_t i = 0; i < sizeof...(Ns); ++i)
      p.wg[i] = v[i];
  }
};
template <std::size_t N> struct kp_apply<kp_sub_group_size<N>> {
  static void to(kernel_props &p) { p.sg = N; }
};
template <typename... Ps> struct kp_apply<kp_list<Ps...>> {
  static void to(kernel_props &p) { (kp_apply<Ps>::to(p), ...); }
};

template <typename F, typename = void> struct kernel_props_of {
  static kernel_props get() { return {}; }
};
template <typename F, typename = void> struct has_paras_inner : std::false_type {};
template <typename F>
struct has_paras_inner<F, std::void_t<typename F::paras_inner>> : std::true_type {};
template <typename F>
struct kernel_props_of<F, std::enable_if_t<!has_paras_inner<F>::value,
                                           std::void_t<typename F::paras_kernel_props>>> {
  static kernel_props get() {
    kernel_props p;
    kp_apply<typename F::paras_kernel_props>::to(p);
    return p;
  }
};

template <typename F>
struct kernel_props_of<F, std::enable_if_t<has_paras_inner<F>::value>>
    : kernel_props_of<typename F::paras_inner> {};

struct kernel_info {
  const void *key;
  std::string name;
  kernel_props props;
};

template <typename T> struct type_key {
  static constexpr char value = 0;
};

template <typename T> std::string type_name() {
#if defined(__clang__) || defined(__GNUC__)
  std::string s = __PRETTY_FUNCTION__;
  const std::string marker = "T = ";
  std::size_t b = s.find(marker);
  if (b == std::string::npos)
    return s;
  b += marker.size();
  std::size_t e = s.find_first_of(";]", b);
  return s.substr(b, e == std::string::npos ? std::string::npos : e - b);
#else
  return "kernel";
#endif
}

struct kernel_registry {
  std::mutex m;
  std::vector<const kernel_info *> order;
  std::map<const void *, const kernel_info *> by_key;
};
inline kernel_registry &kernel_registry_instance() {
  static kernel_registry *r = new kernel_registry;
  return *r;
}

inline const kernel_info *register_kernel(const void *key, std::string name,
                                          kernel_props props) {
  auto &r = kernel_registry_instance();
  std::lock_guard<std::mutex> lock(r.m);
  auto it = r.by_key.find(key);
  if (it != r.by_key.end())
    return it->second;
  auto *ki = new kernel_info{key, std::move(name), std::move(props)};
  r.by_key.emplace(key, ki);
  r.order.push_back(ki);
  return ki;
}

inline const kernel_info *find_kernel(const void *key) {
  auto &r = kernel_registry_instance();
  std::lock_guard<std::mutex> lock(r.m);
  auto it = r.by_key.find(key);
  return it == r.by_key.end() ? nullptr : it->second;
}

inline std::vector<const kernel_info *> all_kernels() {
  auto &r = kernel_registry_instance();
  std::lock_guard<std::mutex> lock(r.m);
  return r.order;
}

struct paras_internal_kernel;

template <typename KernelName, typename Func>
using kernel_key_t =
    std::conditional_t<std::is_void_v<KernelName>, Func, KernelName>;

template <typename KernelName, typename Func> struct kernel_registrar {
  static inline const kernel_info *const info = register_kernel(
      &type_key<kernel_key_t<KernelName, Func>>::value,
      type_name<kernel_key_t<KernelName, Func>>(),
      kernel_props_of<Func>::get());
};

template <typename KernelName, typename Func>
inline const kernel_info *note_kernel() {
  if constexpr (std::is_same_v<KernelName, paras_internal_kernel>)
    return nullptr;
  else
    return kernel_registrar<KernelName, Func>::info;
}

template <typename Props, typename F> struct kernel_with_props {
  using paras_kernel_props = Props;
  F f;
  template <typename... Args>
  PARAS_KERNEL_HD auto operator()(Args &&...args) const
      -> decltype(f(std::forward<Args>(args)...)) {
    return f(std::forward<Args>(args)...);
  }
};
template <typename Props, typename F>
inline kernel_with_props<Props, F> with_kernel_props(F f) {
  return {std::move(f)};
}

struct bundle_impl;

}
}

#endif
