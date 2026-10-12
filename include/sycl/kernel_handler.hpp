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

#ifndef __PARAS_KERNEL_HANDLER_HPP__
#define __PARAS_KERNEL_HANDLER_HPP__

#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename T> class specialization_id {
public:
  using value_type = T;

  template <typename... Args>
  explicit constexpr specialization_id(Args &&...args)
      : default_value_(std::forward<Args>(args)...) {}

  specialization_id(const specialization_id &) = delete;
  specialization_id(specialization_id &&) = delete;
  specialization_id &operator=(const specialization_id &) = delete;
  specialization_id &operator=(specialization_id &&) = delete;

  PARAS_KERNEL_HD constexpr const T &paras_default() const {
    return default_value_;
  }

private:
  T default_value_;
};

namespace detail {

inline std::map<const void *, std::vector<unsigned char>> &spec_const_store() {
  static std::map<const void *, std::vector<unsigned char>> store;
  return store;
}
inline std::mutex &spec_const_mutex() {
  static std::mutex m;
  return m;
}

template <auto &SpecName> auto spec_const_value_host() {
  using T = typename std::remove_reference_t<decltype(SpecName)>::value_type;
  std::lock_guard<std::mutex> lock(spec_const_mutex());
  auto it = spec_const_store().find(&SpecName);
  if (it == spec_const_store().end())
    return SpecName.paras_default();

  T value = SpecName.paras_default();
  std::memcpy(static_cast<void *>(&value), it->second.data(), sizeof(T));
  return value;
}

struct spec_const_restorer {
  struct saved {
    const void *key;
    bool had;
    std::vector<unsigned char> bytes;
  };
  std::vector<saved> v;

  spec_const_restorer() = default;
  spec_const_restorer(const spec_const_restorer &) = default;
  spec_const_restorer(spec_const_restorer &&o) noexcept : v(std::move(o.v)) {
    o.v.clear();
  }
  spec_const_restorer &operator=(const spec_const_restorer &) = default;
  spec_const_restorer &operator=(spec_const_restorer &&o) noexcept {
    v = std::move(o.v);
    o.v.clear();
    return *this;
  }
  ~spec_const_restorer() { restore(); }

  void remember(const void *key) {
    for (auto &e : v)
      if (e.key == key)
        return;
    std::lock_guard<std::mutex> lock(spec_const_mutex());
    auto it = spec_const_store().find(key);
    if (it == spec_const_store().end())
      v.push_back({key, false, {}});
    else
      v.push_back({key, true, it->second});
  }
  void restore() {
    if (v.empty())
      return;
    std::lock_guard<std::mutex> lock(spec_const_mutex());
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
      if (it->had)
        spec_const_store()[it->key] = it->bytes;
      else
        spec_const_store().erase(it->key);
    }
    v.clear();
  }
};

template <auto &SpecName, typename T> void spec_const_set_host(const T &v) {
  std::vector<unsigned char> bytes(sizeof(T));
  std::memcpy(bytes.data(), static_cast<const void *>(&v), sizeof(T));
  std::lock_guard<std::mutex> lock(spec_const_mutex());
  spec_const_store()[&SpecName] = std::move(bytes);
}
}

class kernel_handler {
public:
  template <auto &SpecName>
  PARAS_KERNEL_HD
      typename std::remove_reference_t<decltype(SpecName)>::value_type
      get_specialization_constant() const {
#if PARAS_DEVICE_PASS
    return SpecName.paras_default();
#else
    return detail::spec_const_value_host<SpecName>();
#endif
  }
};

namespace detail {

template <typename Func, typename Arg>
PARAS_KERNEL_HD inline void invoke_with_kh(const Func &f, const Arg &arg) {
  if constexpr (std::is_invocable_v<const Func &, Arg, kernel_handler>)
    f(arg, kernel_handler{});
  else
    f(arg);
}
}

}

#endif
