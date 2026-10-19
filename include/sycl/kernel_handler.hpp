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

#include <cstddef>
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

template <auto &SpecName> PARAS_KERNEL_HD constexpr unsigned long long spec_const_hash() {
  constexpr const char *s = __PRETTY_FUNCTION__;
  unsigned long long h = 1469598103934665603ull;
  for (const char *c = s; *c; ++c)
    h = (h ^ static_cast<unsigned char>(*c)) * 1099511628211ull;
  return h;
}

struct spec_const_blob {
  static constexpr unsigned max_entries = 32;
  static constexpr unsigned max_bytes = 2048;
  struct entry {
    unsigned long long hash;
    unsigned offset;
    unsigned size;
  };
  unsigned count;
  unsigned used;
  entry e[max_entries];
  unsigned char data[max_bytes];
};

inline std::map<const void *, std::vector<unsigned char>> &spec_const_store() {
  static std::map<const void *, std::vector<unsigned char>> store;
  return store;
}
inline std::mutex &spec_const_mutex() {
  static std::mutex m;
  return m;
}

inline std::map<const void *, unsigned long long> &spec_const_keys() {
  static std::map<const void *, unsigned long long> keys;
  return keys;
}

inline unsigned long long &spec_const_generation() {
  static unsigned long long g = 0;
  return g;
}
template <auto &SpecName> void spec_const_register_key() {
  spec_const_keys()[&SpecName] = spec_const_hash<SpecName>();
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
    ++spec_const_generation();
    v.clear();
  }
};

inline void spec_const_set_raw(const void *key,
                               const std::vector<unsigned char> &bytes) {
  std::lock_guard<std::mutex> lock(spec_const_mutex());
  spec_const_store()[key] = bytes;
  ++spec_const_generation();
}

template <auto &SpecName, typename T> void spec_const_set_host(const T &v) {
  std::vector<unsigned char> bytes(sizeof(T));
  std::memcpy(bytes.data(), static_cast<const void *>(&v), sizeof(T));
  std::lock_guard<std::mutex> lock(spec_const_mutex());
  spec_const_register_key<SpecName>();
  spec_const_store()[&SpecName] = std::move(bytes);
  ++spec_const_generation();
}

inline void spec_const_fill_blob(spec_const_blob &b) {
  b.count = 0;
  b.used = 0;
  std::lock_guard<std::mutex> lock(spec_const_mutex());
  for (const auto &kv : spec_const_store()) {
    auto key = spec_const_keys().find(kv.first);
    const unsigned n = static_cast<unsigned>(kv.second.size());
    if (key == spec_const_keys().end() || b.count == spec_const_blob::max_entries ||
        b.used + n > spec_const_blob::max_bytes)
      continue;
    b.e[b.count] = {key->second, b.used, n};
    std::memcpy(b.data + b.used, kv.second.data(), n);
    b.used += (n + 7u) & ~7u;
    ++b.count;
  }
}
}

#if PARAS_GPU_BACKEND
namespace detail {
namespace {

__attribute__((device)) spec_const_blob paras_spec_const_table;
}
#ifdef __CLANG_RDC__

inline std::vector<const void *> &spec_const_tables() {
  static std::vector<const void *> tables;
  return tables;
}
#if !PARAS_DEVICE_PASS
namespace {
[[maybe_unused]] const int paras_spec_const_table_registered =
    (spec_const_tables().push_back(static_cast<const void *>(&paras_spec_const_table)), 0);
}
#endif
#endif
}
#endif

class kernel_handler {
public:
  template <auto &SpecName>
  PARAS_KERNEL_HD
      typename std::remove_reference_t<decltype(SpecName)>::value_type
      get_specialization_constant() const {
#if PARAS_DEVICE_PASS
    using T = typename std::remove_reference_t<decltype(SpecName)>::value_type;
    constexpr unsigned long long key = detail::spec_const_hash<SpecName>();
    const detail::spec_const_blob &b = detail::paras_spec_const_table;
    for (unsigned i = 0; i < b.count && i < detail::spec_const_blob::max_entries; ++i)
      if (b.e[i].hash == key && b.e[i].size == sizeof(T)) {
        T value = SpecName.paras_default();
        __builtin_memcpy(static_cast<void *>(&value), b.data + b.e[i].offset,
                         sizeof(T));
        return value;
      }
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
