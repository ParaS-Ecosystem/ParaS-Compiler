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

#ifndef __PARAS_STREAM_HPP__
#define __PARAS_STREAM_HPP__

#include "data_types.hpp"
#include "half.hpp"
#include "id.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "item.hpp"
#include "h_item.hpp"
#include "group.hpp"
#include "nd_item.hpp"
#include "property_list.hpp"
#include "kem_gpu/gpu_utilities.hpp"
#include <cstddef>
#include <functional>
#include <cstdio>
#include <type_traits>
#if !PARAS_DEVICE_PASS
#include <iostream>
#include <mutex>
#include <string>
#endif

namespace sycl {

class handler;

enum class stream_manipulator {
  dec, hex, oct, noshowbase, showbase, noshowpos, showpos, endl, flush,
  fixed, scientific, hexfloat, defaultfloat
};

inline constexpr stream_manipulator dec = stream_manipulator::dec;
inline constexpr stream_manipulator hex = stream_manipulator::hex;
inline constexpr stream_manipulator oct = stream_manipulator::oct;
inline constexpr stream_manipulator noshowbase = stream_manipulator::noshowbase;
inline constexpr stream_manipulator showbase = stream_manipulator::showbase;
inline constexpr stream_manipulator noshowpos = stream_manipulator::noshowpos;
inline constexpr stream_manipulator showpos = stream_manipulator::showpos;
inline constexpr stream_manipulator endl = stream_manipulator::endl;
inline constexpr stream_manipulator flush = stream_manipulator::flush;
inline constexpr stream_manipulator fixed = stream_manipulator::fixed;
inline constexpr stream_manipulator scientific = stream_manipulator::scientific;
inline constexpr stream_manipulator hexfloat = stream_manipulator::hexfloat;
inline constexpr stream_manipulator defaultfloat =
    stream_manipulator::defaultfloat;

namespace detail {
struct stream_precision_manipulator { int precision; };
struct stream_width_manipulator { int width; };

struct stream_format {
  int base = 10;
  bool show_base = false;
  bool show_pos = false;
  char float_conv = 'g';
  int precision = -1;
  int width = -1;
};

#if !PARAS_DEVICE_PASS
struct stream_host_buffer {
  std::string text;
  stream_format fmt;
  void write_out() {
    if (text.empty())
      return;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    std::cout << text << std::flush;
    text.clear();
  }
  ~stream_host_buffer() { write_out(); }
};
inline stream_host_buffer &stream_host() {
  thread_local stream_host_buffer b;
  return b;
}
#endif
}

PARAS_KERNEL_HD inline detail::stream_precision_manipulator setprecision(int precision) {
  return {precision};
}
PARAS_KERNEL_HD inline detail::stream_width_manipulator setw(int width) { return {width}; }

class stream {
  std::size_t size_ = 0;
  std::size_t work_item_size_ = 0;

public:
  stream(std::size_t totalBufferSize, std::size_t workItemBufferSize,
         handler &, const property_list &propList = {})
      : size_(totalBufferSize), work_item_size_(workItemBufferSize) {
    (void)propList;
  }

  PARAS_KERNEL_HD std::size_t size() const noexcept { return size_; }
  PARAS_KERNEL_HD std::size_t get_size() const { return size_; }
  PARAS_KERNEL_HD std::size_t get_work_item_buffer_size() const {
    return work_item_size_;
  }
  PARAS_KERNEL_HD std::size_t get_max_statement_size() const {
    return work_item_size_;
  }

  template <typename PropT> bool has_property() const noexcept {
    return false;
  }
  template <typename PropT> PropT get_property() const {
    throw sycl::exception(make_error_code(errc::invalid),
                          "stream: property not present");
  }

  friend bool operator==(const stream &a, const stream &b) {
    return a.size_ == b.size_ && a.work_item_size_ == b.work_item_size_;
  }
  friend bool operator!=(const stream &a, const stream &b) {
    return !(a == b);
  }

  PARAS_KERNEL_HD detail::stream_format paras_fmt() const {
#if PARAS_DEVICE_PASS
    return detail::stream_format{};
#else
    return detail::stream_host().fmt;
#endif
  }
  template <typename F> PARAS_KERNEL_HD void paras_update_fmt(F f) const {
#if PARAS_DEVICE_PASS
    (void)f;
#else
    f(detail::stream_host().fmt);
#endif
  }

  static PARAS_KERNEL_HD int paras_put_int(char *spec, int n, int v) {
    char digits[12];
    int k = 0;
    do {
      digits[k++] = static_cast<char>('0' + v % 10);
      v /= 10;
    } while (v > 0 && k < 11);
    while (k > 0)
      spec[n++] = digits[--k];
    return n;
  }

  template <typename V>
  PARAS_KERNEL_HD void paras_put(const char *length, char conv, V v,
                                 bool numeric) const {
    const detail::stream_format f = paras_fmt();
    char spec[32];
    int n = 0;
    spec[n++] = '%';
    if (numeric && f.show_pos)
      spec[n++] = '+';
    if (numeric && f.show_base)
      spec[n++] = '#';
    if (f.width >= 0)
      n = paras_put_int(spec, n, f.width);
    const bool integer = conv == 'd' || conv == 'u' || conv == 'x' || conv == 'o';
    if (numeric && !integer && f.precision >= 0) {
      spec[n++] = '.';
      n = paras_put_int(spec, n, f.precision);
    }
    for (const char *l = length; *l; ++l)
      spec[n++] = *l;
    spec[n++] = conv;
    spec[n] = '\0';
#if PARAS_DEVICE_PASS
    printf(spec, v);
#else
    char buf[512];
    std::snprintf(buf, sizeof buf, spec, v);
    paras_text(buf);
    detail::stream_host().fmt.width = -1;
#endif
  }

  PARAS_KERNEL_HD void paras_text(const char *s) const {
#if PARAS_DEVICE_PASS
    printf("%s", s);
#else
    auto &b = detail::stream_host();
    b.text += s;
    if (work_item_size_ && b.text.size() >= work_item_size_)
      b.write_out();
#endif
  }

  PARAS_KERNEL_HD void paras_flush(bool newline) const {
#if PARAS_DEVICE_PASS
    if (newline)
      printf("\n");
#else
    auto &b = detail::stream_host();
    if (newline)
      b.text += '\n';
    b.write_out();
#endif
  }

  template <typename I> PARAS_KERNEL_HD void paras_integer(I v) const {
    const int base = paras_fmt().base;
    const char conv = base == 16 ? 'x' : base == 8 ? 'o' : (std::is_signed_v<I> ? 'd' : 'u');
    if constexpr (std::is_signed_v<I>) {
      if (base != 10)
        paras_put("ll", conv, static_cast<unsigned long long>(v), true);
      else
        paras_put("ll", conv, static_cast<long long>(v), true);
    } else {
      paras_put("ll", conv, static_cast<unsigned long long>(v), true);
    }
  }

  PARAS_KERNEL_HD void paras_floating(double v) const {
    paras_put("", paras_fmt().float_conv, v, true);
  }
};

namespace detail {
template <typename T> struct stream_is_char : std::false_type {};
template <> struct stream_is_char<char> : std::true_type {};
template <> struct stream_is_char<signed char> : std::true_type {};
template <> struct stream_is_char<unsigned char> : std::true_type {};
}

template <typename T,
          std::enable_if_t<std::is_arithmetic_v<T> || std::is_same_v<T, half>,
                           int> = 0>
PARAS_KERNEL_HD const stream &operator<<(const stream &os, const T &v) {
  if constexpr (std::is_same_v<T, bool>)
    os.paras_text(v ? "true" : "false");
  else if constexpr (detail::stream_is_char<T>::value)
    os.paras_put("", 'c', static_cast<int>(v), false);
  else if constexpr (std::is_integral_v<T>)
    os.paras_integer(v);
  else
  {
    if constexpr (std::is_same_v<T, half>)
      os.paras_floating(static_cast<double>(static_cast<float>(v)));
    else
      os.paras_floating(static_cast<double>(v));
  }
  return os;
}

PARAS_KERNEL_HD inline const stream &operator<<(const stream &os,
                                                const char *s) {
  os.paras_put("", 's', s, false);
  return os;
}
PARAS_KERNEL_HD inline const stream &operator<<(const stream &os, char *s) {
  return os << static_cast<const char *>(s);
}
template <typename T,
          std::enable_if_t<!std::is_same_v<std::remove_cv_t<T>, char>, int> = 0>
PARAS_KERNEL_HD const stream &operator<<(const stream &os, T *p) {
  os.paras_put("", 'p', static_cast<const void *>(p), false);
  return os;
}

PARAS_KERNEL_HD inline const stream &operator<<(const stream &os,
                                                stream_manipulator m) {
  if (m == stream_manipulator::endl) {
    os.paras_flush(true);
    return os;
  }
  if (m == stream_manipulator::flush) {
    os.paras_flush(false);
    return os;
  }
  os.paras_update_fmt([m](detail::stream_format &f) {
    switch (m) {
    case stream_manipulator::dec: f.base = 10; break;
    case stream_manipulator::hex: f.base = 16; break;
    case stream_manipulator::oct: f.base = 8; break;
    case stream_manipulator::noshowbase: f.show_base = false; break;
    case stream_manipulator::showbase: f.show_base = true; break;
    case stream_manipulator::noshowpos: f.show_pos = false; break;
    case stream_manipulator::showpos: f.show_pos = true; break;
    case stream_manipulator::fixed: f.float_conv = 'f'; break;
    case stream_manipulator::scientific: f.float_conv = 'e'; break;
    case stream_manipulator::hexfloat: f.float_conv = 'a'; break;
    case stream_manipulator::defaultfloat: f.float_conv = 'g'; break;
    default: break;
    }
  });
  return os;
}
PARAS_KERNEL_HD inline const stream &
operator<<(const stream &os, detail::stream_precision_manipulator m) {
  const int p = m.precision;
  os.paras_update_fmt([p](detail::stream_format &f) { f.precision = p; });
  return os;
}
PARAS_KERNEL_HD inline const stream &
operator<<(const stream &os, detail::stream_width_manipulator m) {
  const int w = m.width;
  os.paras_update_fmt([w](detail::stream_format &f) { f.width = w; });
  return os;
}

template <typename T, int N>
PARAS_KERNEL_HD const stream &operator<<(const stream &os,
                                         const vec<T, N> &v) {
  os.paras_text("(");
  for (int i = 0; i < N; ++i) {
    if (i)
      os.paras_text(", ");
    os << v[i];
  }
  os.paras_text(")");
  return os;
}

namespace detail {
template <typename A>
PARAS_KERNEL_HD void stream_dims(const stream &os, const A &a, int dims) {
  os.paras_text("{");
  for (int i = 0; i < dims; ++i) {
    if (i)
      os.paras_text(", ");
    os << static_cast<unsigned long long>(a[i]);
  }
  os.paras_text("}");
}
}

template <int D>
PARAS_KERNEL_HD const stream &operator<<(const stream &os, const id<D> &v) {
  detail::stream_dims(os, v, D);
  return os;
}
template <int D>
PARAS_KERNEL_HD const stream &operator<<(const stream &os,
                                         const range<D> &v) {
  detail::stream_dims(os, v, D);
  return os;
}
template <int D>
PARAS_KERNEL_HD const stream &operator<<(const stream &os,
                                         const nd_range<D> &v) {
  os.paras_text("nd_range(global: ");
  os << v.get_global_range();
  os.paras_text(", local: ");
  os << v.get_local_range();
  os.paras_text(")");
  return os;
}
template <int D, bool WithOffset>
PARAS_KERNEL_HD const stream &operator<<(const stream &os,
                                         const item<D, WithOffset> &it) {
  os.paras_text("item(range: ");
  os << it.get_range();
  os.paras_text(", id: ");
  os << it.get_id();
  os.paras_text(")");
  return os;
}
template <int D>
PARAS_KERNEL_HD const stream &operator<<(const stream &os,
                                         const nd_item<D> &it) {
  os.paras_text("nd_item(global id: ");
  os << it.get_global_id();
  os.paras_text(", local id: ");
  os << it.get_local_id();
  os.paras_text(")");
  return os;
}
template <int D>
PARAS_KERNEL_HD const stream &operator<<(const stream &os,
                                         const group<D> &g) {
  os.paras_text("group(id: ");
  os << g.get_group_id();
  os.paras_text(", range: ");
  os << g.get_group_range();
  os.paras_text(")");
  return os;
}
template <int D>
PARAS_KERNEL_HD const stream &operator<<(const stream &os,
                                         const h_item<D> &it) {
  os.paras_text("h_item(global: ");
  os << it.get_global();
  os.paras_text(", local: ");
  os << it.get_local();
  os.paras_text(")");
  return os;
}

}

namespace std {
template <> struct hash<sycl::stream> {
  size_t operator()(const sycl::stream &s) const noexcept {
    return std::hash<size_t>()(s.size() * 31u + s.get_work_item_buffer_size());
  }
};
}

#endif
