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

#ifndef __PARAS_IMAGE_HPP__
#define __PARAS_IMAGE_HPP__

#include "access.hpp"
#include "data_types.hpp"
#include "exception.hpp"
#include "half.hpp"
#include "property_list.hpp"
#include "range.hpp"
#include "sampler.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

namespace sycl {

class handler;

enum class image_format {
  r8g8b8a8_unorm,
  r16g16b16a16_unorm,
  r8g8b8a8_sint,
  r16g16b16a16_sint,
  r32b32g32a32_sint,
  r8g8b8a8_uint,
  r16g16b16a16_uint,
  r32b32g32a32_uint,
  r16b16g16a16_sfloat,
  r32g32b32a32_sfloat,
  b8g8r8a8_unorm
};

struct image_sampler {
  addressing_mode addressing;
  coordinate_normalization_mode coordinate;
  filtering_mode filtering;
};

enum class image_target { device, host_task };

using image_allocator = std::allocator<unsigned char>;

namespace detail {

inline std::size_t image_channel_bytes(image_format f) {
  switch (f) {
  case image_format::r8g8b8a8_unorm:
  case image_format::r8g8b8a8_sint:
  case image_format::r8g8b8a8_uint:
  case image_format::b8g8r8a8_unorm:
    return 1;
  case image_format::r16g16b16a16_unorm:
  case image_format::r16g16b16a16_sint:
  case image_format::r16g16b16a16_uint:
  case image_format::r16b16g16a16_sfloat:
    return 2;
  default:
    return 4;
  }
}

template <typename T> class kernel_shared_ptr {
  alignas(std::shared_ptr<T>) unsigned char sp_[sizeof(std::shared_ptr<T>)];
  T *raw_ = nullptr;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define PARAS_KSP_DEVICE_PASS 1
#endif
  std::shared_ptr<T> &sp() { return *reinterpret_cast<std::shared_ptr<T> *>(sp_); }
  const std::shared_ptr<T> &sp() const {
    return *reinterpret_cast<const std::shared_ptr<T> *>(sp_);
  }
  void copy_from(const kernel_shared_ptr &o) noexcept {
#ifdef PARAS_KSP_DEVICE_PASS
    for (std::size_t i = 0; i < sizeof(sp_); ++i) sp_[i] = o.sp_[i];
#else
    new (sp_) std::shared_ptr<T>(o.sp());
#endif
    raw_ = o.raw_;
  }
  void move_from(kernel_shared_ptr &o) noexcept {
#ifdef PARAS_KSP_DEVICE_PASS
    copy_from(o);
#else
    new (sp_) std::shared_ptr<T>(std::move(o.sp()));
    raw_ = o.raw_;
    o.raw_ = nullptr;
#endif
  }
  void destroy() {
#ifndef PARAS_KSP_DEVICE_PASS
    sp().~shared_ptr();
#endif
  }

public:
  kernel_shared_ptr() {
#ifndef PARAS_KSP_DEVICE_PASS
    new (sp_) std::shared_ptr<T>();
#endif
  }
  kernel_shared_ptr(std::shared_ptr<T> p) : raw_(p.get()) {
#ifndef PARAS_KSP_DEVICE_PASS
    new (sp_) std::shared_ptr<T>(std::move(p));
#endif
  }

  kernel_shared_ptr(const kernel_shared_ptr &o) noexcept { copy_from(o); }
  kernel_shared_ptr(kernel_shared_ptr &&o) noexcept { move_from(o); }
  kernel_shared_ptr &operator=(const kernel_shared_ptr &o) noexcept {
    if (this != &o) {
      destroy();
      copy_from(o);
    }
    return *this;
  }
  kernel_shared_ptr &operator=(kernel_shared_ptr &&o) noexcept {
    if (this != &o) {
      destroy();
      move_from(o);
    }
    return *this;
  }
  ~kernel_shared_ptr() noexcept { destroy(); }
#undef PARAS_KSP_DEVICE_PASS

  T *get() const noexcept { return raw_; }
  T &operator*() const noexcept { return *raw_; }
  T *operator->() const noexcept { return raw_; }
  friend bool operator==(const kernel_shared_ptr &a, const kernel_shared_ptr &b) {
    return a.raw_ == b.raw_;
  }
};

struct image_storage {
  image_format format;
  std::size_t count;
  std::vector<unsigned char> own;
  unsigned char *data;
  image_storage(image_format f, std::size_t n, void *host)
      : format(f), count(n) {
    if (host) {
      data = static_cast<unsigned char *>(host);
    } else {
      own.assign(n * 4 * image_channel_bytes(f), 0);
      data = own.data();
    }
  }
  std::size_t elem_bytes() const { return 4 * image_channel_bytes(format); }
};

inline float image_half_to_float(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ffu, bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) { man <<= 1; --exp; }
      man &= 0x3ffu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}
inline std::uint16_t image_float_to_half(float f) {
  std::uint32_t x;
  __builtin_memcpy(&x, &f, 4);
  const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t exp = (x >> 23) & 0xffu, man = x & 0x7fffffu;
  if (exp == 0xff)
    return sign | 0x7c00u | (man ? 0x200u : 0u);
  const int e = static_cast<int>(exp) - 127 + 15;
  if (e >= 31)
    return sign | 0x7c00u;
  if (e <= 0) {
    if (e < -10)
      return sign;
    const std::uint32_t m = man | 0x800000u;
    const int shift = 14 - e;
    std::uint32_t hm = m >> shift;
    if ((m >> (shift - 1)) & 1u)
      ++hm;
    return static_cast<std::uint16_t>(sign | hm);
  }
  std::uint32_t hm = man >> 13;
  std::uint32_t he = static_cast<std::uint32_t>(e);
  if (man & 0x1000u) {
    if (++hm == 0x400u) { hm = 0; ++he; }
  }
  if (he >= 31)
    return sign | 0x7c00u;
  return static_cast<std::uint16_t>(sign | (he << 10) | hm);
}

inline double image_load_channel(const image_storage &s, std::size_t idx,
                                 int c) {
  if (s.format == image_format::b8g8r8a8_unorm && c < 3)
    c = 2 - c;
  const unsigned char *p =
      s.data + idx * s.elem_bytes() + c * image_channel_bytes(s.format);
  switch (s.format) {
  case image_format::r8g8b8a8_unorm:
  case image_format::b8g8r8a8_unorm:
    return *p / 255.0;
  case image_format::r16g16b16a16_unorm: {
    std::uint16_t v; __builtin_memcpy(&v, p, 2); return v / 65535.0;
  }
  case image_format::r8g8b8a8_sint:
    return static_cast<std::int8_t>(*p);
  case image_format::r8g8b8a8_uint:
    return *p;
  case image_format::r16g16b16a16_sint: {
    std::int16_t v; __builtin_memcpy(&v, p, 2); return v;
  }
  case image_format::r16g16b16a16_uint: {
    std::uint16_t v; __builtin_memcpy(&v, p, 2); return v;
  }
  case image_format::r32b32g32a32_sint: {
    std::int32_t v; __builtin_memcpy(&v, p, 4); return v;
  }
  case image_format::r32b32g32a32_uint: {
    std::uint32_t v; __builtin_memcpy(&v, p, 4); return v;
  }
  case image_format::r16b16g16a16_sfloat: {
    std::uint16_t v; __builtin_memcpy(&v, p, 2); return image_half_to_float(v);
  }
  case image_format::r32g32b32a32_sfloat: {
    float v; __builtin_memcpy(&v, p, 4); return v;
  }
  }
  return 0;
}

inline void image_store_channel(image_storage &s, std::size_t idx, int c,
                                double v) {
  if (s.format == image_format::b8g8r8a8_unorm && c < 3)
    c = 2 - c;
  unsigned char *p =
      s.data + idx * s.elem_bytes() + c * image_channel_bytes(s.format);
  auto clampd = [](double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
  };
  switch (s.format) {
  case image_format::r8g8b8a8_unorm:
  case image_format::b8g8r8a8_unorm:
    *p = static_cast<unsigned char>(std::lround(clampd(v, 0, 1) * 255.0));
    break;
  case image_format::r16g16b16a16_unorm: {
    std::uint16_t x = static_cast<std::uint16_t>(std::lround(clampd(v, 0, 1) * 65535.0));
    __builtin_memcpy(p, &x, 2);
    break;
  }
  case image_format::r8g8b8a8_sint: { std::int8_t x = static_cast<std::int8_t>(v); __builtin_memcpy(p, &x, 1); break; }
  case image_format::r8g8b8a8_uint: { std::uint8_t x = static_cast<std::uint8_t>(v); __builtin_memcpy(p, &x, 1); break; }
  case image_format::r16g16b16a16_sint: { std::int16_t x = static_cast<std::int16_t>(v); __builtin_memcpy(p, &x, 2); break; }
  case image_format::r16g16b16a16_uint: { std::uint16_t x = static_cast<std::uint16_t>(v); __builtin_memcpy(p, &x, 2); break; }
  case image_format::r32b32g32a32_sint: { std::int32_t x = static_cast<std::int32_t>(v); __builtin_memcpy(p, &x, 4); break; }
  case image_format::r32b32g32a32_uint: { std::uint32_t x = static_cast<std::uint32_t>(v); __builtin_memcpy(p, &x, 4); break; }
  case image_format::r16b16g16a16_sfloat: { std::uint16_t x = image_float_to_half(static_cast<float>(v)); __builtin_memcpy(p, &x, 2); break; }
  case image_format::r32g32b32a32_sfloat: { float x = static_cast<float>(v); __builtin_memcpy(p, &x, 4); break; }
  }
}

template <typename DataT>
DataT image_read_texel(const image_storage &s, std::size_t idx) {
  using E = typename DataT::element_type;
  DataT out;
  for (int c = 0; c < 4; ++c) {
    if constexpr (std::is_same_v<E, half>)
      out[c] = static_cast<E>(static_cast<float>(image_load_channel(s, idx, c)));
    else
      out[c] = static_cast<E>(image_load_channel(s, idx, c));
  }
  return out;
}

template <typename DataT>
void image_write_texel(image_storage &s, std::size_t idx, const DataT &v) {
  using E = typename DataT::element_type;
  for (int c = 0; c < 4; ++c) {
    if constexpr (std::is_same_v<E, half>)
      image_store_channel(s, idx, c, static_cast<double>(static_cast<float>(v[c])));
    else
      image_store_channel(s, idx, c, static_cast<double>(v[c]));
  }
}

template <typename CoordT> auto image_coord(const CoordT &c, int i) {
  if constexpr (std::is_arithmetic_v<CoordT>)
    return c;
  else
    return c[i];
}

template <int D, typename CoordT>
std::size_t image_index(const range<D> &r, const CoordT &c) {
  std::size_t idx = 0;
  for (int d = D - 1; d >= 0; --d) {
    long long v = static_cast<long long>(image_coord(c, d));
    if (v < 0 || v >= static_cast<long long>(r[d]))
      return static_cast<std::size_t>(-1);
    idx = idx * r[d] + static_cast<std::size_t>(v);
  }
  return idx;
}

inline long long image_address(long long i, long long n, addressing_mode m) {
  switch (m) {
  case addressing_mode::repeat:
    i %= n;
    return i < 0 ? i + n : i;
  case addressing_mode::mirrored_repeat: {
    long long p = 2 * n;
    i %= p;
    if (i < 0) i += p;
    return i < n ? i : p - 1 - i;
  }
  case addressing_mode::clamp:
    return (i < 0 || i >= n) ? -1 : i;
  default:
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
  }
}

template <typename DataT, int D, typename CoordT>
DataT image_sample(const image_storage &s, const range<D> &r,
                   const image_sampler &smp, const CoordT &c) {
  std::size_t idx = 0;
  for (int d = D - 1; d >= 0; --d) {
    double x = static_cast<double>(image_coord(c, d));
    if (smp.coordinate == coordinate_normalization_mode::normalized)
      x *= static_cast<double>(r[d]);
    long long i = image_address(static_cast<long long>(std::floor(x)),
                                static_cast<long long>(r[d]), smp.addressing);
    if (i < 0)
      return DataT{};
    idx = idx * r[d] + static_cast<std::size_t>(i);
  }
  return image_read_texel<DataT>(s, idx);
}

inline void image_check_index(std::size_t idx) {
  if (idx == static_cast<std::size_t>(-1))
    throw sycl::exception(make_error_code(errc::invalid),
                          "image coordinate out of range");
}
}

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget = image_target::device>
class unsampled_image_accessor {
  detail::kernel_shared_ptr<detail::image_storage> st_;
  range<Dimensions> r_;

public:
  using value_type = std::conditional_t<AccessMode == access_mode::read,
                                        const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;

  unsampled_image_accessor() = default;
  template <typename ImageT,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<ImageT>, unsampled_image_accessor>>>
  unsampled_image_accessor(ImageT &image, handler &,
                           const property_list & = {})
      : st_(image.paras_storage()), r_(image.get_range()) {}
  template <typename ImageT, typename TagT>
  unsampled_image_accessor(ImageT &image, handler &h, TagT,
                           const property_list &p = {})
      : unsampled_image_accessor(image, h, p) {}

  std::size_t size() const noexcept { return r_.size(); }
  bool empty() const noexcept { return size() == 0; }
  template <typename PropT> bool has_property() const noexcept { return false; }
  template <typename PropT> PropT get_property() const {
    throw sycl::exception(make_error_code(errc::invalid), "property not present");
  }

  template <typename CoordT> DataT read(const CoordT &coords) const {
    std::size_t idx = detail::image_index(r_, coords);
    detail::image_check_index(idx);
    return detail::image_read_texel<DataT>(*st_, idx);
  }
  template <typename CoordT>
  void write(const CoordT &coords, const DataT &color) const {
    std::size_t idx = detail::image_index(r_, coords);
    detail::image_check_index(idx);
    detail::image_write_texel(*st_, idx, color);
  }

  friend bool operator==(const unsampled_image_accessor &a,
                         const unsampled_image_accessor &b) {
    return a.st_ == b.st_;
  }
  friend bool operator!=(const unsampled_image_accessor &a,
                         const unsampled_image_accessor &b) {
    return !(a == b);
  }
  const void *paras_id() const { return st_.get(); }
};

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read
                                      : access_mode::read_write)>
class host_unsampled_image_accessor {
  std::shared_ptr<detail::image_storage> st_;
  range<Dimensions> r_;

public:
  using value_type = std::conditional_t<AccessMode == access_mode::read,
                                        const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;

  host_unsampled_image_accessor() = default;
  template <typename ImageT,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<ImageT>, host_unsampled_image_accessor>>>
  host_unsampled_image_accessor(ImageT &image, const property_list & = {})
      : st_(image.paras_storage()), r_(image.get_range()) {}
  template <typename ImageT, typename TagT>
  host_unsampled_image_accessor(ImageT &image, TagT,
                                const property_list &p = {})
      : host_unsampled_image_accessor(image, p) {}

  std::size_t size() const noexcept { return r_.size(); }
  bool empty() const noexcept { return size() == 0; }
  template <typename PropT> bool has_property() const noexcept { return false; }
  template <typename PropT> PropT get_property() const {
    throw sycl::exception(make_error_code(errc::invalid), "property not present");
  }

  template <typename CoordT> DataT read(const CoordT &coords) const noexcept {
    std::size_t idx = detail::image_index(r_, coords);
    if (idx == static_cast<std::size_t>(-1))
      return DataT{};
    return detail::image_read_texel<DataT>(*st_, idx);
  }
  template <typename CoordT>
  void write(const CoordT &coords, const DataT &color) const noexcept {
    std::size_t idx = detail::image_index(r_, coords);
    if (idx != static_cast<std::size_t>(-1))
      detail::image_write_texel(*st_, idx, color);
  }

  friend bool operator==(const host_unsampled_image_accessor &a,
                         const host_unsampled_image_accessor &b) {
    return a.st_ == b.st_;
  }
  friend bool operator!=(const host_unsampled_image_accessor &a,
                         const host_unsampled_image_accessor &b) {
    return !(a == b);
  }
  const void *paras_id() const { return st_.get(); }
};

template <typename DataT, int Dimensions,
          image_target AccessTarget = image_target::device>
class sampled_image_accessor {
  detail::kernel_shared_ptr<detail::image_storage> st_;
  range<Dimensions> r_;
  image_sampler smp_{};

public:
  using value_type = const DataT;
  using reference = const DataT &;
  using const_reference = const DataT &;

  sampled_image_accessor() = default;
  template <typename ImageT,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<ImageT>, sampled_image_accessor>>>
  sampled_image_accessor(ImageT &image, handler &, const property_list & = {})
      : st_(image.paras_storage()), r_(image.get_range()),
        smp_(image.paras_sampler()) {}
  template <typename ImageT, typename TagT>
  sampled_image_accessor(ImageT &image, handler &h, TagT,
                         const property_list &p = {})
      : sampled_image_accessor(image, h, p) {}

  std::size_t size() const noexcept { return r_.size(); }
  bool empty() const noexcept { return size() == 0; }
  template <typename PropT> bool has_property() const noexcept { return false; }
  template <typename PropT> PropT get_property() const {
    throw sycl::exception(make_error_code(errc::invalid), "property not present");
  }

  template <typename CoordT> DataT read(const CoordT &coords) const noexcept {
    return detail::image_sample<DataT>(*st_, r_, smp_, coords);
  }

  friend bool operator==(const sampled_image_accessor &a,
                         const sampled_image_accessor &b) {
    return a.st_ == b.st_;
  }
  friend bool operator!=(const sampled_image_accessor &a,
                         const sampled_image_accessor &b) {
    return !(a == b);
  }
  const void *paras_id() const { return st_.get(); }
};

template <typename DataT, int Dimensions = 1>
class host_sampled_image_accessor {
  std::shared_ptr<detail::image_storage> st_;
  range<Dimensions> r_;
  image_sampler smp_{};

public:
  using value_type = const DataT;
  using reference = const DataT &;
  using const_reference = const DataT &;

  host_sampled_image_accessor() = default;
  template <typename ImageT,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<ImageT>, host_sampled_image_accessor>>>
  host_sampled_image_accessor(ImageT &image, const property_list & = {})
      : st_(image.paras_storage()), r_(image.get_range()),
        smp_(image.paras_sampler()) {}

  std::size_t size() const noexcept { return r_.size(); }
  bool empty() const noexcept { return size() == 0; }
  template <typename PropT> bool has_property() const noexcept { return false; }
  template <typename PropT> PropT get_property() const {
    throw sycl::exception(make_error_code(errc::invalid), "property not present");
  }

  template <typename CoordT> DataT read(const CoordT &coords) const noexcept {
    return detail::image_sample<DataT>(*st_, r_, smp_, coords);
  }

  friend bool operator==(const host_sampled_image_accessor &a,
                         const host_sampled_image_accessor &b) {
    return a.st_ == b.st_;
  }
  friend bool operator!=(const host_sampled_image_accessor &a,
                         const host_sampled_image_accessor &b) {
    return !(a == b);
  }
  const void *paras_id() const { return st_.get(); }
};

template <int Dimensions = 1, typename AllocatorT = image_allocator>
class unsampled_image {
  std::shared_ptr<detail::image_storage> st_;
  range<Dimensions> r_;
  property_list props_;

public:
  unsampled_image(image_format format, const range<Dimensions> &rangeRef,
                  const property_list &propList = {})
      : st_(std::make_shared<detail::image_storage>(format, rangeRef.size(), nullptr)),
        r_(rangeRef), props_(propList) {}
  unsampled_image(image_format format, const range<Dimensions> &rangeRef,
                  AllocatorT, const property_list &propList = {})
      : unsampled_image(format, rangeRef, propList) {}
  unsampled_image(void *hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const property_list &propList = {})
      : st_(std::make_shared<detail::image_storage>(format, rangeRef.size(), hostPointer)),
        r_(rangeRef), props_(propList) {}
  unsampled_image(void *hostPointer, image_format format,
                  const range<Dimensions> &rangeRef, AllocatorT,
                  const property_list &propList = {})
      : unsampled_image(hostPointer, format, rangeRef, propList) {}

  template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
  unsampled_image(image_format format, const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &, const property_list &propList = {})
      : unsampled_image(format, rangeRef, propList) {}
  template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
  unsampled_image(void *hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &, const property_list &propList = {})
      : unsampled_image(hostPointer, format, rangeRef, propList) {}
  unsampled_image(std::shared_ptr<void> &hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const property_list &propList = {})
      : unsampled_image(hostPointer.get(), format, rangeRef, propList) {}

  range<Dimensions> get_range() const { return r_; }
  template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
  range<Dimensions - 1> get_pitch() const {
    range<Dimensions - 1> p;
    std::size_t row = st_->elem_bytes();
    for (int d = 0; d < Dimensions - 1; ++d) {
      row *= r_[d];
      p[d] = row;
    }
    return p;
  }
  std::size_t size() const noexcept { return r_.size(); }
  std::size_t byte_size() const noexcept { return size() * st_->elem_bytes(); }
  AllocatorT get_allocator() const { return AllocatorT{}; }

  template <typename PropT> bool has_property() const noexcept {
    return props_.template has_property<PropT>();
  }
  template <typename PropT> PropT get_property() const {
    return props_.template get_property<PropT>();
  }
  template <typename DestT> void set_final_data(DestT) {}
  void set_write_back(bool = true) {}

  template <typename DataT,
            access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read
                                                             : access_mode::read_write),
            image_target AccessTarget = image_target::device>
  unsampled_image_accessor<DataT, Dimensions, AccessMode, AccessTarget>
  get_access(handler &cgh, const property_list &p = {}) {
    return {*this, cgh, p};
  }
  template <typename DataT,
            access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read
                                                             : access_mode::read_write)>
  host_unsampled_image_accessor<DataT, Dimensions, AccessMode>
  get_host_access(const property_list &p = {}) {
    return host_unsampled_image_accessor<DataT, Dimensions, AccessMode>(*this, p);
  }

  friend bool operator==(const unsampled_image &a, const unsampled_image &b) {
    return a.st_ == b.st_;
  }
  friend bool operator!=(const unsampled_image &a, const unsampled_image &b) {
    return !(a == b);
  }

  std::shared_ptr<detail::image_storage> paras_storage() const { return st_; }
};

template <int Dimensions = 1, typename AllocatorT = image_allocator>
class sampled_image {
  std::shared_ptr<detail::image_storage> st_;
  range<Dimensions> r_;
  image_sampler smp_;
  property_list props_;

public:
  sampled_image(const void *hostPointer, image_format format,
                image_sampler sampler, const range<Dimensions> &rangeRef,
                const property_list &propList = {})
      : st_(std::make_shared<detail::image_storage>(
            format, rangeRef.size(), const_cast<void *>(hostPointer))),
        r_(rangeRef), smp_(sampler), props_(propList) {}
  template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
  sampled_image(const void *hostPointer, image_format format,
                image_sampler sampler, const range<Dimensions> &rangeRef,
                const range<Dimensions - 1> &, const property_list &propList = {})
      : sampled_image(hostPointer, format, sampler, rangeRef, propList) {}
  sampled_image(std::shared_ptr<const void> &hostPointer, image_format format,
                image_sampler sampler, const range<Dimensions> &rangeRef,
                const property_list &propList = {})
      : sampled_image(hostPointer.get(), format, sampler, rangeRef, propList) {}

  range<Dimensions> get_range() const { return r_; }
  std::size_t size() const noexcept { return r_.size(); }
  std::size_t byte_size() const noexcept { return size() * st_->elem_bytes(); }

  template <typename PropT> bool has_property() const noexcept {
    return props_.template has_property<PropT>();
  }
  template <typename PropT> PropT get_property() const {
    return props_.template get_property<PropT>();
  }

  template <typename DataT, image_target AccessTarget = image_target::device>
  sampled_image_accessor<DataT, Dimensions, AccessTarget>
  get_access(handler &cgh, const property_list &p = {}) {
    return {*this, cgh, p};
  }
  template <typename DataT>
  host_sampled_image_accessor<DataT, Dimensions>
  get_host_access(const property_list &p = {}) {
    return host_sampled_image_accessor<DataT, Dimensions>(*this, p);
  }

  friend bool operator==(const sampled_image &a, const sampled_image &b) {
    return a.st_ == b.st_;
  }
  friend bool operator!=(const sampled_image &a, const sampled_image &b) {
    return !(a == b);
  }

  std::shared_ptr<detail::image_storage> paras_storage() const { return st_; }
  image_sampler paras_sampler() const { return smp_; }
};

}

namespace std {
template <int D, typename A> struct hash<sycl::unsampled_image<D, A>> {
  size_t operator()(const sycl::unsampled_image<D, A> &i) const noexcept {
    return std::hash<const void *>()(i.paras_storage().get());
  }
};
template <int D, typename A> struct hash<sycl::sampled_image<D, A>> {
  size_t operator()(const sycl::sampled_image<D, A> &i) const noexcept {
    return std::hash<const void *>()(i.paras_storage().get());
  }
};
template <typename T, int D, sycl::access_mode M, sycl::image_target Tg>
struct hash<sycl::unsampled_image_accessor<T, D, M, Tg>> {
  size_t operator()(const sycl::unsampled_image_accessor<T, D, M, Tg> &a) const noexcept {
    return std::hash<const void *>()(a.paras_id());
  }
};
template <typename T, int D, sycl::image_target Tg>
struct hash<sycl::sampled_image_accessor<T, D, Tg>> {
  size_t operator()(const sycl::sampled_image_accessor<T, D, Tg> &a) const noexcept {
    return std::hash<const void *>()(a.paras_id());
  }
};
template <typename T, int D, sycl::access_mode M>
struct hash<sycl::host_unsampled_image_accessor<T, D, M>> {
  size_t operator()(const sycl::host_unsampled_image_accessor<T, D, M> &a) const noexcept {
    return std::hash<const void *>()(a.paras_id());
  }
};
template <typename T, int D>
struct hash<sycl::host_sampled_image_accessor<T, D>> {
  size_t operator()(const sycl::host_sampled_image_accessor<T, D> &a) const noexcept {
    return std::hash<const void *>()(a.paras_id());
  }
};
}

#endif
