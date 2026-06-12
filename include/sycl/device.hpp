/**<
 * Copyright 2025 The ParaS-Compiler Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_DEVICE_HPP__
#define __PARAS_DEVICE_HPP__

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "aspect.hpp"

namespace paras_extension {
struct device_ctor_tag {};
} // namespace paras_extension

namespace sycl {

namespace info {

enum class device_type { cpu, gpu, accelerator, all };

enum class local_mem_type { none, local, global };

enum class partition_property {
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};
enum class partition_affinity_domain { next_partition };

namespace device {

struct name {};
struct vendor {};
struct version {};
struct driver_version {};
struct max_compute_units {};
struct max_work_group_size {};
struct local_mem_type {};
struct global_mem_size {};
struct queue_profiling {};
} // namespace device

} // namespace info

enum class backend { cuda, host, hip };

template <backend Backend, class T> struct backend_return;

template <backend Backend, class T>
using backend_return_t = typename backend_return<Backend, T>::type;

template <backend Backend, class T>
backend_return_t<Backend, T> get_native(const T &n_obj);

class platform {
public:
  platform() = default;
  explicit platform(std::string name) : name_(std::move(name)) {}
  const std::string &get_name() const noexcept { return name_; }

private:
  std::string name_{"host"};
};

class device {
public:
  device();

  device(::paras_extension::device_ctor_tag, std::string name,
         std::string vendor, std::string driver_version, std::string version,
         std::uint32_t max_compute_units, std::size_t max_work_group_size,
         std::uint64_t global_mem_size_bytes,
         info::local_mem_type local_mem_type, bool is_cpu, bool is_gpu,
         bool is_accelerator, int native_id, bool queue_profiling);

  template <typename DeviceSelector>
  explicit device(const DeviceSelector &) : device() {}

  backend get_backend() const noexcept;

  bool is_cpu() const noexcept;
  bool is_gpu() const noexcept;
  bool is_accelerator() const noexcept;
  platform get_platform() const;

  bool has(const aspect &aspect_name) const {
    switch (aspect_name) {
    case aspect::cpu:
      return is_cpu_;
    case aspect::queue_profiling:
      return have_queue_profiling_;
    default:
      return false;
    }
  }

  int get_native_id() const noexcept { return native_id_; }

  template <typename Param> auto get_info() const;

  template <typename Param> auto get_backend_info() const {
    return get_info<Param>();
  }

  template <info::partition_property Prop>
  std::vector<device> create_sub_devices(std::size_t) const {
    return {};
  }
  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(const std::vector<std::size_t> &) const {
    return {};
  }
  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain) const {
    return {};
  }

  static std::vector<device>
  get_devices(info::device_type type = info::device_type::all);

private:
  std::string name_;
  std::string vendor_;
  std::string driver_version_;
  std::string version_;
  std::uint32_t max_compute_units_{0};
  std::size_t max_work_group_size_{0};
  std::uint64_t global_mem_size_bytes_{0};
  info::local_mem_type local_mem_type_{info::local_mem_type::none};

  bool is_cpu_{true};
  bool is_gpu_{false};
  bool is_accelerator_{false};

  int native_id_{0};
  bool have_queue_profiling_;
};

template <> inline auto device::get_info<info::device::name>() const {
  return name_;
}

template <> inline auto device::get_info<info::device::vendor>() const {
  return vendor_;
}

template <> inline auto device::get_info<info::device::version>() const {
  return version_;
}

template <> inline auto device::get_info<info::device::driver_version>() const {
  return driver_version_;
}

template <>
inline auto device::get_info<info::device::max_compute_units>() const {
  return max_compute_units_;
}

template <>
inline auto device::get_info<info::device::max_work_group_size>() const {
  return max_work_group_size_;
}

template <> inline auto device::get_info<info::device::local_mem_type>() const {
  return local_mem_type_;
}

template <>
inline auto device::get_info<info::device::global_mem_size>() const {
  return global_mem_size_bytes_;
}

template <>
inline auto device::get_info<info::device::queue_profiling>() const {
  return have_queue_profiling_;
}

template <> struct backend_return<backend::cuda, device> {
  using type = int;
};

template <> struct backend_return<backend::hip, device> {
  using type = int;
};

template <> struct backend_return<backend::host, device> {
  using type = int;
};

template <> inline int get_native<backend::cuda, device>(const device &dev) {
  return dev.get_native_id();
}

template <> inline int get_native<backend::hip, device>(const device &dev) {
  return dev.get_native_id();
}

template <> inline int get_native<backend::host, device>(const device &dev) {
  return dev.get_native_id();
}

} // namespace sycl

#endif
