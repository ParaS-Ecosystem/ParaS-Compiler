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

#include "sycl/device.hpp"
#include "sycl/aspect.hpp"
#include "utilities/gpu_detection.hpp"
#include "utilities/internal_utils.hpp"

#include <thread>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace sycl {

device::device()
    : name_("Unknown CPU"), vendor_("Unknown"), driver_version_(""),
      version_(""), max_compute_units_(0), max_work_group_size_(1024),
      global_mem_size_bytes_(0), local_mem_type_(info::local_mem_type::none),
      is_cpu_(true), is_gpu_(false), is_accelerator_(false),
      have_queue_profiling_(false) {
  unsigned int hw = std::thread::hardware_concurrency();
  max_compute_units_ = hw ? hw : 1;

#ifdef __linux__
  {
    std::vector<std::string> lscpu;
    if (paras_extension::run_cmd_lines_cached("lscpu 2>/dev/null", lscpu)) {
      unsigned long sockets = 0, cores_per_socket = 0;
      for (auto ln : lscpu) {
        paras_extension::trim_inplace(ln);
        if (ln.rfind("Model name:", 0) == 0) {
          auto v = ln.substr(std::string("Model name:").size());
          paras_extension::trim_inplace(v);
          if (!v.empty())
            name_ = v;
        } else if (ln.rfind("Vendor ID:", 0) == 0) {
          auto v = ln.substr(std::string("Vendor ID:").size());
          paras_extension::trim_inplace(v);
          if (!v.empty())
            vendor_ = v;
        } else if (ln.rfind("Socket(s):", 0) == 0) {
          auto v = ln.substr(std::string("Socket(s):").size());
          paras_extension::trim_inplace(v);
          sockets = paras_extension::parse_ul_or0(v);
        } else if (ln.rfind("Core(s) per socket:", 0) == 0) {
          auto v = ln.substr(std::string("Core(s) per socket:").size());
          paras_extension::trim_inplace(v);
          cores_per_socket = paras_extension::parse_ul_or0(v);
        }
      }
      if (sockets > 0 && cores_per_socket > 0) {
        const unsigned long cores = sockets * cores_per_socket;
        if (cores > 0)
          max_compute_units_ = static_cast<std::uint32_t>(cores);
      }
    }
  }

  {
    auto kern =
        paras_extension::run_cmd_first_line_cached("uname -r 2>/dev/null");
    paras_extension::trim_inplace(kern);
    if (!kern.empty())
      version_ = "Linux " + kern;
  }

  {
    auto gcc = paras_extension::run_cmd_first_line_cached(
        "gcc --version 2>/dev/null | head -n1");
    paras_extension::trim_inplace(gcc);
    if (!gcc.empty())
      driver_version_ = gcc;
  }

  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    unsigned long long bytes =
        static_cast<unsigned long long>(info.totalram) * info.mem_unit;
    global_mem_size_bytes_ = static_cast<std::uint64_t>(bytes);
  }
#endif

  paras_extension::trim_inplace(name_);
  paras_extension::trim_inplace(vendor_);
  paras_extension::trim_inplace(driver_version_);
  paras_extension::trim_inplace(version_);
}

device::device(::paras_extension::device_ctor_tag, std::string name,
               std::string vendor, std::string driver_version,
               std::string version, std::uint32_t max_compute_units,
               std::size_t max_work_group_size,
               std::uint64_t global_mem_size_bytes,
               info::local_mem_type local_mem_type, bool is_cpu, bool is_gpu,
               bool is_accelerator, int native_id, bool have_queue_profiling)
    : name_(std::move(name)), vendor_(std::move(vendor)),
      driver_version_(std::move(driver_version)), version_(std::move(version)),
      max_compute_units_(max_compute_units),
      max_work_group_size_(max_work_group_size),
      global_mem_size_bytes_(global_mem_size_bytes),
      local_mem_type_(local_mem_type), is_cpu_(is_cpu), is_gpu_(is_gpu),
      is_accelerator_(is_accelerator), native_id_(native_id),
      have_queue_profiling_(have_queue_profiling) {
  paras_extension::trim_inplace(name_);
  paras_extension::trim_inplace(vendor_);
  paras_extension::trim_inplace(driver_version_);
  paras_extension::trim_inplace(version_);
}

backend device::get_backend() const noexcept { return backend::host; }
bool device::is_cpu() const noexcept { return is_cpu_; }
bool device::is_gpu() const noexcept { return is_gpu_; }
bool device::is_accelerator() const noexcept { return is_accelerator_; }
platform device::get_platform() const { return platform("host"); }

std::vector<device> device::get_devices(info::device_type type) {
  std::vector<device> out;

  if (type == info::device_type::cpu || type == info::device_type::all) {
    out.emplace_back(device{});
  }

  if (type == info::device_type::gpu || type == info::device_type::all) {
    auto gpus = ::paras_extension::detect_all_gpus();
    out.insert(out.end(), gpus.begin(), gpus.end());
  }
  
  return out;
}

} // namespace sycl
