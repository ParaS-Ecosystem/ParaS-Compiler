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

#ifndef __PARAS_DEVICE_HPP__
#define __PARAS_DEVICE_HPP__

#include "kem/host_group_runtime.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include "id.hpp"

#include "atomic_ref.hpp"

#include "aspect.hpp"
#include "kem_gpu/gpu_utilities.hpp"
#include "exception.hpp"

namespace paras_extension {
struct device_ctor_tag {};
} // namespace paras_extension

namespace paras_extension {
struct gpu_limits {
    std::uint64_t local_mem = 0;
    std::size_t block_dim[3] = {0, 0, 0}; // x, y, z
};
inline bool query_gpu_limits(int native_id, gpu_limits& out) {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && PARAS_HIP_BACKEND
    int v[4] = {0, 0, 0, 0};
    if (hipDeviceGetAttribute(&v[0], hipDeviceAttributeMaxSharedMemoryPerBlock, native_id) !=
            hipSuccess ||
        hipDeviceGetAttribute(&v[1], hipDeviceAttributeMaxBlockDimX, native_id) != hipSuccess ||
        hipDeviceGetAttribute(&v[2], hipDeviceAttributeMaxBlockDimY, native_id) != hipSuccess ||
        hipDeviceGetAttribute(&v[3], hipDeviceAttributeMaxBlockDimZ, native_id) != hipSuccess)
        return false;
#elif !defined(__CUDA_ARCH__) && PARAS_CUDA_BACKEND
    int v[4] = {0, 0, 0, 0};
    if (cudaDeviceGetAttribute(&v[0], cudaDevAttrMaxSharedMemoryPerBlock, native_id) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&v[1], cudaDevAttrMaxBlockDimX, native_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&v[2], cudaDevAttrMaxBlockDimY, native_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&v[3], cudaDevAttrMaxBlockDimZ, native_id) != cudaSuccess)
        return false;
#else
    (void)native_id;
    (void)out;
    return false;
#endif
#if (!defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && PARAS_HIP_BACKEND) ||          \
    (!defined(__CUDA_ARCH__) && PARAS_CUDA_BACKEND)
    out.local_mem = static_cast<std::uint64_t>(v[0]);
    for (int i = 0; i < 3; ++i)
        out.block_dim[i] = static_cast<std::size_t>(v[i + 1]);
    return true;
#endif
}
} // namespace paras_extension

struct CUstream_st;

namespace sycl {

class device;

class kernel_id;

class platform;

namespace info {

enum class device_type { cpu, gpu, accelerator, custom, automatic, host, all };

enum class local_mem_type { none, local, global };

enum class partition_property {
    no_partition,
    partition_equally,
    partition_by_counts,
    partition_by_affinity_domain
};
enum class partition_affinity_domain {
    not_applicable,
    numa,
    L4_cache,
    L3_cache,
    L2_cache,
    L1_cache,
    next_partitionable
};

enum class fp_config {
    denorm,
    inf_nan,
    round_to_nearest,
    round_to_zero,
    round_to_inf,
    fma,
    correctly_rounded_divide_sqrt,
    soft_float
};
enum class global_mem_cache_type { none, read_only, read_write };

enum class execution_capability { exec_kernel, exec_native_kernel };

namespace device {
struct name {
    using return_type = std::string;
};
struct vendor {
    using return_type = std::string;
};
struct version {
    using return_type = std::string;
};
struct driver_version {
    using return_type = std::string;
};
struct max_compute_units {
    using return_type = std::uint32_t;
};
struct max_work_group_size {
    using return_type = std::size_t;
};
struct vendor_id {
    using return_type = std::uint32_t;
};
struct max_work_item_dimensions {
    using return_type = std::uint32_t;
};
template <int Dimensions>
struct max_work_item_sizes {
    using return_type = sycl::range<Dimensions>;
};
struct max_num_sub_groups {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_char {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_short {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_int {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_long {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_long_long {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_float {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_double {
    using return_type = std::uint32_t;
};
struct preferred_vector_width_half {
    using return_type = std::uint32_t;
};
struct address_bits {
    using return_type = std::uint32_t;
};
struct max_clock_frequency {
    using return_type = std::uint32_t;
};
struct max_mem_alloc_size {
    using return_type = std::uint64_t;
};
struct image_support {
    using return_type = bool;
};
struct max_read_image_args {
    using return_type = std::uint32_t;
};
struct max_write_image_args {
    using return_type = std::uint32_t;
};
struct image2d_max_width {
    using return_type = std::size_t;
};
struct image2d_max_height {
    using return_type = std::size_t;
};
struct image3d_max_width {
    using return_type = std::size_t;
};
struct image3d_max_height {
    using return_type = std::size_t;
};
struct image3d_max_depth {
    using return_type = std::size_t;
};
struct native_vector_width_char {
    using return_type = std::uint32_t;
};
struct native_vector_width_short {
    using return_type = std::uint32_t;
};
struct native_vector_width_int {
    using return_type = std::uint32_t;
};
struct native_vector_width_long {
    using return_type = std::uint32_t;
};
struct native_vector_width_long_long {
    using return_type = std::uint32_t;
};
struct native_vector_width_float {
    using return_type = std::uint32_t;
};
struct native_vector_width_double {
    using return_type = std::uint32_t;
};
struct native_vector_width_half {
    using return_type = std::uint32_t;
};
struct image_max_buffer_size {
    using return_type = std::size_t;
};
struct max_samplers {
    using return_type = std::uint32_t;
};
struct max_parameter_size {
    using return_type = std::size_t;
};
struct mem_base_addr_align {
    using return_type = std::uint32_t;
};
struct half_fp_config {
    using return_type = std::vector<sycl::info::fp_config>;
};
struct single_fp_config {
    using return_type = std::vector<sycl::info::fp_config>;
};
struct double_fp_config {
    using return_type = std::vector<sycl::info::fp_config>;
};
struct global_mem_cache_type {
    using return_type = sycl::info::global_mem_cache_type;
};
struct global_mem_cache_line_size {
    using return_type = std::uint32_t;
};
struct global_mem_cache_size {
    using return_type = std::uint64_t;
};
struct max_constant_buffer_size {
    using return_type = std::uint64_t;
};
struct max_constant_args {
    using return_type = std::uint32_t;
};
struct local_mem_size {
    using return_type = std::uint64_t;
};
struct error_correction_support {
    using return_type = bool;
};
struct host_unified_memory {
    using return_type = bool;
};
struct profiling_timer_resolution {
    using return_type = std::size_t;
};
struct is_endian_little {
    using return_type = bool;
};
struct is_available {
    using return_type = bool;
};
struct built_in_kernel_ids {
    using return_type = std::vector<sycl::kernel_id>;
};
struct built_in_kernels {
    using return_type = std::vector<std::string>;
};
struct platform {
    using return_type = sycl::platform;
};
struct backend_version {
    using return_type = std::string;
};
struct aspects {
    using return_type = std::vector<sycl::aspect>;
};
struct extensions {
    using return_type = std::vector<std::string>;
};
struct printf_buffer_size {
    using return_type = std::size_t;
};
struct parent_device {
    using return_type = sycl::device;
};
struct partition_max_sub_devices {
    using return_type = std::uint32_t;
};
struct partition_type_property {
    using return_type = sycl::info::partition_property;
};
struct partition_type_affinity_domain {
    using return_type = sycl::info::partition_affinity_domain;
};
struct sub_group_sizes {
    using return_type = std::vector<std::size_t>;
};
struct local_mem_type {
    using return_type = sycl::info::local_mem_type;
};
struct global_mem_size {
    using return_type = std::uint64_t;
};
struct queue_profiling {
    using return_type = bool;
};
struct device_type {
    using return_type = sycl::info::device_type;
};
struct is_compiler_available {
    using return_type = bool;
};
struct is_linker_available {
    using return_type = bool;
};
struct atomic_fence_order_capabilities {
    using return_type = std::vector<memory_order>;
};
struct atomic_fence_scope_capabilities {
    using return_type = std::vector<memory_scope>;
};

struct atomic_memory_order_capabilities {
    using return_type = std::vector<memory_order>;
};

struct atomic_memory_scope_capabilities {
    using return_type = std::vector<memory_scope>;
};

struct partition_properties {
    using return_type = std::vector<sycl::info::partition_property>;
};
struct partition_affinity_domains {
    using return_type = std::vector<sycl::info::partition_affinity_domain>;
};

} // namespace device

namespace platform {
struct name {
    using return_type = std::string;
};
struct vendor {
    using return_type = std::string;
};
struct version {
    using return_type = std::string;
};
struct extensions {
    using return_type = std::vector<std::string>;
};

} // namespace platform

} // namespace info

enum class backend { cuda, host, hip };

class queue;

template <backend Backend, class T>
struct backend_return;

template <backend Backend, class T>
using backend_return_t = typename backend_return<Backend, T>::type;

template <backend Backend, class T>
backend_return_t<Backend, T> get_native(const T& n_obj);

class device;

class platform {

public:
    platform() = default;

    explicit platform(std::string name) : name_(std::move(name)) {}

    template <typename DeviceSelector,
              typename = std::enable_if_t<!std::is_convertible_v<DeviceSelector, std::string>>>
    explicit platform(const DeviceSelector& selector);

    bool has_extension(const std::string&) const noexcept { return false; }

    bool operator==(const platform& rhs) const noexcept { return name_ == rhs.name_; }
    bool operator!=(const platform& rhs) const noexcept { return !(*this == rhs); }
    const std::string& get_name() const noexcept { return name_; }

    backend get_backend() const noexcept {
        if (name_ == "cuda")
            return backend::cuda;
        if (name_ == "hip")
            return backend::hip;
        return backend::host;
    }

    std::vector<sycl::device>
        get_devices(sycl::info::device_type = sycl::info::device_type::all) const;

    bool has(aspect aspect_name) const;

    template <typename Param>
    auto get_info() const;

    static std::vector<platform> get_platforms();

private:
    std::string name_{"host"};
};

template <>
inline auto platform::get_info<info::platform::name>() const {
    return name_;
}

template <>
inline auto platform::get_info<info::platform::vendor>() const {
    return std::string("ParaS-Compiler");
}

template <>
inline auto platform::get_info<info::platform::version>() const {
    return std::string("SYCL 2020");
}

template <>
inline auto platform::get_info<info::platform::extensions>() const {
    return std::vector<std::string>{};
}

class device {
public:
    device();

    device(::paras_extension::device_ctor_tag, std::string name, std::string vendor,
           std::string driver_version, std::string version, std::uint32_t max_compute_units,
           std::size_t max_work_group_size, std::uint64_t global_mem_size_bytes,
           info::local_mem_type local_mem_type, bool is_cpu, bool is_gpu, bool is_accelerator,
           int native_id, bool queue_profiling);

    template <typename DeviceSelector>
    explicit device(const DeviceSelector& selector) : device() {
        int best_score = -1;
        for (const device& d : get_devices(info::device_type::all)) {
            const int score = selector(d);
            if (score > best_score) {
                best_score = score;
                *this = d;
            }
        }
        if (best_score < 0) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                                  "no device satisfies the device selector");
        }
    }

    backend get_backend() const noexcept;

    bool is_cpu() const noexcept;
    bool is_gpu() const noexcept;
    bool is_accelerator() const noexcept;
    platform get_platform() const;

    bool has(const aspect& aspect_name) const {
        switch (aspect_name) {
        case aspect::cpu:
            return is_cpu_;
        case aspect::gpu:
            return is_gpu_;
        case aspect::accelerator:
            return is_accelerator_;
        case aspect::queue_profiling:
            return have_queue_profiling_;
        case aspect::fp64:
        case aspect::atomic64:
            return false;
        case aspect::usm_device_allocations:
        case aspect::usm_host_allocations:
        case aspect::usm_shared_allocations:
            return is_cpu_ || is_gpu_;
        default:
            return false;
        }
    }

    bool operator==(const device& rhs) const noexcept {
        return native_id_ == rhs.native_id_ && is_cpu_ == rhs.is_cpu_ && is_gpu_ == rhs.is_gpu_ &&
               is_accelerator_ == rhs.is_accelerator_;
    }

    bool operator!=(const device& rhs) const noexcept { return !(*this == rhs); }

    int get_native_id() const noexcept { return native_id_; }

    template <typename Param>
    auto get_info() const;

    template <typename Param>
    auto get_backend_info() const {
        return get_info<Param>();
    }
    template <info::partition_property Prop>
    std::vector<device> create_sub_devices(std::size_t) const {
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported));
    }
    template <info::partition_property Prop>
    std::vector<device> create_sub_devices(const std::vector<std::size_t>&) const {
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported));
    }
    template <info::partition_property Prop>
    std::vector<device> create_sub_devices(info::partition_affinity_domain) const {
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported));
    }

    static std::vector<device> get_devices(info::device_type type = info::device_type::all);

private:
    std::string name_;
    std::string vendor_;
    std::string driver_version_;
    std::string version_;
    std::uint32_t max_compute_units_{0};
    std::size_t max_work_group_size_{0};
    std::uint64_t global_mem_size_bytes_{0};
    info::local_mem_type local_mem_type_{info::local_mem_type::none};
    std::uint32_t vendor_id_{0};
    std::uint32_t max_work_item_dimensions_{3};
    std::uint32_t max_num_sub_groups_{1};
    std::uint32_t preferred_vector_width_char_{1};
    std::uint32_t preferred_vector_width_short_{1};
    std::uint32_t preferred_vector_width_int_{1};
    std::uint32_t preferred_vector_width_long_{1};
    std::uint32_t preferred_vector_width_long_long_{1};
    std::uint32_t preferred_vector_width_float_{1};
    std::uint32_t preferred_vector_width_double_{1};
    std::uint32_t preferred_vector_width_half_{1};

    std::uint32_t address_bits_{64};
    std::uint32_t max_clock_frequency_{0};
    std::uint64_t max_mem_alloc_size_{0};
    bool image_support_{false};
    std::uint32_t max_read_image_args_{0};
    std::uint32_t max_write_image_args_{0};
    std::size_t image2d_max_width_{0};
    std::size_t image2d_max_height_{0};
    std::size_t image3d_max_width_{0};
    std::size_t image3d_max_height_{0};
    std::size_t image3d_max_depth_{0};
    std::uint32_t native_vector_width_char_{1};
    std::uint32_t native_vector_width_short_{1};
    std::uint32_t native_vector_width_int_{1};
    std::uint32_t native_vector_width_long_{1};
    std::uint32_t native_vector_width_long_long_{1};
    std::uint32_t native_vector_width_float_{1};
    std::uint32_t native_vector_width_double_{1};
    std::uint32_t native_vector_width_half_{1};

    std::size_t image_max_buffer_size_{0};
    std::uint32_t max_samplers_{0};
    std::size_t max_parameter_size_{0};
    std::uint32_t mem_base_addr_align_{0};
    std::uint32_t global_mem_cache_line_size_{0};
    std::uint64_t global_mem_cache_size_{0};
    std::uint64_t max_constant_buffer_size_{0};
    std::uint32_t max_constant_args_{0};
    std::uint64_t local_mem_size_{0};
    bool error_correction_support_{false};
    bool host_unified_memory_{true};
    std::size_t profiling_timer_resolution_{0};
    bool is_endian_little_{true};
    bool is_available_{true};

    std::size_t printf_buffer_size_{0};
    std::uint32_t partition_max_sub_devices_{0};

    bool is_cpu_{true};
    bool is_gpu_{false};
    bool is_accelerator_{false};

    int native_id_{0};
    bool have_queue_profiling_;
};

inline std::vector<platform> platform::get_platforms() {
    std::vector<platform> platforms;
    const auto gpus = device::get_devices(info::device_type::gpu);
    bool has_cuda = false;
    bool has_hip = false;
    for (const auto& gpu : gpus) {
        if (gpu.get_backend() == backend::cuda) {
            has_cuda = true;
        } else if (gpu.get_backend() == backend::hip) {
            has_hip = true;
        }
    }
    if (has_cuda) {
        platforms.emplace_back("cuda");
    }
    if (has_hip) {
        platforms.emplace_back("hip");
    }
    platforms.emplace_back("host");
    return platforms;
}

inline std::vector<sycl::device> platform::get_devices(sycl::info::device_type type) const {
    if (name_ == "cuda" || name_ == "hip") {
        if (type == sycl::info::device_type::gpu || type == sycl::info::device_type::all) {
            const auto all_gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
            std::vector<sycl::device> platform_devices;
            const auto requested_backend =
                name_ == "cuda" ? sycl::backend::cuda : sycl::backend::hip;
            for (const auto& gpu : all_gpus) {
                if (gpu.get_backend() == requested_backend) {
                    platform_devices.push_back(gpu);
                }
            }
            return platform_devices;
        }
        return {};
    }

    switch (type) {
    case sycl::info::device_type::cpu:
    case sycl::info::device_type::host:
    case sycl::info::device_type::automatic:
    case sycl::info::device_type::all:
        return {sycl::device{}};
    default:
        return {};
    }
}

template <typename DeviceSelector, typename>
inline platform::platform(const DeviceSelector& selector) : platform() {
    if (selector(sycl::device{}) < 0) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime));
    }
}

inline bool platform::has(aspect aspect_name) const {
    const auto devices = get_devices();
    return !devices.empty() && devices.front().has(aspect_name);
}

template <typename KernelName>
inline bool is_compatible(const device&) {
    return true;
}

template <>
inline auto device::get_info<info::device::name>() const {
    return name_;
}

template <>
inline auto device::get_info<info::device::vendor>() const {
    return vendor_;
}

template <>
inline auto device::get_info<info::device::version>() const {
    return version_;
}

template <>
inline auto device::get_info<info::device::driver_version>() const {
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

template <>
inline auto device::get_info<info::device::vendor_id>() const {
    return vendor_id_;
}

template <>
inline auto device::get_info<info::device::max_work_item_dimensions>() const {
    return max_work_item_dimensions_;
}

template <>
inline auto device::get_info<info::device::max_work_item_sizes<1>>() const {
    paras_extension::gpu_limits g;
    if (is_gpu_ && paras_extension::query_gpu_limits(native_id_, g))
        return sycl::range<1>(g.block_dim[0]);
    return sycl::range<1>(max_work_group_size_);
}

template <>
inline auto device::get_info<info::device::max_work_item_sizes<2>>() const {
    paras_extension::gpu_limits g; // SYCL dimension N-1 is GPU x
    if (is_gpu_ && paras_extension::query_gpu_limits(native_id_, g))
        return sycl::range<2>(g.block_dim[1], g.block_dim[0]);
    return sycl::range<2>(max_work_group_size_, max_work_group_size_);
}

template <>
inline auto device::get_info<info::device::max_work_item_sizes<3>>() const {
    paras_extension::gpu_limits g; // SYCL dimension N-1 is GPU x
    if (is_gpu_ && paras_extension::query_gpu_limits(native_id_, g))
        return sycl::range<3>(g.block_dim[2], g.block_dim[1], g.block_dim[0]);
    return sycl::range<3>(max_work_group_size_, max_work_group_size_, max_work_group_size_);
}

template <>
inline auto device::get_info<info::device::max_num_sub_groups>() const {
    return max_num_sub_groups_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_char>() const {
    return preferred_vector_width_char_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_short>() const {
    return preferred_vector_width_short_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_int>() const {
    return preferred_vector_width_int_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_long>() const {
    return preferred_vector_width_long_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_long_long>() const {
    return preferred_vector_width_long_long_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_float>() const {
    return preferred_vector_width_float_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_double>() const {
    return preferred_vector_width_double_;
}

template <>
inline auto device::get_info<info::device::preferred_vector_width_half>() const {
    return preferred_vector_width_half_;
}

template <>
inline auto device::get_info<info::device::address_bits>() const {
    return address_bits_;
}

template <>
inline auto device::get_info<info::device::max_clock_frequency>() const {
    return max_clock_frequency_;
}

template <>
inline auto device::get_info<info::device::max_mem_alloc_size>() const {
    return max_mem_alloc_size_;
}

template <>
inline auto device::get_info<info::device::image_support>() const {
    return image_support_;
}

template <>
inline auto device::get_info<info::device::max_read_image_args>() const {
    return max_read_image_args_;
}

template <>
inline auto device::get_info<info::device::max_write_image_args>() const {
    return max_write_image_args_;
}

template <>
inline auto device::get_info<info::device::image2d_max_width>() const {
    return image2d_max_width_;
}

template <>
inline auto device::get_info<info::device::image2d_max_height>() const {
    return image2d_max_height_;
}

template <>
inline auto device::get_info<info::device::image3d_max_width>() const {
    return image3d_max_width_;
}

template <>
inline auto device::get_info<info::device::image3d_max_height>() const {
    return image3d_max_height_;
}

template <>
inline auto device::get_info<info::device::image3d_max_depth>() const {
    return image3d_max_depth_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_char>() const {
    return native_vector_width_char_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_short>() const {
    return native_vector_width_short_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_int>() const {
    return native_vector_width_int_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_long>() const {
    return native_vector_width_long_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_long_long>() const {
    return native_vector_width_long_long_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_float>() const {
    return native_vector_width_float_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_double>() const {
    return native_vector_width_double_;
}

template <>
inline auto device::get_info<info::device::native_vector_width_half>() const {
    return native_vector_width_half_;
}

template <>
inline auto device::get_info<info::device::image_max_buffer_size>() const {
    return image_max_buffer_size_;
}

template <>
inline auto device::get_info<info::device::max_samplers>() const {
    return max_samplers_;
}

template <>
inline auto device::get_info<info::device::max_parameter_size>() const {
    return max_parameter_size_;
}

template <>
inline auto device::get_info<info::device::mem_base_addr_align>() const {
    return mem_base_addr_align_;
}

template <>
inline auto device::get_info<info::device::half_fp_config>() const {
    return std::vector<sycl::info::fp_config>{};
}

template <>
inline auto device::get_info<info::device::single_fp_config>() const {
    return std::vector<sycl::info::fp_config>{sycl::info::fp_config::round_to_nearest,
                                              sycl::info::fp_config::inf_nan};
}

template <>
inline auto device::get_info<info::device::double_fp_config>() const {
    return std::vector<sycl::info::fp_config>{sycl::info::fp_config::round_to_nearest,
                                              sycl::info::fp_config::inf_nan};
}

template <>
inline auto device::get_info<info::device::global_mem_cache_type>() const {
    return sycl::info::global_mem_cache_type::read_write;
}

template <>
inline auto device::get_info<info::device::global_mem_cache_line_size>() const {
    return global_mem_cache_line_size_;
}

template <>
inline auto device::get_info<info::device::global_mem_cache_size>() const {
    return global_mem_cache_size_;
}

template <>
inline auto device::get_info<info::device::max_constant_buffer_size>() const {
    return max_constant_buffer_size_;
}

template <>
inline auto device::get_info<info::device::max_constant_args>() const {
    return max_constant_args_;
}

template <>
inline auto device::get_info<info::device::local_mem_size>() const {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    if (is_cpu_)
        return static_cast<std::uint64_t>(paras_host_detail::kHostLocalMemBytes);
    paras_extension::gpu_limits g;
    if (is_gpu_ && paras_extension::query_gpu_limits(native_id_, g))
        return g.local_mem;
#endif
    return local_mem_size_;
}

template <>
inline auto device::get_info<info::device::error_correction_support>() const {
    return error_correction_support_;
}

template <>
inline auto device::get_info<info::device::host_unified_memory>() const {
    return host_unified_memory_;
}

template <>
inline auto device::get_info<info::device::profiling_timer_resolution>() const {
    return profiling_timer_resolution_;
}

template <>
inline auto device::get_info<info::device::is_endian_little>() const {
    return is_endian_little_;
}

template <>
inline auto device::get_info<info::device::is_available>() const {
    return is_available_;
}

template <>
inline auto device::get_info<info::device::built_in_kernels>() const {
    return std::vector<std::string>{};
}

template <>
inline auto device::get_info<info::device::platform>() const {
    return get_platform();
}

template <>
inline auto device::get_info<info::device::backend_version>() const {
    return std::string{};
}

template <>
inline auto device::get_info<info::device::aspects>() const {
    std::vector<sycl::aspect> result;
    for (sycl::aspect a :
         {sycl::aspect::cpu, sycl::aspect::gpu, sycl::aspect::accelerator, sycl::aspect::fp64,
          sycl::aspect::atomic64, sycl::aspect::queue_profiling,
          sycl::aspect::usm_device_allocations, sycl::aspect::usm_host_allocations,
          sycl::aspect::usm_shared_allocations}) {
        if (has(a))
            result.push_back(a);
    }
    return result;
}

template <>
inline auto device::get_info<info::device::extensions>() const {
    return std::vector<std::string>{};
}

template <>
inline auto device::get_info<info::device::printf_buffer_size>() const {
    return printf_buffer_size_;
}

template <>
inline auto device::get_info<info::device::parent_device>() const {
    return *this;
}

template <>
inline auto device::get_info<info::device::partition_max_sub_devices>() const {
    return partition_max_sub_devices_;
}

template <>
inline auto device::get_info<info::device::partition_type_property>() const {
    return sycl::info::partition_property::no_partition;
}

template <>
inline auto device::get_info<info::device::partition_type_affinity_domain>() const {
    return sycl::info::partition_affinity_domain::not_applicable;
}

template <>
inline auto device::get_info<info::device::sub_group_sizes>() const {
    if (is_gpu_) {
#if (PARAS_HIP_BACKEND)
        return std::vector<std::size_t>{64};
#elif (PARAS_CUDA_BACKEND)
        return std::vector<std::size_t>{32};
#else
        return std::vector<std::size_t>{1};
#endif
    }
    return std::vector<std::size_t>{1};
}

template <>
inline auto device::get_info<info::device::atomic_fence_order_capabilities>() const {
    return std::vector<memory_order>{memory_order::relaxed, memory_order::acquire,
                                     memory_order::release, memory_order::acq_rel,
                                     memory_order::seq_cst};
}

template <>
inline auto device::get_info<info::device::atomic_fence_scope_capabilities>() const {
    return std::vector<memory_scope>{memory_scope::work_item, memory_scope::sub_group,
                                     memory_scope::work_group, memory_scope::device,
                                     memory_scope::system};
}

template <>
inline auto device::get_info<info::device::partition_properties>() const {
    return std::vector<info::partition_property>{info::partition_property::no_partition};
}

template <>
inline auto device::get_info<info::device::partition_affinity_domains>() const {
    return std::vector<info::partition_affinity_domain>{
        info::partition_affinity_domain::not_applicable};
}

template <>
inline auto device::get_info<info::device::atomic_memory_order_capabilities>() const {

    return std::vector<memory_order>{memory_order::relaxed, memory_order::acquire,
                                     memory_order::release, memory_order::acq_rel,
                                     memory_order::seq_cst};
}

template <>
inline auto device::get_info<info::device::atomic_memory_scope_capabilities>() const {

    return std::vector<memory_scope>{memory_scope::work_item, memory_scope::sub_group,
                                     memory_scope::work_group, memory_scope::device,
                                     memory_scope::system};
}

template <>
inline auto device::get_info<info::device::local_mem_type>() const {
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

template <>
inline auto device::get_info<info::device::device_type>() const {
    if (is_cpu_)
        return info::device_type::cpu;
    if (is_gpu_)
        return info::device_type::gpu;
    if (is_accelerator_)
        return info::device_type::accelerator;
    return info::device_type::host;
}

template <>
inline auto device::get_info<info::device::is_compiler_available>() const {
    return false;
}

template <>
inline auto device::get_info<info::device::is_linker_available>() const {
    return false;
}

template <>
struct backend_return<backend::cuda, device> {
    using type = int;
};

template <>
struct backend_return<backend::hip, device> {
    using type = int;
};

template <>
struct backend_return<backend::host, device> {
    using type = int;
};

template <>
struct backend_return<backend::cuda, queue> {
#if (PARAS_CUDA_BACKEND)
    using type = CUstream_st*;
#else
    using type = void*;
#endif
};

template <>
struct backend_return<backend::hip, queue> {
#if (PARAS_HIP_BACKEND)
    using type = hipStream_t;
#else
    using type = void*;
#endif
};

template <>
struct backend_return<backend::host, queue> {
    using type = void*;
};

template <>
inline int get_native<backend::cuda, device>(const device& dev) {
    return dev.get_native_id();
}

template <>
inline int get_native<backend::hip, device>(const device& dev) {
    return dev.get_native_id();
}

template <>
inline int get_native<backend::host, device>(const device& dev) {
    return dev.get_native_id();
}
} // namespace sycl

namespace std {
template <>
struct hash<sycl::device> {
    std::size_t operator()(const sycl::device& d) const noexcept {
        std::size_t h = std::hash<int>{}(d.get_native_id());
        h ^= std::hash<bool>{}(d.is_cpu()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(d.is_gpu()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(d.is_accelerator()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

template <>
struct hash<sycl::platform> {
    std::size_t operator()(const sycl::platform& p) const noexcept {
        return std::hash<std::string>{}(p.get_name());
    }
};

} // namespace std

#endif
