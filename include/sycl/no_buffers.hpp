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

#ifndef __PARAS_NO_BUFFERS_HPP__
#define __PARAS_NO_BUFFERS_HPP__

#include <memory>
#include "accessor.hpp"
#if PARAS_GPU_BACKEND && !PARAS_DEVICE_PASS
#include "kem_gpu/gpu_runtime_api.hpp"
#endif

namespace sycl
{
    namespace detail
    {
        struct paras_buffers_disabled_t {};

        template <typename...>
        inline constexpr bool paras_buffers_enabled = false;

#if PARAS_GPU_BACKEND && !PARAS_DEVICE_PASS
        inline void gpu_device_sync() { cudaDeviceSynchronize(); }
#else
        inline void gpu_device_sync() {}
#endif
    }

    template <typename T>
    class buffer_allocator : public std::allocator<T>
    {
    public:
        buffer_allocator() = default;
        template <typename U>
        buffer_allocator(const buffer_allocator<U> &) noexcept {}

        template <typename U>
        struct rebind
        {
            using other = buffer_allocator<U>;
        };
    };

    template <
        typename T,
        int Dimensions = 1,
        typename AllocatorT = buffer_allocator<T>>
    class buffer
    {
        static_assert(detail::paras_buffers_enabled<T>,
                      "sycl::buffer is not available: this ParaS build has "
                      "PARAS_NO_BUFFERS=1. Use USM (sycl::malloc_device / "
                      "sycl::malloc_shared) instead.");

    public:
        template <typename... Args>
        buffer(Args &&...) {}
    };

    template <typename... Args>
    buffer(Args &&...) -> buffer<detail::paras_buffers_disabled_t>;

    template <
        typename DataT,
        int Dimensions = 1,
        access::mode AccessMode = access::mode::read_write,
        target AccessTarget = target::device,
        access::placeholder IsPlaceholder = access::placeholder::false_t>
    class accessor
    {
        static_assert(detail::paras_buffers_enabled<DataT>,
                      "sycl::accessor is not available: this ParaS build has "
                      "PARAS_NO_BUFFERS=1. Use USM pointers instead "
                      "(sycl::local_accessor is still available).");

    public:
        template <typename... Args>
        accessor(Args &&...) {}
    };

    template <typename... Args>
    accessor(Args &&...) -> accessor<detail::paras_buffers_disabled_t>;

    template <
        typename DataT,
        int Dimensions = 1,
        access::mode AccessMode = access::mode::read_write>
    class host_accessor
    {
        static_assert(detail::paras_buffers_enabled<DataT>,
                      "sycl::host_accessor is not available: this ParaS build "
                      "has PARAS_NO_BUFFERS=1. Use USM instead.");

    public:
        template <typename... Args>
        host_accessor(Args &&...) {}
    };

    template <typename... Args>
    host_accessor(Args &&...) -> host_accessor<detail::paras_buffers_disabled_t>;
}

#endif
