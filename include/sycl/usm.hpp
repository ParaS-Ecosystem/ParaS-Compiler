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

#ifndef __PARAS_USM_HPP__
#define __PARAS_USM_HPP__

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include "context.hpp"
#include "device.hpp"
#include "property_list.hpp"
#include "queue.hpp"
#include "usm_alloc.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

namespace paras_usm_detail {

template <typename Q, typename = void>
struct is_queue_like : std::false_type {};
template <typename Q>
struct is_queue_like<Q, std::void_t<decltype(std::declval<const Q&>().get_device()),
                                    decltype(std::declval<const Q&>().get_context())>>
    : std::true_type {};
template <typename Q>
inline constexpr bool is_queue_like_v = is_queue_like<std::decay_t<Q>>::value;

template <typename Q>
using if_queue = std::enable_if_t<is_queue_like_v<Q>, int>;

inline device first_device(const context& ctx) {
    const std::vector<device> devs = ctx.get_devices();
    return devs.empty() ? device{} : devs.front();
}

inline bool is_power_of_two(std::size_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

inline void* allocate(std::size_t bytes, std::size_t alignment, usm::alloc kind,
                      const device& dev) {
    if (bytes == 0 || kind == usm::alloc::unknown)
        return nullptr;
    if (alignment != 0 && !is_power_of_two(alignment))
        return nullptr;
#if PARAS_CUDA_BACKEND
    if (dev.is_gpu()) {
        if (cudaSetDevice(dev.get_native_id()) != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        void* p = nullptr;
        const cudaError_t err = kind == usm::alloc::device   ? cudaMalloc(&p, bytes)
                                : kind == usm::alloc::shared ? cudaMallocManaged(&p, bytes)
                                                             : cudaMallocHost(&p, bytes);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        if (alignment != 0 && reinterpret_cast<std::uintptr_t>(p) % alignment != 0) {
            kind == usm::alloc::host ? (void)cudaFreeHost(p) : (void)cudaFree(p);
            return nullptr;
        }
        return p;
    }
#elif PARAS_HIP_BACKEND
    if (dev.is_gpu()) {
        if (hipSetDevice(dev.get_native_id()) != hipSuccess) {
            (void)hipGetLastError();
            return nullptr;
        }
        void* p = nullptr;
        const hipError_t err = kind == usm::alloc::device ? hipMalloc(&p, bytes)
                               : kind == usm::alloc::shared
                                   ? hipMallocManaged(&p, bytes)
                                   : hipHostMalloc(&p, bytes, hipHostMallocDefault);
        if (err != hipSuccess) {
            (void)hipGetLastError();
            return nullptr;
        }
        if (alignment != 0 && reinterpret_cast<std::uintptr_t>(p) % alignment != 0) {
            kind == usm::alloc::host ? (void)hipHostFree(p) : (void)hipFree(p);
            return nullptr;
        }
        return p;
    }
#else
    (void)dev;
#endif
    std::size_t align = alignment ? alignment : alignof(std::max_align_t);
    if (align < sizeof(void*))
        align = sizeof(void*);
    void* p = nullptr;
    if (::posix_memalign(&p, align, bytes) != 0)
        return nullptr;
    register_host_alloc(p, bytes, kind);
    return p;
}

template <typename T>
T* allocate_typed(std::size_t count, std::size_t alignment, usm::alloc kind, const device& dev) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return nullptr;
    return static_cast<T*>(allocate(count * sizeof(T), alignment, kind, dev));
}

inline void deallocate(void* ptr) {
    if (ptr == nullptr)
        return;
#if PARAS_CUDA_BACKEND
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess ||
        attr.type == cudaMemoryTypeUnregistered) {
        (void)cudaGetLastError();
        unregister_host_alloc(ptr);
        std::free(ptr);
        return;
    }
    if (attr.device >= 0)
        (void)cudaSetDevice(attr.device);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = attr.type == cudaMemoryTypeHost ? cudaFreeHost(ptr) : cudaFree(ptr);
    }
    if (err != cudaSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("sycl::free failed: ") + cudaGetErrorString(err));
    }
#elif PARAS_HIP_BACKEND
    hipPointerAttribute_t attr{};
    if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
        (void)hipGetLastError();
        unregister_host_alloc(ptr);
        std::free(ptr);
        return;
    }
#if HIP_VERSION >= 50000000
    const hipMemoryType type = attr.type;
#else
    const hipMemoryType type = attr.memoryType;
#endif
    if (attr.device >= 0)
        (void)hipSetDevice(attr.device);
    hipError_t err = hipDeviceSynchronize();
    if (err == hipSuccess) {
        err = type == hipMemoryTypeHost ? hipHostFree(ptr) : hipFree(ptr);
    }
    if (err != hipSuccess) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              std::string("sycl::free failed: ") + hipGetErrorString(err));
    }
#else
    unregister_host_alloc(ptr);
    std::free(ptr);
#endif
}

} // namespace paras_usm_detail

#define PARAS_USM_KIND_FUNCTIONS(NAME, KIND)                                                       \
    template <typename T, typename Q, paras_usm_detail::if_queue<Q> = 0>                           \
    T* malloc_##NAME(std::size_t count, const Q& q, const property_list& = {}) {                   \
        return paras_usm_detail::allocate_typed<T>(count, 0, KIND, q.get_device());                \
    }                                                                                              \
    template <typename T>                                                                          \
    T* malloc_##NAME(std::size_t count, const device& dev, const context&,                         \
                     const property_list& = {}) {                                                  \
        return paras_usm_detail::allocate_typed<T>(count, 0, KIND, dev);                           \
    }                                                                                              \
    template <typename T>                                                                          \
    T* malloc_##NAME(std::size_t count, const context& ctx, const property_list& = {}) {           \
        return paras_usm_detail::allocate_typed<T>(count, 0, KIND,                                 \
                                                   paras_usm_detail::first_device(ctx));           \
    }                                                                                              \
    template <typename Q, paras_usm_detail::if_queue<Q> = 0>                                       \
    void* malloc_##NAME(std::size_t bytes, const Q& q, const property_list& = {}) {                \
        return paras_usm_detail::allocate(bytes, 0, KIND, q.get_device());                         \
    }                                                                                              \
    inline void* malloc_##NAME(std::size_t bytes, const device& dev, const context&,               \
                               const property_list& = {}) {                                        \
        return paras_usm_detail::allocate(bytes, 0, KIND, dev);                                    \
    }                                                                                              \
    inline void* malloc_##NAME(std::size_t bytes, const context& ctx, const property_list& = {}) { \
        return paras_usm_detail::allocate(bytes, 0, KIND, paras_usm_detail::first_device(ctx));    \
    }                                                                                              \
    template <typename T, typename Q, paras_usm_detail::if_queue<Q> = 0>                           \
    T* aligned_alloc_##NAME(std::size_t alignment, std::size_t count, const Q& q,                  \
                            const property_list& = {}) {                                           \
        return paras_usm_detail::allocate_typed<T>(count, alignment, KIND, q.get_device());        \
    }                                                                                              \
    template <typename T>                                                                          \
    T* aligned_alloc_##NAME(std::size_t alignment, std::size_t count, const device& dev,           \
                            const context&, const property_list& = {}) {                           \
        return paras_usm_detail::allocate_typed<T>(count, alignment, KIND, dev);                   \
    }                                                                                              \
    template <typename T>                                                                          \
    T* aligned_alloc_##NAME(std::size_t alignment, std::size_t count, const context& ctx,          \
                            const property_list& = {}) {                                           \
        return paras_usm_detail::allocate_typed<T>(count, alignment, KIND,                         \
                                                   paras_usm_detail::first_device(ctx));           \
    }                                                                                              \
    template <typename Q, paras_usm_detail::if_queue<Q> = 0>                                       \
    void* aligned_alloc_##NAME(std::size_t alignment, std::size_t bytes, const Q& q,               \
                               const property_list& = {}) {                                        \
        return paras_usm_detail::allocate(bytes, alignment, KIND, q.get_device());                 \
    }                                                                                              \
    inline void* aligned_alloc_##NAME(std::size_t alignment, std::size_t bytes, const device& dev, \
                                      const context&, const property_list& = {}) {                 \
        return paras_usm_detail::allocate(bytes, alignment, KIND, dev);                            \
    }                                                                                              \
    inline void* aligned_alloc_##NAME(std::size_t alignment, std::size_t bytes,                    \
                                      const context& ctx, const property_list& = {}) {             \
        return paras_usm_detail::allocate(bytes, alignment, KIND,                                  \
                                          paras_usm_detail::first_device(ctx));                    \
    }

PARAS_USM_KIND_FUNCTIONS(device, usm::alloc::device)
PARAS_USM_KIND_FUNCTIONS(host, usm::alloc::host)
PARAS_USM_KIND_FUNCTIONS(shared, usm::alloc::shared)

#undef PARAS_USM_KIND_FUNCTIONS

template <typename T, typename Q, paras_usm_detail::if_queue<Q> = 0>
T* malloc(std::size_t count, const Q& q, usm::alloc kind, const property_list& = {}) {
    return paras_usm_detail::allocate_typed<T>(count, 0, kind, q.get_device());
}
template <typename T>
T* malloc(std::size_t count, const device& dev, const context&, usm::alloc kind,
          const property_list& = {}) {
    return paras_usm_detail::allocate_typed<T>(count, 0, kind, dev);
}
template <typename Q, paras_usm_detail::if_queue<Q> = 0>
void* malloc(std::size_t bytes, const Q& q, usm::alloc kind, const property_list& = {}) {
    return paras_usm_detail::allocate(bytes, 0, kind, q.get_device());
}
inline void* malloc(std::size_t bytes, const device& dev, const context&, usm::alloc kind,
                    const property_list& = {}) {
    return paras_usm_detail::allocate(bytes, 0, kind, dev);
}
template <typename T, typename Q, paras_usm_detail::if_queue<Q> = 0>
T* aligned_alloc(std::size_t alignment, std::size_t count, const Q& q, usm::alloc kind,
                 const property_list& = {}) {
    return paras_usm_detail::allocate_typed<T>(count, alignment, kind, q.get_device());
}
template <typename T>
T* aligned_alloc(std::size_t alignment, std::size_t count, const device& dev, const context&,
                 usm::alloc kind, const property_list& = {}) {
    return paras_usm_detail::allocate_typed<T>(count, alignment, kind, dev);
}
template <typename Q, paras_usm_detail::if_queue<Q> = 0>
void* aligned_alloc(std::size_t alignment, std::size_t bytes, const Q& q, usm::alloc kind,
                    const property_list& = {}) {
    return paras_usm_detail::allocate(bytes, alignment, kind, q.get_device());
}
inline void* aligned_alloc(std::size_t alignment, std::size_t bytes, const device& dev,
                           const context&, usm::alloc kind, const property_list& = {}) {
    return paras_usm_detail::allocate(bytes, alignment, kind, dev);
}

template <typename Q, paras_usm_detail::if_queue<Q> = 0>
void free(void* ptr, const Q&) {
    paras_usm_detail::deallocate(ptr);
}
inline void free(void* ptr, const context&) {
    paras_usm_detail::deallocate(ptr);
}

template <typename T, usm::alloc AllocKind, std::size_t Alignment = 0>
class usm_allocator {
    static_assert(AllocKind != usm::alloc::device,
                  "usm_allocator does not support usm::alloc::device");

public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    template <typename U>
    struct rebind {
        using other = usm_allocator<U, AllocKind, Alignment>;
    };

    usm_allocator() = delete;
    usm_allocator(const context&, const device& dev, const property_list& = {}) noexcept
        : dev_(dev) {}
    template <typename Q, paras_usm_detail::if_queue<Q> = 0>
    usm_allocator(const Q& q, const property_list& = {}) noexcept : dev_(q.get_device()) {}
    usm_allocator(const usm_allocator&) noexcept = default;
    usm_allocator(usm_allocator&&) noexcept = default;
    usm_allocator& operator=(const usm_allocator&) = default;
    usm_allocator& operator=(usm_allocator&&) = default;
    template <typename U>
    usm_allocator(const usm_allocator<U, AllocKind, Alignment>& other) noexcept
        : dev_(other.paras_device()) {}

    T* allocate(std::size_t count) {
        T* p = paras_usm_detail::allocate_typed<T>(count, Alignment, AllocKind, dev_);
        if (p == nullptr && count != 0)
            throw std::bad_alloc();
        return p;
    }
    void deallocate(T* ptr, std::size_t) { paras_usm_detail::deallocate(ptr); }

    const device& paras_device() const noexcept { return dev_; }

    template <typename U, usm::alloc OtherKind, std::size_t OtherAlignment>
    bool operator==(const usm_allocator<U, OtherKind, OtherAlignment>& other) const noexcept {
        return AllocKind == OtherKind && dev_ == other.paras_device();
    }
    template <typename U, usm::alloc OtherKind, std::size_t OtherAlignment>
    bool operator!=(const usm_allocator<U, OtherKind, OtherAlignment>& other) const noexcept {
        return !(*this == other);
    }

private:
    device dev_;
};

} // namespace sycl

#endif
