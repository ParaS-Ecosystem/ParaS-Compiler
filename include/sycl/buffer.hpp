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

#ifndef __PARAS_BUFFER_HPP__
#define __PARAS_BUFFER_HPP__

#include <memory>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "kem_gpu/gpu_utilities.hpp"
#include "range.hpp"
#include "event.hpp"
#include "property_list.hpp"
#include "access.hpp"
#include "handler.hpp"
#include "accessor.hpp"
#include "exception.hpp"
#if PARAS_GPU_BACKEND && !PARAS_DEVICE_PASS
#include "kem_gpu/gpu_runtime_api.hpp"
#endif

namespace sycl {

template <typename T>
class buffer_allocator : public std::allocator<T> {
public:
    buffer_allocator() = default;
    template <typename U>
    buffer_allocator(const buffer_allocator<U>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = buffer_allocator<U>;
    };
};

template <typename T, typename U>
bool operator==(const buffer_allocator<T>&, const buffer_allocator<U>&) noexcept {
    return true;
}
template <typename T, typename U>
bool operator!=(const buffer_allocator<T>&, const buffer_allocator<U>&) noexcept {
    return false;
}

namespace detail {

struct buffer_mirror {
    char *host_base = nullptr;
    std::size_t bytes = 0;
    void *dst = nullptr;
    bool write_back = false;
    std::shared_ptr<void> keep;
    void *dev = nullptr;
    std::mutex m;
    buffer_mirror() = default;
    buffer_mirror(const buffer_mirror &) = delete;
    buffer_mirror &operator=(const buffer_mirror &) = delete;
    ~buffer_mirror();

    char *storage();
};

inline std::mutex &buffer_mirror_registry_mutex() {
    static std::mutex *m = new std::mutex;
    return *m;
}
inline std::map<const char *, buffer_mirror *> &buffer_mirror_registry() {
    static auto *r = new std::map<const char *, buffer_mirror *>;
    return *r;
}
inline void buffer_mirror_register(buffer_mirror *m) {
    std::lock_guard<std::mutex> lock(buffer_mirror_registry_mutex());
    buffer_mirror_registry()[static_cast<const char *>(m->dev)] = m;
}
inline void buffer_mirror_unregister(buffer_mirror *m) {
    std::lock_guard<std::mutex> lock(buffer_mirror_registry_mutex());
    buffer_mirror_registry().erase(static_cast<const char *>(m->dev));
}
void buffer_mirror_sync_device();

inline void buffer_mirror_update_host(const void *p) {
    if (!p)
        return;
    buffer_mirror *m = nullptr;
    {
        std::lock_guard<std::mutex> lock(buffer_mirror_registry_mutex());
        auto &r = buffer_mirror_registry();
        auto it = r.upper_bound(static_cast<const char *>(p));
        if (it != r.begin()) {
            --it;
            if (static_cast<const char *>(p) < it->first + it->second->bytes)
                m = it->second;
        }
    }
    if (!m || !m->dst)
        return;
    wait_all_host_tasks();
    buffer_mirror_sync_device();
    std::memcpy(m->dst, m->dev, m->bytes);
}

#if PARAS_GPU_BACKEND && !PARAS_DEVICE_PASS
inline bool buffer_mirror_wanted() {
    static const bool wanted = [] {
        const bool debug = std::getenv("PARAS_BUFFER_MIRROR_DEBUG") != nullptr;

        if (const char *e = std::getenv("PARAS_BUFFER_MIRROR"); e && e[0]) {
            if (debug)
                std::fprintf(stderr, "[ParaS] buffer storage: forced %s\n",
                             e[0] == '1' ? "separate" : "host");
            return e[0] == '1';
        }

        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess)
            return true;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
            return true;
#if PARAS_HIP_BACKEND

        const bool separate = true;
        if (debug)
            std::fprintf(stderr, "[ParaS] buffer storage: separate (%s)\n",
                         prop.gcnArchName);
#else
        const bool separate = prop.pageableMemoryAccess == 0;
        if (debug)
            std::fprintf(stderr, "[ParaS] buffer storage: %s (pageableMemoryAccess=%d)\n",
                         separate ? "separate" : "host", prop.pageableMemoryAccess);
#endif
        return separate;
    }();
    return wanted;
}

struct buffer_mirror_graveyard {
    std::mutex m;

    std::vector<std::pair<void *, std::size_t>> q;
    std::size_t bytes = 0;
    static constexpr std::size_t max_bytes = std::size_t(64) << 20;
    static constexpr std::size_t max_count = 256;
    void release(std::size_t keep_bytes, std::size_t keep_count) {
        std::lock_guard<std::mutex> lock(m);
        while (!q.empty() && (bytes > keep_bytes || q.size() > keep_count)) {
            cudaFree(q.front().first);
            bytes -= q.front().second;
            q.erase(q.begin());
        }
    }
    void bury(void *p, std::size_t n) {
        {
            std::lock_guard<std::mutex> lock(m);
            q.emplace_back(p, n);
            bytes += n;
        }
        release(max_bytes, max_count);
    }
};
inline buffer_mirror_graveyard &buffer_mirror_dead() {
    static auto *g = new buffer_mirror_graveyard;
    return *g;
}
inline void *buffer_mirror_alloc(std::size_t bytes) {
    void *p = nullptr;
    cudaError_t err = cudaMallocManaged(&p, bytes ? bytes : 1);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        buffer_mirror_dead().release(0, 0);
        err = cudaMallocManaged(&p, bytes ? bytes : 1);
    }
    if (err != cudaSuccess)
        throw sycl::exception(make_error_code(errc::memory_allocation),
                              std::string("buffer: cannot allocate managed memory (") +
                                  std::to_string(bytes) + " bytes): " +
                                  cudaGetErrorString(err));
    return p;
}
inline void buffer_mirror_sync_device() { cudaDeviceSynchronize(); }

inline void buffer_host_access_sync() {
    if (!tl_in_host_task)
        cudaDeviceSynchronize();
}
inline void gpu_device_sync() { cudaDeviceSynchronize(); }

inline void buffer_mirror_copy(void *to, const void *from, std::size_t n) {
#if PARAS_HIP_BACKEND
    static const bool runtime = [] {
        const char *e = std::getenv("PARAS_BUFFER_RUNTIME_COPY");
        return e != nullptr && e[0] == '1';
    }();
    if (runtime && n != 0 &&
        cudaMemcpy(to, from, n, cudaMemcpyDefault) == cudaSuccess)
        return;
    (void)cudaGetLastError();
#endif
    std::memcpy(to, from, n);
}
inline char *buffer_mirror::storage() {
    std::lock_guard<std::mutex> lock(m);
    if (!dev) {
        dev = buffer_mirror_alloc(bytes);
        if (host_base)
            buffer_mirror_copy(dev, host_base, bytes);
        buffer_mirror_register(this);
    }
    return static_cast<char *>(dev);
}
inline buffer_mirror::~buffer_mirror() {
    if (!dev)
        return;
    buffer_mirror_unregister(this);
    wait_all_host_tasks();
    cudaDeviceSynchronize();
    if (write_back && dst)
        buffer_mirror_copy(dst, dev, bytes);
    buffer_mirror_dead().bury(dev, bytes);
}
#elif !PARAS_DEVICE_PASS

inline bool buffer_mirror_wanted() {
    static const bool wanted = [] {
        const char *e = std::getenv("PARAS_BUFFER_MIRROR");
        return e && e[0] == '1';
    }();
    return wanted;
}
inline void *buffer_mirror_alloc(std::size_t bytes) {
    return std::calloc(bytes ? bytes : 1, 1);
}
inline void buffer_mirror_sync_device() {}
inline void buffer_host_access_sync() {}
inline void gpu_device_sync() {}
inline char *buffer_mirror::storage() {
    std::lock_guard<std::mutex> lock(m);
    if (!dev) {
        dev = buffer_mirror_alloc(bytes);
        if (host_base)
            std::memcpy(dev, host_base, bytes);
        buffer_mirror_register(this);
    }
    return static_cast<char *>(dev);
}
inline buffer_mirror::~buffer_mirror() {
    if (!dev)
        return;
    buffer_mirror_unregister(this);
    wait_all_host_tasks();
    if (write_back && dst)
        std::memcpy(dst, dev, bytes);

}
#else
inline bool buffer_mirror_wanted() { return false; }
inline void *buffer_mirror_alloc(std::size_t) { return nullptr; }
inline void buffer_mirror_sync_device() {}
inline void buffer_host_access_sync() {}
inline void gpu_device_sync() {}
inline char *buffer_mirror::storage() { return host_base; }
inline buffer_mirror::~buffer_mirror() {}
#endif

struct buffer_view_tag {};
}

template <
    typename T,
    int Dimensions = 1,
    typename AllocatorT = buffer_allocator<T>>
class buffer
{
    T* data_;
    std::shared_ptr<detail::buffer_mirror> mirror_;

    void paras_mirror(std::size_t n, T *back, std::shared_ptr<void> keep = {}) {
        if (!detail::buffer_mirror_wanted() ||
            props_.template has_property<property::buffer::use_host_ptr>())
            return;
        auto m = std::make_shared<detail::buffer_mirror>();
        m->host_base = const_cast<char *>(reinterpret_cast<const char *>(data_));
        m->bytes = n * sizeof(T);
        m->dst = const_cast<void *>(static_cast<const void *>(back));
        m->write_back = back != nullptr;
        m->keep = std::move(keep);
        mirror_ = std::move(m);
    }

    range<Dimensions> r_;
    property_list props_;
    bool is_sub_buffer_ = false;

    template <typename>
    friend struct std::hash;

public:

    ~buffer() { ::sycl::detail::wait_all_host_tasks(); }
    buffer(const buffer &) = default;
    buffer(buffer &&) = default;
    buffer &operator=(const buffer &) = default;
    buffer &operator=(buffer &&) = default;

    using value_type      = T;
    using reference       = T&;
    using const_reference = const T&;
    using allocator_type  = AllocatorT;

    buffer(const range<Dimensions>& rr,
           const property_list& propList = {})
        : r_(rr), props_(propList)
    {

        size_t N = rr.size();
        data_ = static_cast<T*>(::operator new[](N * sizeof(T)));
        paras_mirror(N, nullptr);
    }

    buffer(T* hostPtr,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : data_(hostPtr), r_(rr), props_(propList)
    {
        if (hostPtr)
            paras_mirror(rr.size(), hostPtr);
    }

    buffer(detail::buffer_view_tag, T *p, const range<Dimensions> &rr,
           const property_list &propList,
           std::shared_ptr<detail::buffer_mirror> m)
        : data_(p), r_(rr), props_(propList), mirror_(std::move(m)) {}

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    buffer(const T* hostPtr,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : r_(rr), props_(propList)
    {

        size_t N = rr.size();
        data_ = static_cast<T*>(::operator new[](N * sizeof(T)));
        for (size_t i = 0; i < N; ++i) ::new (data_ + i) T(hostPtr[i]);
        paras_mirror(N, nullptr);
    }

    buffer(const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(rr, propList) {}

    buffer(T* hostPtr, const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostPtr, rr, propList) {}

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    buffer(const T* hostPtr, const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostPtr, rr, propList) {}

    template <typename InputIterator>
    buffer(InputIterator first, InputIterator last,
           const property_list& propList = {})
        : r_(range<Dimensions>(static_cast<std::size_t>(std::distance(first, last)))), props_(propList)
    {

        std::size_t n = static_cast<std::size_t>(std::distance(first, last));
        data_ = static_cast<T*>(::operator new[](n * sizeof(T)));
        std::size_t i = 0;
        for (auto it = first; it != last; ++it, ++i) ::new (data_ + i) T(*it);
        paras_mirror(n, nullptr);
    }

    template <typename InputIterator>
    buffer(InputIterator first, InputIterator last, AllocatorT,
           const property_list& propList = {})
        : buffer(first, last, propList) {}

    template <typename Container,
              typename = decltype(std::declval<Container&>().data()),
              typename = decltype(std::declval<Container&>().size()),
              int D = Dimensions,
              typename = std::enable_if_t<D == 1 &&
                  !std::is_same_v<std::decay_t<Container>, buffer>>>
    buffer(Container& container, AllocatorT,
           const property_list& propList = {})
        : buffer(container.data(), range<1>(container.size()), propList) {}

    template <typename Container,
              typename = decltype(std::declval<Container&>().data()),
              typename = decltype(std::declval<Container&>().size()),
              int D = Dimensions,
              typename = std::enable_if_t<D == 1 &&
                  !std::is_same_v<std::decay_t<Container>, buffer>>>
    buffer(Container& container,
           const property_list& propList = {})
        : buffer(container.data(), range<1>(container.size()), propList) {}

    buffer(const std::shared_ptr<T>& hostData,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : data_(hostData.get()), r_(rr), props_(propList)
    {
        if (hostData)
            paras_mirror(rr.size(), hostData.get(), hostData);
    }

    buffer(const std::shared_ptr<T>& hostData,
           const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostData, rr, propList) {}

    buffer(const std::shared_ptr<T[]>& hostData,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : data_(hostData.get()), r_(rr), props_(propList)
    {
        if (hostData)
            paras_mirror(rr.size(), hostData.get(),
                         std::shared_ptr<void>(hostData, hostData.get()));
    }

    buffer(const std::shared_ptr<T[]>& hostData,
           const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostData, rr, propList) {}

    template <typename PropT>
    bool has_property() const noexcept {
        return props_.template has_property<PropT>();
    }

    template <typename PropT>
    PropT get_property() const {
        return props_.template get_property<PropT>();
    }

    template <typename T2>
    void set_final_data(T2 dst) {
        if (!mirror_)
            return;
        if constexpr (std::is_convertible_v<T2, T *>) {
            mirror_->dst = const_cast<void *>(static_cast<const void *>(static_cast<T *>(dst)));
            mirror_->write_back = mirror_->dst != nullptr;
        } else if constexpr (std::is_convertible_v<T2, std::weak_ptr<T>>) {
            auto sp = std::weak_ptr<T>(dst).lock();
            mirror_->dst = const_cast<void *>(static_cast<const void *>(sp.get()));
            mirror_->keep = sp;
            mirror_->write_back = sp != nullptr;
        } else {

        }
    }
    void set_final_data(std::nullptr_t) {
        if (mirror_)
            mirror_->write_back = false;
    }

    void set_write_back(bool flag = true) {
        if (mirror_)
            mirror_->write_back = flag && mirror_->dst != nullptr;
    }

    bool is_sub_buffer() const noexcept { return is_sub_buffer_; }
    AllocatorT get_allocator() const { return AllocatorT{}; }

    buffer(buffer& b, id<Dimensions> baseIndex, range<Dimensions> subRange)
        : r_(subRange), props_(b.props_), is_sub_buffer_(true)
    {
        std::size_t flat_offset = baseIndex[0];
        if constexpr (Dimensions == 2) {
            flat_offset = baseIndex[0] * b.r_[1] + baseIndex[1];
        } else if constexpr (Dimensions == 3) {
            flat_offset = (baseIndex[0] * b.r_[1] + baseIndex[1]) * b.r_[2] + baseIndex[2];
        }
        data_ = b.data_ + flat_offset;
        mirror_ = b.mirror_;
    }

    friend bool operator==(const buffer &lhs, const buffer &rhs) noexcept {
        return lhs.data_ == rhs.data_;
    }
    friend bool operator!=(const buffer &lhs, const buffer &rhs) noexcept {
        return !(lhs == rhs);
    }

    template <typename ReinterpretT, int ReinterpretDim = Dimensions>
    buffer<ReinterpretT, ReinterpretDim> reinterpret(range<ReinterpretDim> reinterpretRange) const
    {
        return buffer<ReinterpretT, ReinterpretDim>(
            detail::buffer_view_tag{}, reinterpret_cast<ReinterpretT*>(data_),
            reinterpretRange, props_, mirror_);
    }

    template <typename ReinterpretT, int ReinterpretDim = Dimensions>
    buffer<ReinterpretT, ReinterpretDim> reinterpret() const
    {
        std::size_t total_bytes = r_.size() * sizeof(T);
        std::size_t new_count = total_bytes / sizeof(ReinterpretT);
        if constexpr (ReinterpretDim == 1) {
            return reinterpret<ReinterpretT, 1>(range<1>(new_count));
        } else {
            range<ReinterpretDim> new_range;
            for (int i = 0; i < ReinterpretDim; ++i) new_range[i] = r_[i];
            new_range[ReinterpretDim - 1] =
                (r_[Dimensions - 1] * sizeof(T)) / sizeof(ReinterpretT);
            return reinterpret<ReinterpretT, ReinterpretDim>(new_range);
        }
    }

    template <
        access::mode Mode = access::mode::read_write,
        target AccessTarget = target::device>
    accessor<T, Dimensions, Mode, AccessTarget>
    get_access(handler& cgh)
    {

        return accessor<T, Dimensions, Mode, AccessTarget>(*this, cgh);
    }

    template <
        access::mode Mode = access::mode::read_write,
        target AccessTarget = target::device>
    accessor<T, Dimensions, Mode, AccessTarget>
    get_access(handler& cgh, range<Dimensions> accessRange, id<Dimensions> accessOffset)
    {

        return accessor<T, Dimensions, Mode, AccessTarget>(
            *this, cgh, accessRange, accessOffset);
    }

    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer>
    get_access()
    {

        return accessor<T, Dimensions, Mode, target::host_buffer>(*this);
    }

    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer>
    get_access(range<Dimensions> accessRange, id<Dimensions> accessOffset)
    {
        return accessor<T, Dimensions, Mode, target::host_buffer>(
            *this, accessRange, accessOffset);
    }

    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer>
    get_access(range<Dimensions> accessRange)
    {
        return get_access<Mode>(accessRange, id<Dimensions>());
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::device>
    get_access(handler& cgh, TagT)
    {
        return accessor<T, Dimensions, TagT::value, target::device>(*this, cgh);
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::device>
    get_access(handler& cgh, TagT, property::no_init)
    {
        return get_access(cgh, TagT{});
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::device>
    get_access(handler& cgh, range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT)
    {
        return get_access<TagT::value, target::device>(cgh, accessRange, accessOffset);
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::host_buffer>
    get_access(TagT)
    {
        return get_access<TagT::value>();
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::host_buffer>
    get_access(range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT)
    {
        return get_access<TagT::value>(accessRange, accessOffset);
    }

    accessor<T, Dimensions, access::mode::read_write, target::host_buffer>
    get_host_access()
    {
        return accessor<T, Dimensions, access::mode::read_write, target::host_buffer>(*this);
    }

    T* data() const {
        if (!mirror_)
            return data_;
        return reinterpret_cast<T *>(
            mirror_->storage() +
            (reinterpret_cast<const char *>(data_) - mirror_->host_base));
    }

    range<Dimensions> get_range() const noexcept { return r_; }

    std::size_t size() const noexcept { return r_.size(); }

    std::size_t byte_size() const noexcept { return r_.size() * sizeof(T); }

    [[deprecated("use size()")]] std::size_t get_count() const { return size(); }
    [[deprecated("use byte_size()")]] std::size_t get_size() const { return byte_size(); }

};

template <typename InputIterator,
          typename T = typename std::iterator_traits<InputIterator>::value_type>
buffer(InputIterator, InputIterator, const property_list& = {})
    -> buffer<T, 1, buffer_allocator<T>>;

template <typename InputIterator, typename AllocatorT,
          typename T = typename std::iterator_traits<InputIterator>::value_type>
buffer(InputIterator, InputIterator, AllocatorT, const property_list& = {})
    -> buffer<T, 1, AllocatorT>;

template <typename Container,
          typename T = typename Container::value_type>
buffer(Container&, const property_list& = {})
    -> buffer<T, 1, buffer_allocator<T>>;

template <typename Container, typename AllocatorT,
          typename T = typename Container::value_type>
buffer(Container&, AllocatorT, const property_list& = {})
    -> buffer<T, 1, AllocatorT>;

}

template <typename T, int Dimensions, typename AllocatorT>
struct std::hash<sycl::buffer<T, Dimensions, AllocatorT>> {
    std::size_t operator()(const sycl::buffer<T, Dimensions, AllocatorT>& b) const noexcept {
        return std::hash<const void*>{}(static_cast<const void*>(b.data_));
    }
};

#endif
