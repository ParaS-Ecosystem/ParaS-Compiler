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

#ifndef __PARAS_KERNEL_HPP__
#define __PARAS_KERNEL_HPP__

#include <algorithm>
#include "context.hpp"
#include "bundle_state.hpp"
#include "paras_kernel_registry.hpp"
#include <functional>
#include <map>
#include <memory>
#include "spec_constant.hpp"

namespace sycl {

class kernel_id;
template <bundle_state State>
class kernel_bundle;
template <bundle_state State>
class device_image;
namespace paras_kb {
struct access;
}

class kernel {
public:
    kernel() = delete;

    sycl::context get_context() const { return context_; }

    sycl::backend get_backend() const noexcept { return context_.get_backend(); }

    kernel_bundle<bundle_state::executable> get_kernel_bundle() const;

    template <typename Param>
    typename Param::return_type get_info() const {
        return typename Param::return_type{};
    }

    template <typename Param>
    typename Param::return_type get_info(const device&) const {
        return typename Param::return_type{};
    }

    bool operator==(const kernel& rhs) const {
        return context_ == rhs.context_ && paras_record_ == rhs.paras_record_;
    }
    bool operator!=(const kernel& rhs) const { return !(*this == rhs); }

private:
    template <bundle_state>
    friend class kernel_bundle;
    friend struct paras_kb::access;

    explicit kernel(sycl::context ctx) : context_(ctx) {}

    kernel(sycl::context ctx, std::vector<device> devs, const paras_kb::kernel_record* rec)
        : context_(ctx), paras_devices_(std::move(devs)), paras_record_(rec) {}

    sycl::context context_;
    std::vector<device> paras_devices_{};
    const paras_kb::kernel_record* paras_record_ = nullptr;
};

template <>
inline std::uint32_t kernel::get_info<info::kernel::num_args>() const {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "info::kernel::num_args is only available for interop or built-in "
                          "kernels");
}
template <>
inline std::string kernel::get_info<info::kernel::attributes>() const {
    return std::string{};
}
template <>
inline range<3>
kernel::get_info<info::kernel_device_specific::global_work_size>(const device& dev) const {
    (void)dev;
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "global_work_size needs a custom device or a built-in kernel");
}
template <>
inline range<3>
kernel::get_info<info::kernel_device_specific::compile_work_group_size>(const device&) const {
    if (paras_record_ && !paras_record_->reqd_work_group.empty()) {
        const auto& w = paras_record_->reqd_work_group;
        if (w.size() == 1)
            return range<3>(1, 1, w[0]);
        if (w.size() == 2)
            return range<3>(1, w[0], w[1]);
        return range<3>(w[0], w[1], w[2]);
    }
    return range<3>(0, 0, 0);
}
template <>
inline std::size_t
kernel::get_info<info::kernel_device_specific::work_group_size>(const device& dev) const {
    return dev.get_info<info::device::max_work_group_size>();
}
template <>
inline std::size_t kernel::get_info<
    info::kernel_device_specific::preferred_work_group_size_multiple>(const device&) const {
    return 1;
}
template <>
inline std::uint32_t
kernel::get_info<info::kernel_device_specific::max_num_sub_groups>(const device& dev) const {
    return std::max<std::uint32_t>(1, dev.get_info<info::device::max_num_sub_groups>());
}
template <>
inline std::uint32_t
kernel::get_info<info::kernel_device_specific::max_sub_group_size>(const device& dev) const {
    const auto sizes = dev.get_info<info::device::sub_group_sizes>();
    std::size_t m = 1;
    for (auto s : sizes)
        m = std::max<std::size_t>(m, s);
    return static_cast<std::uint32_t>(m);
}
template <>
inline std::uint32_t
kernel::get_info<info::kernel_device_specific::compile_sub_group_size>(const device&) const {
    if (paras_record_ && !paras_record_->reqd_sub_group.empty())
        return static_cast<std::uint32_t>(paras_record_->reqd_sub_group.front());
    return 0;
}

class kernel_id {
public:
    kernel_id() = delete;

    const char* get_name() const noexcept {
        return paras_record_ ? paras_record_->name.c_str() : "";
    }

    friend bool operator==(const kernel_id& a, const kernel_id& b) noexcept {
        return a.paras_record_ == b.paras_record_;
    }
    friend bool operator!=(const kernel_id& a, const kernel_id& b) noexcept { return !(a == b); }

private:
    friend struct paras_kb::access;
    explicit kernel_id(const paras_kb::kernel_record* rec) : paras_record_(rec) {}
    const paras_kb::kernel_record* paras_record_ = nullptr;
};

template <bundle_state State>
class device_image {
public:
    device_image() = delete;

    bool has_kernel(const kernel_id& kernelId) const noexcept;
    bool has_kernel(const kernel_id& kernelId, const device& dev) const noexcept;

    friend bool operator==(const device_image& a, const device_image& b) noexcept {
        return a.paras_record_ == b.paras_record_;
    }
    friend bool operator!=(const device_image& a, const device_image& b) noexcept {
        return !(a == b);
    }

private:
    friend struct paras_kb::access;
    explicit device_image(const paras_kb::kernel_record* rec) : paras_record_(rec) {}
    const paras_kb::kernel_record* paras_record_ = nullptr;
};

namespace paras_kb {
struct spec_entry {
    std::shared_ptr<void> value;
    std::function<void()> apply;
    std::function<void()> reset;
};
struct spec_values {
    std::map<const void*, spec_entry> values;
};
}

template <bundle_state State>
class kernel_bundle {
public:
    using device_image_iterator = const device_image<State>*;

    kernel_bundle() = delete;

    bool empty() const noexcept { return paras_images_.empty(); }
    backend get_backend() const noexcept { return paras_context_.get_backend(); }
    context get_context() const noexcept { return paras_context_; }
    std::vector<device> get_devices() const noexcept { return paras_devices_; }

    bool has_kernel(const kernel_id& kernelId) const noexcept {
        for (const auto& img : paras_images_)
            if (img.has_kernel(kernelId))
                return true;
        return false;
    }
    bool has_kernel(const kernel_id& kernelId, const device& dev) const noexcept {
        for (const auto& img : paras_images_)
            if (img.has_kernel(kernelId, dev))
                return true;
        return false;
    }
    template <typename KernelName>
    bool has_kernel() const noexcept;
    template <typename KernelName>
    bool has_kernel(const device& dev) const noexcept;

    std::vector<kernel_id> get_kernel_ids() const;

    template <bundle_state S = State, typename = std::enable_if_t<S == bundle_state::executable>>
    kernel get_kernel(const kernel_id& kernelId) const;
    template <typename KernelName, bundle_state S = State,
              typename = std::enable_if_t<S == bundle_state::executable>>
    kernel get_kernel() const;

    bool contains_specialization_constants() const noexcept { return false; }
    bool native_specialization_constant() const noexcept { return false; }

    template <auto& SpecName, bundle_state S = State,
              typename = std::enable_if_t<S == bundle_state::input>>
    void set_specialization_constant(
        typename std::remove_reference_t<decltype(SpecName)>::value_type value) {
        using T = typename std::remove_reference_t<decltype(SpecName)>::value_type;
        auto v = std::make_shared<T>(value);
        paras_spec_->values[&SpecName] = {v, [v] { paras_spec_detail::set<SpecName>(*v); },
                                          [] { paras_spec_detail::set<SpecName>(SpecName.value_); }};
    }
    template <auto& SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type
    get_specialization_constant() const {
        using T = typename std::remove_reference_t<decltype(SpecName)>::value_type;
        auto it = paras_spec_->values.find(&SpecName);
        if (it != paras_spec_->values.end())
            return *std::static_pointer_cast<T>(it->second.value);
        return SpecName.value_;
    }

    device_image_iterator begin() const { return paras_images_.data(); }
    device_image_iterator end() const { return paras_images_.data() + paras_images_.size(); }

    friend bool operator==(const kernel_bundle& a, const kernel_bundle& b) {
        return a.paras_context_ == b.paras_context_ && a.paras_devices_ == b.paras_devices_ &&
               a.paras_images_ == b.paras_images_;
    }
    friend bool operator!=(const kernel_bundle& a, const kernel_bundle& b) { return !(a == b); }

private:
    friend struct paras_kb::access;
    kernel_bundle(context ctx, std::vector<device> devs, std::vector<device_image<State>> images)
        : paras_context_(std::move(ctx)), paras_devices_(std::move(devs)),
          paras_images_(std::move(images)) {}

    context paras_context_;
    std::vector<device> paras_devices_;
    std::vector<device_image<State>> paras_images_;
    std::shared_ptr<paras_kb::spec_values> paras_spec_ = std::make_shared<paras_kb::spec_values>();
};

namespace paras_kb {

struct access {
    static kernel_id make_id(const kernel_record* r) { return kernel_id(r); }
    static const kernel_record* record(const kernel_id& id) { return id.paras_record_; }
    static const kernel_record* record(const kernel& k) { return k.paras_record_; }
    template <bundle_state State>
    static const kernel_record* record(const device_image<State>& img) {
        return img.paras_record_;
    }
    static kernel make_kernel(const context& ctx, const std::vector<device>& devs,
                              const kernel_record* r) {
        return kernel(ctx, devs, r);
    }
    static const std::vector<device>& kernel_devices(const kernel& k) { return k.paras_devices_; }
    template <bundle_state State>
    static kernel_bundle<State> make_bundle(const context& ctx, std::vector<device> devs,
                                            const std::vector<const kernel_record*>& ks) {
        std::vector<device_image<State>> images;
        for (const auto* r : ks) {
            bool dup = false;
            for (const auto& img : images)
                dup = dup || img.paras_record_ == r;
            if (!dup)
                images.push_back(device_image<State>(r));
        }
        return kernel_bundle<State>(ctx, std::move(devs), std::move(images));
    }
    template <bundle_state State>
    static const paras_kb::spec_values& spec(const kernel_bundle<State>& b) {
        return *b.paras_spec_;
    }
    template <bundle_state From, bundle_state To>
    static void copy_spec(const kernel_bundle<From>& from, kernel_bundle<To>& to) {
        for (const auto& kv : from.paras_spec_->values)
            to.paras_spec_->values[kv.first] = kv.second;
    }
    template <bundle_state State>
    static const std::vector<device_image<State>>& images(const kernel_bundle<State>& b) {
        return b.paras_images_;
    }
};

inline std::vector<device> unique_devices(const std::vector<device>& devs) {
    std::vector<device> out;
    for (const auto& d : devs)
        if (std::find(out.begin(), out.end(), d) == out.end())
            out.push_back(d);
    return out;
}

inline void check_devices(const context& ctx, const std::vector<device>& devs) {
    if (devs.empty())
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "kernel bundle: empty device list");
    const auto ctx_devs = ctx.get_devices();
    for (const auto& d : devs)
        if (std::find(ctx_devs.begin(), ctx_devs.end(), d) == ctx_devs.end())
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "kernel bundle: device is not in the context");
}

inline bool compatible_with_any(const kernel_record& r, const std::vector<device>& devs) {
    for (const auto& d : devs)
        if (compatible(r, d))
            return true;
    return false;
}

inline std::vector<const kernel_record*> compatible_kernels(const std::vector<device>& devs) {
    std::vector<const kernel_record*> out;
    for (const auto& r : records())
        if (compatible_with_any(r, devs))
            out.push_back(&r);
    return out;
}

template <bundle_state State>
inline void require_state_support(const std::vector<device>& devs) {
    for (const auto& d : devs) {
        if (State == bundle_state::input && !d.has(aspect::online_compiler))
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "kernel bundle: device lacks aspect::online_compiler");
        if (State == bundle_state::object && !d.has(aspect::online_linker))
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "kernel bundle: device lacks aspect::online_linker");
    }
}

}

template <bundle_state State>
inline bool device_image<State>::has_kernel(const kernel_id& kernelId) const noexcept {
    return paras_record_ != nullptr && paras_record_ == paras_kb::access::record(kernelId);
}
template <bundle_state State>
inline bool device_image<State>::has_kernel(const kernel_id& kernelId,
                                            const device& dev) const noexcept {
    return has_kernel(kernelId) && paras_kb::compatible(*paras_record_, dev);
}

template <bundle_state State>
template <typename KernelName>
inline bool kernel_bundle<State>::has_kernel() const noexcept {
    const auto* rec = paras_kb::find<KernelName>();
    return rec && has_kernel(paras_kb::access::make_id(rec));
}
template <bundle_state State>
template <typename KernelName>
inline bool kernel_bundle<State>::has_kernel(const device& dev) const noexcept {
    const auto* rec = paras_kb::find<KernelName>();
    return rec && has_kernel(paras_kb::access::make_id(rec), dev);
}
template <bundle_state State>
inline std::vector<kernel_id> kernel_bundle<State>::get_kernel_ids() const {
    std::vector<kernel_id> ids;
    for (const auto& img : paras_images_)
        ids.push_back(paras_kb::access::make_id(paras_kb::access::record(img)));
    return ids;
}
template <bundle_state State>
template <bundle_state S, typename>
inline kernel kernel_bundle<State>::get_kernel(const kernel_id& kernelId) const {
    if (!has_kernel(kernelId))
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "kernel_bundle::get_kernel: kernel not in bundle");
    return paras_kb::access::make_kernel(paras_context_, paras_devices_,
                                         paras_kb::access::record(kernelId));
}
template <bundle_state State>
template <typename KernelName, bundle_state S, typename>
inline kernel kernel_bundle<State>::get_kernel() const {
    const auto* rec = paras_kb::find<KernelName>();
    if (!rec)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "kernel_bundle::get_kernel: kernel not in bundle");
    return get_kernel(paras_kb::access::make_id(rec));
}

inline kernel_bundle<bundle_state::executable> kernel::get_kernel_bundle() const {
    return paras_kb::access::make_bundle<bundle_state::executable>(context_, paras_devices_,
                                                                   {paras_record_});
}

template <typename KernelName>
inline kernel_id get_kernel_id() {
    const paras_kb::kernel_record* rec = paras_kb::find<KernelName>();
    if (!rec)
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              "get_kernel_id: no such kernel in the application");
    return paras_kb::access::make_id(rec);
}

inline std::vector<kernel_id> get_kernel_ids() {
    std::vector<kernel_id> ids;
    for (const auto& r : paras_kb::records())
        ids.push_back(paras_kb::access::make_id(&r));
    return ids;
}

inline bool is_compatible(const std::vector<kernel_id>& kernelIds, const device& dev) {
    for (const auto& id : kernelIds) {
        const auto* r = paras_kb::access::record(id);
        if (!r || !paras_kb::compatible(*r, dev))
            return false;
    }
    return true;
}

template <bundle_state State>
inline kernel_bundle<State> join(const std::vector<kernel_bundle<State>>& bundles) {
    if (bundles.empty()) {
        const context ctx = paras_default_context(platform{});
        return paras_kb::access::make_bundle<State>(ctx, ctx.get_devices(), {});
    }
    const auto& first = bundles.front();
    std::vector<const paras_kb::kernel_record*> ks;
    for (const auto& b : bundles) {
        if (b.get_context() != first.get_context())
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "join: bundles have different contexts");
        auto da = b.get_devices(), db = first.get_devices();
        const bool same_devices =
            da.size() == db.size() &&
            std::all_of(da.begin(), da.end(),
                        [&](const device& d) { return std::find(db.begin(), db.end(), d) != db.end(); });
        if (!same_devices)
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "join: bundles have different device sets");
        for (const auto& img : paras_kb::access::images(b))
            ks.push_back(paras_kb::access::record(img));
    }
    auto out = paras_kb::access::make_bundle<State>(first.get_context(), first.get_devices(), ks);
    for (const auto& b : bundles)
        paras_kb::access::copy_spec(b, out);
    return out;
}

template <bundle_state State>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt, const std::vector<device>& devs) {
    paras_kb::check_devices(ctxt, devs);
    paras_kb::require_state_support<State>(devs);
    const auto udevs = paras_kb::unique_devices(devs);
    return paras_kb::access::make_bundle<State>(ctxt, udevs, paras_kb::compatible_kernels(udevs));
}

template <bundle_state State>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt) {
    return get_kernel_bundle<State>(ctxt, ctxt.get_devices());
}

template <bundle_state State>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt, const std::vector<device>& devs,
                                              const std::vector<kernel_id>& kernelIds) {
    paras_kb::check_devices(ctxt, devs);
    paras_kb::require_state_support<State>(devs);
    const auto udevs = paras_kb::unique_devices(devs);
    std::vector<const paras_kb::kernel_record*> ks;
    for (const auto& id : kernelIds) {
        const auto* r = paras_kb::access::record(id);
        if (!r || !paras_kb::compatible_with_any(*r, udevs))
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "get_kernel_bundle: kernel is incompatible with all devices");
        ks.push_back(r);
    }
    return paras_kb::access::make_bundle<State>(ctxt, udevs, ks);
}

template <bundle_state State>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt,
                                              const std::vector<kernel_id>& kernelIds) {
    return get_kernel_bundle<State>(ctxt, ctxt.get_devices(), kernelIds);
}

template <typename KernelName, bundle_state State>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt, const std::vector<device>& devs) {
    return get_kernel_bundle<State>(ctxt, devs, {get_kernel_id<KernelName>()});
}

template <typename KernelName, bundle_state State>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt) {
    return get_kernel_bundle<State>(ctxt, ctxt.get_devices(), {get_kernel_id<KernelName>()});
}

template <bundle_state State, typename Selector>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt, const std::vector<device>& devs,
                                              Selector selector) {
    paras_kb::check_devices(ctxt, devs);
    paras_kb::require_state_support<State>(devs);
    const auto udevs = paras_kb::unique_devices(devs);
    auto all = paras_kb::access::make_bundle<State>(ctxt, udevs, paras_kb::compatible_kernels(udevs));
    std::vector<const paras_kb::kernel_record*> chosen;
    for (const auto& img : paras_kb::access::images(all))
        if (selector(img))
            chosen.push_back(paras_kb::access::record(img));
    return paras_kb::access::make_bundle<State>(ctxt, udevs, chosen);
}

template <bundle_state State, typename Selector>
inline kernel_bundle<State> get_kernel_bundle(const context& ctxt, Selector selector) {
    return get_kernel_bundle<State>(ctxt, ctxt.get_devices(), selector);
}

template <bundle_state State>
inline bool has_kernel_bundle(const context& ctxt, const std::vector<device>& devs) {
    paras_kb::check_devices(ctxt, devs);
    if constexpr (State != bundle_state::executable)
        return false;
    return !paras_kb::compatible_kernels(devs).empty();
}

template <bundle_state State>
inline bool has_kernel_bundle(const context& ctxt) {
    return has_kernel_bundle<State>(ctxt, ctxt.get_devices());
}

template <bundle_state State>
inline bool has_kernel_bundle(const context& ctxt, const std::vector<device>& devs,
                              const std::vector<kernel_id>& kernelIds) {
    paras_kb::check_devices(ctxt, devs);
    if constexpr (State != bundle_state::executable)
        return false;
    for (const auto& id : kernelIds) {
        const auto* r = paras_kb::access::record(id);
        if (!r || !paras_kb::compatible_with_any(*r, devs))
            return false;
    }
    return true;
}

template <bundle_state State>
inline bool has_kernel_bundle(const context& ctxt, const std::vector<kernel_id>& kernelIds) {
    return has_kernel_bundle<State>(ctxt, ctxt.get_devices(), kernelIds);
}

template <typename KernelName, bundle_state State>
inline bool has_kernel_bundle(const context& ctxt, const std::vector<device>& devs) {
    return has_kernel_bundle<State>(ctxt, devs, {get_kernel_id<KernelName>()});
}

template <typename KernelName, bundle_state State>
inline bool has_kernel_bundle(const context& ctxt) {
    return has_kernel_bundle<State>(ctxt, ctxt.get_devices(), {get_kernel_id<KernelName>()});
}

template <typename KernelName>
inline bool paras_kb::is_compatible_impl(const device& dev) {
    const auto* rec = paras_kb::find<KernelName>();
#if !PARASDEVICE
    if (!rec)
        return !dev.is_gpu();
#else
    if (!rec)
        return true;
#endif
    return paras_kb::compatible(*rec, dev);
}

namespace paras_kb {
template <bundle_state State>
inline void check_subset(const kernel_bundle<State>& b, const std::vector<device>& devs) {
    if (devs.empty())
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid), "empty device list");
    const auto bd = b.get_devices();
    for (const auto& d : devs)
        if (std::find(bd.begin(), bd.end(), d) == bd.end())
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "device is not associated with the kernel bundle");
}
}

inline kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input>& inputBundle, const std::vector<device>& devs,
        const property_list& = {}) {
    paras_kb::check_subset(inputBundle, devs);
    std::vector<const paras_kb::kernel_record*> ks;
    for (const auto& img : paras_kb::access::images(inputBundle))
        if (paras_kb::compatible_with_any(*paras_kb::access::record(img), devs))
            ks.push_back(paras_kb::access::record(img));
    auto out = paras_kb::access::make_bundle<bundle_state::object>(
        inputBundle.get_context(), paras_kb::unique_devices(devs), ks);
    paras_kb::access::copy_spec(inputBundle, out);
    return out;
}

inline kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input>& inputBundle, const property_list& propList = {}) {
    return compile(inputBundle, inputBundle.get_devices(), propList);
}

inline kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>>& objectBundles,
     const std::vector<device>& devs, const property_list& = {}) {
    if (devs.empty())
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "link: empty device list");
    std::vector<const paras_kb::kernel_record*> ks;
    for (const auto& b : objectBundles) {
        if (b.get_context() != objectBundles.front().get_context())
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "link: bundles have different contexts");
        paras_kb::check_subset(b, devs);
        for (const auto& img : paras_kb::access::images(b))
            if (paras_kb::compatible_with_any(*paras_kb::access::record(img), devs))
                ks.push_back(paras_kb::access::record(img));
    }
    const context ctx = objectBundles.empty() ? context{devs} : objectBundles.front().get_context();
    auto out = paras_kb::access::make_bundle<bundle_state::executable>(
        ctx, paras_kb::unique_devices(devs), ks);
    for (const auto& b : objectBundles)
        paras_kb::access::copy_spec(b, out);
    return out;
}

inline kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object>& objectBundle, const std::vector<device>& devs,
     const property_list& propList = {}) {
    return link(std::vector<kernel_bundle<bundle_state::object>>{objectBundle}, devs, propList);
}

inline kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>>& objectBundles,
     const property_list& propList = {}) {
    std::vector<device> devs;
    if (!objectBundles.empty()) {
        devs = objectBundles.front().get_devices();
        for (const auto& b : objectBundles) {
            const auto bd = b.get_devices();
            devs.erase(std::remove_if(devs.begin(), devs.end(),
                                      [&](const device& d) {
                                          return std::find(bd.begin(), bd.end(), d) == bd.end();
                                      }),
                       devs.end());
        }
    }
    return link(objectBundles, devs, propList);
}

inline kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object>& objectBundle, const property_list& propList = {}) {
    return link(std::vector<kernel_bundle<bundle_state::object>>{objectBundle}, propList);
}

inline kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input>& inputBundle, const std::vector<device>& devs,
      const property_list& = {}) {
    paras_kb::check_subset(inputBundle, devs);
    std::vector<const paras_kb::kernel_record*> ks;
    for (const auto& img : paras_kb::access::images(inputBundle))
        if (paras_kb::compatible_with_any(*paras_kb::access::record(img), devs))
            ks.push_back(paras_kb::access::record(img));
    auto out = paras_kb::access::make_bundle<bundle_state::executable>(
        inputBundle.get_context(), paras_kb::unique_devices(devs), ks);
    paras_kb::access::copy_spec(inputBundle, out);
    return out;
}

inline kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input>& inputBundle, const property_list& propList = {}) {
    return build(inputBundle, inputBundle.get_devices(), propList);
}

template <>
inline auto device::get_info<info::device::built_in_kernel_ids>() const {
    return std::vector<kernel_id>{};
}

} // namespace sycl

namespace std {
template <>
struct hash<sycl::kernel> {
    std::size_t operator()(const sycl::kernel& k) const noexcept {
        return std::hash<sycl::context>{}(k.get_context()) ^
               std::hash<const void*>{}(sycl::paras_kb::access::record(k));
    }
};
template <>
struct hash<sycl::kernel_id> {
    std::size_t operator()(const sycl::kernel_id& k) const noexcept {
        return std::hash<const void*>{}(sycl::paras_kb::access::record(k));
    }
};
template <sycl::bundle_state State>
struct hash<sycl::device_image<State>> {
    std::size_t operator()(const sycl::device_image<State>& d) const noexcept {
        return std::hash<const void*>{}(sycl::paras_kb::access::record(d));
    }
};
template <sycl::bundle_state State>
struct hash<sycl::kernel_bundle<State>> {
    std::size_t operator()(const sycl::kernel_bundle<State>& b) const noexcept {
        std::size_t h = std::hash<sycl::context>{}(b.get_context());
        for (const auto& img : b)
            h = h * 31 + std::hash<sycl::device_image<State>>{}(img);
        return h;
    }
};
} // namespace std

#endif
