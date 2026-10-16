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
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "context.hpp"
#include "info.hpp"
#include "kernel_handler.hpp"
#include "kernel_registry.hpp"
#include "property_list.hpp"

namespace sycl {

enum class bundle_state { input, object, executable };

template <bundle_state State> class kernel_bundle;
template <bundle_state State> class device_image;
class kernel;

class kernel_id {
public:
  kernel_id() = delete;

  const char *get_name() const noexcept { return ki_->name.c_str(); }

  friend bool operator==(const kernel_id &a, const kernel_id &b) noexcept {
    return a.ki_ == b.ki_;
  }
  friend bool operator!=(const kernel_id &a, const kernel_id &b) noexcept {
    return a.ki_ != b.ki_;
  }

  const detail::kernel_info *paras_info() const noexcept { return ki_; }
  static kernel_id paras_make(const detail::kernel_info *ki) {
    return kernel_id(ki);
  }

private:
  explicit kernel_id(const detail::kernel_info *ki) : ki_(ki) {}
  const detail::kernel_info *ki_;
};

namespace detail {

inline bool kernel_compatible(const kernel_info *ki, const device &d) {
  for (aspect a : ki->props.aspects)
    if (!d.has(a))
      return false;
  if (ki->props.sg != 0) {
    const auto sizes = d.get_info<info::device::sub_group_sizes>();
    if (std::find(sizes.begin(), sizes.end(), ki->props.sg) == sizes.end())
      return false;
  }
  if (ki->props.wg_dims > 0) {
    std::size_t n = 1;
    for (int i = 0; i < ki->props.wg_dims; ++i)
      n *= ki->props.wg[i];
    if (n > d.get_info<info::device::max_work_group_size>())
      return false;
  }
  return true;
}

inline bool state_supported(bundle_state s, const device &d) {
  if (s == bundle_state::input)
    return d.has(aspect::online_compiler);
  if (s == bundle_state::object)
    return d.has(aspect::online_linker);
  return true;
}

inline const char *state_name(bundle_state s) {
  return s == bundle_state::input    ? "input"
         : s == bundle_state::object ? "object"
                                     : "executable";
}

struct image_impl {
  const kernel_info *ki;
  std::vector<device> devs;
};

struct bundle_impl {
  context ctx;
  std::vector<device> devs;
  bundle_state state = bundle_state::executable;
  std::vector<std::shared_ptr<image_impl>> images;

  std::map<const void *, std::vector<unsigned char>> spec;

  const image_impl *find(const kernel_info *ki) const {
    for (const auto &im : images)
      if (im->ki == ki)
        return im.get();
    return nullptr;
  }
};

inline bool contains_device(const std::vector<device> &v, const device &d) {
  return std::find(v.begin(), v.end(), d) != v.end();
}

inline void check_bundle_devices(const context &ctx,
                                 const std::vector<device> &devs,
                                 bundle_state state, bool for_bundle) {
  if (devs.empty())
    throw exception(make_error_code(errc::invalid),
                    "kernel bundle: the device list is empty");
  const auto ctx_devs = ctx.get_devices();
  for (const auto &d : devs)
    if (!contains_device(ctx_devs, d))
      throw exception(make_error_code(errc::invalid),
                      "kernel bundle: a device is not in the context");
  if (for_bundle)
    for (const auto &d : devs)
      if (!state_supported(state, d))
        throw exception(make_error_code(errc::invalid),
                        std::string("kernel bundle: no device image in ") +
                            state_name(state) +
                            " state (the device has no online " +
                            (state == bundle_state::input ? "compiler"
                                                          : "linker") +
                            ")");
}

inline std::shared_ptr<image_impl>
make_image(const kernel_info *ki, const std::vector<device> &devs) {
  auto im = std::make_shared<image_impl>();
  im->ki = ki;
  for (const auto &d : devs)
    if (kernel_compatible(ki, d))
      im->devs.push_back(d);
  return im;
}

inline std::shared_ptr<bundle_impl>
make_bundle(const context &ctx, const std::vector<device> &devs,
            bundle_state state, const std::vector<const kernel_info *> &kis) {
  auto b = std::make_shared<bundle_impl>();
  b->ctx = ctx;

  for (const device &d : devs)
    if (std::find(b->devs.begin(), b->devs.end(), d) == b->devs.end())
      b->devs.push_back(d);
  b->state = state;
  for (const kernel_info *ki : kis) {
    if (b->find(ki))
      continue;
    auto im = make_image(ki, devs);
    if (!im->devs.empty())
      b->images.push_back(std::move(im));
  }
  return b;
}

inline std::vector<const kernel_info *>
compatible_kernels(const std::vector<device> &devs) {
  std::vector<const kernel_info *> out;
  for (const kernel_info *ki : all_kernels())
    for (const auto &d : devs)
      if (kernel_compatible(ki, d)) {
        out.push_back(ki);
        break;
      }
  return out;
}

inline std::vector<const kernel_info *>
infos_of(const std::vector<kernel_id> &ids) {
  std::vector<const kernel_info *> out;
  for (const auto &id : ids)
    out.push_back(id.paras_info());
  return out;
}

inline void kernel_launch_check(const kernel_info *ki, const bundle_impl *kb,
                                const device &dev) {
  if (ki == nullptr)
    return;
  if (kb != nullptr) {
    const image_impl *im = kb->find(ki);
    if (im == nullptr)
      throw exception(make_error_code(errc::kernel_not_supported),
                      "kernel " + ki->name +
                          " is not in the kernel bundle used by the "
                          "command group");
    if (!contains_device(im->devs, dev))
      throw exception(make_error_code(errc::kernel_not_supported),
                      "kernel " + ki->name +
                          " in the kernel bundle does not support the "
                          "queue's device");
  }
  if (!kernel_compatible(ki, dev))
    throw exception(make_error_code(errc::kernel_not_supported),
                    "kernel " + ki->name +
                        " uses an optional feature the device does not "
                        "support");
}

template <int D>
inline void check_nd_launch(const kernel_info *ki, const range<D> &local,
                            const device &dev) {
  if (local.size() > dev.get_info<info::device::max_work_group_size>())
    throw exception(make_error_code(errc::nd_range),
                    "nd_range: the work-group size exceeds the device's "
                    "max_work_group_size");
  if (ki != nullptr && ki->props.wg_dims > 0) {
    const auto &p = ki->props;
    const int declared = int(p.wg_has[0]) + int(p.wg_has[1]) + int(p.wg_has[2]);
    bool same = true;
    if (p.wg_has[D - 1]) {
      for (int i = 0; same && i < D; ++i)
        same = p.wg_by_dim[D - 1][i] == local[i];
    } else if (declared == 1) {
      same = false;
    }
    if (!same)
      throw exception(make_error_code(errc::nd_range),
                      "nd_range: the work-group size differs from the "
                      "kernel's reqd_work_group_size");
  }
}

template <typename KernelName, typename Func, int D>
inline void note_and_check_nd_kernel(const device &dev, const range<D> &local) {
#if !PARAS_DEVICE_PASS
  const kernel_info *ki = note_kernel<KernelName, Func>();
  if (ki)
    kernel_launch_check(ki, nullptr, dev);
  check_nd_launch(ki, local, dev);
#else
  (void)dev;
  (void)local;
#endif
}

template <typename KernelName, typename Func>
inline void note_and_check_kernel(const device &dev) {
#if !PARAS_DEVICE_PASS
  if (const kernel_info *ki = note_kernel<KernelName, Func>())
    kernel_launch_check(ki, nullptr, dev);
#else
  (void)dev;
#endif
}

template <auto &SpecName>
const std::vector<unsigned char> *bundle_spec_value(const bundle_impl &b) {
  auto it = b.spec.find(&SpecName);
  return it == b.spec.end() ? nullptr : &it->second;
}

}

class kernel {
public:
  kernel() = delete;

  context get_context() const { return bundle_->ctx; }

  backend get_backend() const noexcept {
    return bundle_->devs.empty() ? backend::host
                                 : bundle_->devs.front().get_backend();
  }

  kernel_bundle<bundle_state::executable> get_kernel_bundle() const;

  template <typename Param> typename Param::return_type get_info() const {

    if constexpr (std::is_same_v<Param, info::kernel::num_args>)
      throw exception(make_error_code(errc::invalid),
                      "info::kernel::num_args needs an interop or built-in "
                      "kernel");
    return typename Param::return_type{};
  }

  template <typename Param>
  typename Param::return_type get_info(const device &dev) const {
    namespace kds = info::kernel_device_specific;
    const auto &p = ki_->props;
    if constexpr (std::is_same_v<Param, kds::global_work_size>) {

      throw exception(make_error_code(errc::invalid),
                      "global_work_size needs a custom device or a built-in "
                      "kernel");
    } else if constexpr (std::is_same_v<Param, kds::work_group_size>) {
      return dev.get_info<info::device::max_work_group_size>();
    } else if constexpr (std::is_same_v<
                             Param, kds::preferred_work_group_size_multiple>) {
      return 1;
    } else if constexpr (std::is_same_v<Param, kds::compile_work_group_size>) {
      range<3> r(0, 0, 0);
      if (p.wg_dims > 0) {
        r = range<3>(1, 1, 1);

        for (int i = 0; i < p.wg_dims; ++i)
          r[3 - p.wg_dims + i] = p.wg[i];
      }
      return r;
    } else if constexpr (std::is_same_v<Param, kds::max_num_sub_groups>) {
      const std::uint32_t n = dev.get_info<info::device::max_num_sub_groups>();
      return n ? n : 1;
    } else if constexpr (std::is_same_v<Param, kds::max_sub_group_size>) {
      if (p.sg != 0)
        return static_cast<std::uint32_t>(p.sg);
      std::uint32_t m = 0;
      for (auto s : dev.get_info<info::device::sub_group_sizes>())
        if (static_cast<std::uint32_t>(s) > m)
          m = static_cast<std::uint32_t>(s);
      return m ? m : 1;
    } else if constexpr (std::is_same_v<Param, kds::compile_sub_group_size>) {
      return static_cast<std::uint32_t>(p.sg);
    } else {
      return typename Param::return_type{};
    }
  }

  friend bool operator==(const kernel &a, const kernel &b) {
    return a.ki_ == b.ki_ && a.bundle_ == b.bundle_;
  }
  friend bool operator!=(const kernel &a, const kernel &b) { return !(a == b); }

  const detail::kernel_info *paras_info() const noexcept { return ki_; }
  std::size_t paras_hash() const noexcept {
    return std::hash<const void *>{}(ki_) ^
           (std::hash<const void *>{}(bundle_.get()) << 1);
  }

private:
  template <bundle_state> friend class kernel_bundle;
  kernel(const detail::kernel_info *ki,
         std::shared_ptr<const detail::bundle_impl> b)
      : ki_(ki), bundle_(std::move(b)) {}

  const detail::kernel_info *ki_;
  std::shared_ptr<const detail::bundle_impl> bundle_;
};

template <bundle_state State> class device_image {
public:
  device_image() = delete;

  bool has_kernel(const kernel_id &id) const noexcept {
    return id.paras_info() == im_->ki;
  }
  bool has_kernel(const kernel_id &id, const device &dev) const noexcept {
    return has_kernel(id) && detail::contains_device(im_->devs, dev);
  }

  friend bool operator==(const device_image &a, const device_image &b) {
    return a.im_ == b.im_;
  }
  friend bool operator!=(const device_image &a, const device_image &b) {
    return !(a == b);
  }

  const std::shared_ptr<detail::image_impl> &paras_impl() const { return im_; }
  explicit device_image(std::shared_ptr<detail::image_impl> im)
      : im_(std::move(im)) {}

private:
  std::shared_ptr<detail::image_impl> im_;
};

template <bundle_state State> class kernel_bundle {
public:
  using device_image_iterator = const device_image<State> *;

  kernel_bundle() = delete;

  bool empty() const noexcept { return b_->images.empty(); }

  backend get_backend() const noexcept {
    return b_->devs.empty() ? backend::host : b_->devs.front().get_backend();
  }

  context get_context() const noexcept { return b_->ctx; }

  std::vector<device> get_devices() const noexcept { return b_->devs; }

  bool has_kernel(const kernel_id &id) const noexcept {
    return b_->find(id.paras_info()) != nullptr;
  }
  bool has_kernel(const kernel_id &id, const device &dev) const noexcept {
    const detail::image_impl *im = b_->find(id.paras_info());
    return im != nullptr && detail::contains_device(im->devs, dev);
  }
  template <typename KernelName> bool has_kernel() const noexcept {
    const detail::kernel_info *ki =
        detail::find_kernel(&detail::type_key<KernelName>::value);
    return ki != nullptr && b_->find(ki) != nullptr;
  }
  template <typename KernelName>
  bool has_kernel(const device &dev) const noexcept {
    const detail::kernel_info *ki =
        detail::find_kernel(&detail::type_key<KernelName>::value);
    if (ki == nullptr)
      return false;
    const detail::image_impl *im = b_->find(ki);
    return im != nullptr && detail::contains_device(im->devs, dev);
  }

  std::vector<kernel_id> get_kernel_ids() const {
    std::vector<kernel_id> out;
    for (const auto &im : b_->images)
      out.push_back(kernel_id::paras_make(im->ki));
    return out;
  }

  template <bundle_state S = State,
            typename = std::enable_if_t<S == bundle_state::executable>>
  kernel get_kernel(const kernel_id &id) const {
    if (!has_kernel(id))
      throw exception(make_error_code(errc::invalid),
                      "kernel_bundle::get_kernel: the kernel is not in the "
                      "bundle");
    return kernel(id.paras_info(), b_);
  }
  template <typename KernelName, bundle_state S = State,
            typename = std::enable_if_t<S == bundle_state::executable>>
  kernel get_kernel() const {
    const detail::kernel_info *ki =
        detail::find_kernel(&detail::type_key<KernelName>::value);
    if (ki == nullptr || b_->find(ki) == nullptr)
      throw exception(make_error_code(errc::invalid),
                      "kernel_bundle::get_kernel: the kernel is not in the "
                      "bundle");
    return kernel(ki, b_);
  }

  bool contains_specialization_constants() const noexcept {
    return !empty();
  }
  bool native_specialization_constant() const noexcept { return false; }

  template <auto &SpecName>
  bool has_specialization_constant() const noexcept {

    for (const auto &im : b_->images) {
      const detail::kernel_info *ki = im->ki;
      if (std::find(ki->props.spec_uses.begin(), ki->props.spec_uses.end(),
                    static_cast<const void *>(&SpecName)) !=
          ki->props.spec_uses.end())
        return true;
    }
    return false;
  }

  template <auto &SpecName,
            bundle_state S = State,
            typename = std::enable_if_t<S == bundle_state::input>>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type value) {
    using T = typename std::remove_reference_t<decltype(SpecName)>::value_type;
    std::vector<unsigned char> bytes(sizeof(T));
    std::memcpy(bytes.data(), static_cast<const void *>(&value), sizeof(T));
    b_->spec[&SpecName] = std::move(bytes);
  }

  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const {
    using T = typename std::remove_reference_t<decltype(SpecName)>::value_type;
    T value = SpecName.paras_default();
    if (const auto *raw = detail::bundle_spec_value<SpecName>(*b_))
      std::memcpy(static_cast<void *>(&value), raw->data(), sizeof(T));
    return value;
  }

  device_image_iterator begin() const noexcept { return view().data(); }
  device_image_iterator end() const noexcept {
    return view().data() + view().size();
  }

  friend bool operator==(const kernel_bundle &a, const kernel_bundle &b) {
    return a.b_ == b.b_;
  }
  friend bool operator!=(const kernel_bundle &a, const kernel_bundle &b) {
    return !(a == b);
  }

  explicit kernel_bundle(std::shared_ptr<detail::bundle_impl> b)
      : b_(std::move(b)) {}
  const std::shared_ptr<detail::bundle_impl> &paras_impl() const noexcept {
    return b_;
  }

private:

  const std::vector<device_image<State>> &view() const {
    if (!images_)
      images_ = std::make_shared<std::vector<device_image<State>>>();
    if (images_->size() != b_->images.size()) {
      images_->clear();
      for (const auto &im : b_->images)
        images_->emplace_back(im);
    }
    return *images_;
  }

  std::shared_ptr<detail::bundle_impl> b_;
  mutable std::shared_ptr<std::vector<device_image<State>>> images_;
};

inline kernel_bundle<bundle_state::executable>
kernel::get_kernel_bundle() const {
  return kernel_bundle<bundle_state::executable>(
      std::const_pointer_cast<detail::bundle_impl>(bundle_));
}

template <typename KernelName> kernel_id get_kernel_id() {
  const detail::kernel_info *ki =
      detail::find_kernel(&detail::type_key<KernelName>::value);
  if (ki == nullptr)
    throw exception(make_error_code(errc::runtime),
                    "get_kernel_id: no kernel named " +
                        detail::type_name<KernelName>());
  return kernel_id::paras_make(ki);
}

inline std::vector<kernel_id> get_kernel_ids() {
  std::vector<kernel_id> out;
  for (const detail::kernel_info *ki : detail::all_kernels())
    out.push_back(kernel_id::paras_make(ki));
  return out;
}

inline bool is_compatible(const std::vector<kernel_id> &ids,
                          const device &dev) {
  for (const auto &id : ids)
    if (!detail::kernel_compatible(id.paras_info(), dev))
      return false;
  return true;
}

template <typename KernelName> bool is_compatible(const device &dev) {
  const detail::kernel_info *ki =
      detail::find_kernel(&detail::type_key<KernelName>::value);
  return ki == nullptr || detail::kernel_compatible(ki, dev);
}

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctx,
                                       const std::vector<device> &devs) {
  detail::check_bundle_devices(ctx, devs, State, true);
  return kernel_bundle<State>(detail::make_bundle(
      ctx, devs, State, detail::compatible_kernels(devs)));
}

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctx) {
  return get_kernel_bundle<State>(ctx, ctx.get_devices());
}

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctx,
                                       const std::vector<device> &devs,
                                       const std::vector<kernel_id> &ids) {
  detail::check_bundle_devices(ctx, devs, State, true);
  for (const auto &id : ids) {
    bool any = false;
    for (const auto &d : devs)
      any = any || detail::kernel_compatible(id.paras_info(), d);
    if (!any)
      throw exception(make_error_code(errc::invalid),
                      std::string("get_kernel_bundle: kernel ") +
                          id.get_name() + " is not compatible with any of "
                          "the devices");
  }
  return kernel_bundle<State>(
      detail::make_bundle(ctx, devs, State, detail::infos_of(ids)));
}

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctx,
                                       const std::vector<kernel_id> &ids) {
  return get_kernel_bundle<State>(ctx, ctx.get_devices(), ids);
}

template <bundle_state State, typename Selector,
          typename = std::enable_if_t<
              std::is_invocable_r_v<bool, Selector, const device_image<State> &>>>
kernel_bundle<State> get_kernel_bundle(const context &ctx,
                                       const std::vector<device> &devs,
                                       Selector selector) {
  auto b = get_kernel_bundle<State>(ctx, devs).paras_impl();
  std::vector<std::shared_ptr<detail::image_impl>> kept;
  for (const auto &im : b->images)
    if (selector(device_image<State>(im)))
      kept.push_back(im);
  b->images = std::move(kept);
  return kernel_bundle<State>(b);
}

template <bundle_state State, typename Selector,
          typename = std::enable_if_t<
              std::is_invocable_r_v<bool, Selector, const device_image<State> &>>>
kernel_bundle<State> get_kernel_bundle(const context &ctx, Selector selector) {
  return get_kernel_bundle<State>(ctx, ctx.get_devices(), selector);
}

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctx,
                                       const std::vector<device> &devs) {
  detail::check_bundle_devices(ctx, devs, State, true);
  return get_kernel_bundle<State>(ctx, devs, {get_kernel_id<KernelName>()});
}

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctx) {
  return get_kernel_bundle<KernelName, State>(ctx, ctx.get_devices());
}

template <bundle_state State>
bool has_kernel_bundle(const context &ctx, const std::vector<device> &devs) {
  detail::check_bundle_devices(ctx, devs, State, false);
  for (const auto &d : devs)
    if (!detail::state_supported(State, d))
      return false;
  return !detail::compatible_kernels(devs).empty();
}

template <bundle_state State> bool has_kernel_bundle(const context &ctx) {
  return has_kernel_bundle<State>(ctx, ctx.get_devices());
}

template <bundle_state State>
bool has_kernel_bundle(const context &ctx, const std::vector<device> &devs,
                       const std::vector<kernel_id> &ids) {
  detail::check_bundle_devices(ctx, devs, State, false);
  for (const auto &id : ids) {
    bool ok = false;
    for (const auto &d : devs)
      ok = ok || (detail::state_supported(State, d) &&
                  detail::kernel_compatible(id.paras_info(), d));
    if (!ok)
      return false;
  }
  return true;
}

template <bundle_state State>
bool has_kernel_bundle(const context &ctx, const std::vector<kernel_id> &ids) {
  return has_kernel_bundle<State>(ctx, ctx.get_devices(), ids);
}

template <typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctx, const std::vector<device> &devs) {
  return has_kernel_bundle<State>(ctx, devs, {get_kernel_id<KernelName>()});
}

template <typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctx) {
  return has_kernel_bundle<KernelName, State>(ctx, ctx.get_devices());
}

template <bundle_state State>
kernel_bundle<State> join(const std::vector<kernel_bundle<State>> &bundles) {
  auto out = std::make_shared<detail::bundle_impl>();
  out->state = State;
  if (bundles.empty())
    return kernel_bundle<State>(out);
  const context ctx = bundles.front().get_context();
  out->ctx = ctx;
  for (const auto &kb : bundles) {
    const auto &b = *kb.paras_impl();
    if (b.ctx != ctx)
      throw exception(make_error_code(errc::invalid),
                      "join: the kernel bundles have different contexts");
    for (const auto &d : b.devs)
      if (!detail::contains_device(out->devs, d))
        out->devs.push_back(d);
    for (const auto &im : b.images)
      if (out->find(im->ki) == nullptr)
        out->images.push_back(im);
    for (const auto &sv : b.spec)
      out->spec.insert(sv);
  }
  return kernel_bundle<State>(out);
}

namespace detail {

template <bundle_state To, bundle_state From>
kernel_bundle<To> transition(const kernel_bundle<From> &kb,
                             const std::vector<device> &devs,
                             const char *what) {
  const auto &b = *kb.paras_impl();
  if (devs.empty())
    throw exception(make_error_code(errc::invalid),
                    std::string(what) + ": the device list is empty");
  for (const auto &d : devs)
    if (!contains_device(b.devs, d))
      throw exception(make_error_code(errc::invalid),
                      std::string(what) +
                          ": a device is not associated with the bundle");
  for (const auto &d : devs) {
    if (To == bundle_state::object && !d.has(aspect::online_compiler))
      throw exception(make_error_code(errc::feature_not_supported),
                      std::string(what) + ": the device has no online compiler");
    if (From == bundle_state::object && !d.has(aspect::online_linker))
      throw exception(make_error_code(errc::feature_not_supported),
                      std::string(what) + ": the device has no online linker");
  }
  std::vector<const kernel_info *> kis;
  for (const auto &im : b.images)
    kis.push_back(im->ki);
  auto out = make_bundle(b.ctx, devs, To, kis);
  out->spec = b.spec;
  return kernel_bundle<To>(out);
}
}

inline kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &in,
        const std::vector<device> &devs, const property_list & = {}) {
  return detail::transition<bundle_state::object>(in, devs, "compile");
}
inline kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &in,
        const property_list &props = {}) {
  return compile(in, in.get_devices(), props);
}

inline kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &objs,
     const std::vector<device> &devs, const property_list & = {}) {
  if (objs.empty())
    throw exception(make_error_code(errc::invalid), "link: no kernel bundles");
  auto joined = join(objs);
  return detail::transition<bundle_state::executable>(joined, devs, "link");
}
inline kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &objs,
     const property_list &props = {}) {
  std::vector<device> devs;
  for (const auto &o : objs)
    for (const auto &d : o.get_devices())
      if (!detail::contains_device(devs, d))
        devs.push_back(d);
  return link(objs, devs, props);
}
inline kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &obj,
     const std::vector<device> &devs, const property_list &props = {}) {
  return link(std::vector<kernel_bundle<bundle_state::object>>{obj}, devs,
              props);
}
inline kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &obj,
     const property_list &props = {}) {
  return link(obj, obj.get_devices(), props);
}

inline kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &in,
      const std::vector<device> &devs, const property_list & = {}) {
  for (const auto &d : devs)
    if (!d.has(aspect::online_compiler) || !d.has(aspect::online_linker))
      throw exception(make_error_code(errc::feature_not_supported),
                      "build: the device has no online compiler / linker");
  return detail::transition<bundle_state::executable>(in, devs, "build");
}
inline kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &in,
      const property_list &props = {}) {
  return build(in, in.get_devices(), props);
}

template <>
inline auto device::get_info<info::device::built_in_kernel_ids>() const {
  return std::vector<kernel_id>{};
}

}

namespace std {
template <> struct hash<sycl::kernel> {
  std::size_t operator()(const sycl::kernel &k) const noexcept {
    return k.paras_hash();
  }
};
template <> struct hash<sycl::kernel_id> {
  std::size_t operator()(const sycl::kernel_id &k) const noexcept {
    return std::hash<const void *>{}(k.paras_info());
  }
};
template <sycl::bundle_state State> struct hash<sycl::device_image<State>> {
  std::size_t operator()(const sycl::device_image<State> &d) const noexcept {
    return std::hash<const void *>{}(d.paras_impl().get());
  }
};
template <sycl::bundle_state State> struct hash<sycl::kernel_bundle<State>> {
  std::size_t operator()(const sycl::kernel_bundle<State> &b) const noexcept {
    return std::hash<const void *>{}(b.paras_impl().get());
  }
};
}

#endif
