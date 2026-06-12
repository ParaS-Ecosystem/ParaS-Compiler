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

#ifndef __PARAS_CONTEXT_HPP__
#define __PARAS_CONTEXT_HPP__

#include "sycl/device.hpp"
#include "sycl/property_list.hpp"
#include <vector>

namespace sycl {

using async_handler = void (*)(void);

class context {
public:
  explicit context(const property_list & = {}) : devices_{device{}} {}

  explicit context(async_handler, const property_list & = {})
      : devices_{device{}} {}

  explicit context(const device &dev, const property_list & = {})
      : devices_{dev} {}

  explicit context(const device &dev, async_handler, const property_list & = {})
      : devices_{dev} {}

  explicit context(const std::vector<device> &deviceList,
                   const property_list & = {})
      : devices_{deviceList.empty() ? std::vector<device>{device{}}
                                    : deviceList} {}

  explicit context(const std::vector<device> &deviceList, async_handler,
                   const property_list & = {})
      : devices_{deviceList.empty() ? std::vector<device>{device{}}
                                    : deviceList} {}

  backend get_backend() const noexcept {
    return devices_.empty() ? backend::host : devices_.front().get_backend();
  }

  platform get_platform() const {
    return devices_.empty() ? platform("host")
                            : devices_.front().get_platform();
  }

  std::vector<device> get_devices() const { return devices_; }

  const device &get_device() const {
    static device fallback{};
    return devices_.empty() ? fallback : devices_.front();
  }

  template <typename Param> typename Param::return_type get_info() const {
    return typename Param::return_type{};
  }

  template <typename Param>
  typename Param::return_type get_backend_info() const {
    return typename Param::return_type{};
  }

private:
  std::vector<device> devices_;
};

} // namespace sycl

#endif
