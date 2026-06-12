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

#ifndef __PARAS_QUEUE_HPP__
#define __PARAS_QUEUE_HPP__

#include "context.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "property_list.hpp"

namespace sycl {

class queue {
public:
  queue() = default;

  explicit queue(const decltype(cpu_selector_v) &, const property_list & = {});
  explicit queue(const decltype(gpu_selector_v) &, const property_list & = {});
  explicit queue(const decltype(accelerator_selector_v) &,
                 const property_list & = {});
  explicit queue(const decltype(default_selector_v) &,
                 const property_list & = {});

  template <typename CGF> event submit(CGF cgf) {
    handler h;
    cgf(h);
    return event();
  }

  void wait() {}

  device get_device() const { return dev_; }
  context get_context() const { return ctx_; }
  bool is_in_order() const { return false; }

  backend get_backend() const {

    if (dev_.get_backend() == backend::cuda) {
      return backend::cuda;
    } else {
      return backend::host;
    }
  }

  template <typename KernelName, typename Func, int dim>
  void parallel_for(range<dim> r, Func f) {}

  template <typename KernelName, typename Func, int dim>
  void parallel_for(const nd_range<dim> &r, Func f) {}

  template <typename injectCustomFunc>
  event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.parasSYCL_enqueue_custom_operation(cgf);
    });
  }

  event memset(void *ptr, int value, size_t numBytes);

  event memcpy(void *dest, const void *src, size_t numBytes);

private:
  context ctx_{};
  device dev_{};
};

} // namespace sycl

#endif
