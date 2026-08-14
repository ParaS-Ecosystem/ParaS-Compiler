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

#ifndef __PARAS_QUEUE_HPP__
#define __PARAS_QUEUE_HPP__

#include "context.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "property_list.hpp"
#include "utilities/selector_logic.hpp"
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

class cuda_threadpool;

namespace sycl {

class queue {
public:
  queue() : dev_(::paras_extension::select_device_no_selector()) {}

  explicit queue(const decltype(cpu_selector_v) &, const property_list & = {});
  explicit queue(const decltype(gpu_selector_v) &, const property_list & = {});
  explicit queue(const decltype(accelerator_selector_v) &,
                 const property_list & = {});
  explicit queue(const decltype(default_selector_v) &,
                 const property_list & = {});

  explicit queue(const context &ctx, const device &dev,
                 const property_list &props = {});
  explicit queue(const device &dev, const property_list &props = {});

  template <typename CGF> event submit(CGF cgf) {
    if (dev_.is_gpu()) {
      handler h(get_or_create_gpu_pool());
      cgf(h);
    } else {
      handler h;
      cgf(h);
    }

    return event{};
  }

  void wait();

  void wait_and_throw();

  device get_device() const { return dev_; }
  context get_context() const { return ctx_; }
  bool is_in_order() const { return false; }

  backend get_backend() const {
    return dev_.is_gpu() ? backend::cuda : backend::host;
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
  template <typename T> event copy(const T *src, T *dest, size_t count);
  event memcpy(void *dest, const void *src, size_t numBytes);

private:
  mutable std::shared_ptr<cuda_threadpool> gpu_pool_{};

  cuda_threadpool &get_or_create_gpu_pool() const;

  context ctx_{};
  device dev_{};

  struct async_state {
    std::mutex mtx;
    std::vector<std::exception_ptr> exceptions;
  };

  inline void rec_async_exceptn(std::exception_ptr excp);
  inline void throw_async_exceptns();

  std::shared_ptr<async_state> async_state_ = std::make_shared<async_state>();
};

inline sycl::queue::queue(const sycl::context &ctx, const sycl::device &dev,
                          const sycl::property_list &props)
    : ctx_(ctx), dev_(dev) {
  (void)props;
}

inline void sycl::queue::rec_async_exceptn(std::exception_ptr ex) {
  std::lock_guard<std::mutex> lock(async_state_->mtx);
  async_state_->exceptions.push_back(std::move(ex));
}

inline void sycl::queue::throw_async_exceptns() {
  std::vector<std::exception_ptr> excps;

  {
    std::lock_guard<std::mutex> lock(async_state_->mtx);
    excps.swap(async_state_->exceptions);
  }

  for (const auto &ex : excps) {
    std::rethrow_exception(ex);
  }
}

} // namespace sycl

#endif
