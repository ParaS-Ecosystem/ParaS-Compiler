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

#ifndef __PARAS_DEVICE_SELECTOR_HPP__
#define __PARAS_DEVICE_SELECTOR_HPP__

#include "device.hpp"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef PARASDEVICE
#define PARASDEVICE 0
#endif

namespace sycl {

template <typename F> class device_selector {
public:
  constexpr device_selector(F func) : select_(func) {}
  int operator()(const device &dev) const { return select_(dev); }

private:
  F select_;
};

inline constexpr auto cpu_selector_v = device_selector{
    [](const device &dev) -> int { return dev.is_cpu() ? 1 : -1; }};

inline constexpr auto gpu_selector_v = device_selector{
    [](const device &dev) -> int { return dev.is_gpu() ? 1 : -1; }};

inline constexpr auto accelerator_selector_v = device_selector{
    [](const device &dev) -> int { return dev.is_accelerator() ? 1 : -1; }};

inline constexpr auto default_selector_v =
    device_selector{[](const device &dev) -> int {
      return dev.is_cpu()           ? 3
             : dev.is_gpu()         ? 2
             : dev.is_accelerator() ? 1
                                    : -1;
    }};

} // namespace sycl

#endif
