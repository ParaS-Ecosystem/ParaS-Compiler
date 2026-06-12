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

#ifndef __PARAS_DETAIL_HPP__
#define __PARAS_DETAIL_HPP__

#include <string>

namespace sycl {
namespace detail {

#define SYCL_DETAIL_STRINGIFY_IMPL(x) #x
#define SYCL_DETAIL_STRINGIFY(x) SYCL_DETAIL_STRINGIFY_IMPL(x)

inline std::string version_string() {
  std::string s;

#if defined(__clang__)
  s += "clang ";
  s += __clang_version__;
#elif defined(__GNUC__)
  s += "gcc ";
  s += std::to_string(__GNUC__);
  s += ".";
  s += std::to_string(__GNUC_MINOR__);
  s += ".";
  s += std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
  s += "msvc ";
  s += std::to_string(_MSC_VER);
#else
  s += "unknown-compiler";
#endif

  s += "; c++=";
  s += SYCL_DETAIL_STRINGIFY(__cplusplus);

#ifdef __GLIBCXX__
  s += "; libstdc++=";
  s += SYCL_DETAIL_STRINGIFY(__GLIBCXX__);
#endif

#ifdef _LIBCPP_VERSION
  s += "; libc++=";
  s += SYCL_DETAIL_STRINGIFY(_LIBCPP_VERSION);
#endif

#ifdef PARAS_GPU_BACKEND
  s += "; PARAS_GPU_BACKEND=";
  s += SYCL_DETAIL_STRINGIFY(PARAS_GPU_BACKEND);
#endif

  return s;
}

#undef SYCL_DETAIL_STRINGIFY_IMPL
#undef SYCL_DETAIL_STRINGIFY

} // namespace detail
} // namespace sycl

#endif
