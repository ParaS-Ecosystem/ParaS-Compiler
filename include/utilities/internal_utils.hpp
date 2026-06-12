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

#ifndef __PARAS_INTERNAL_UTILS_HPP__
#define __PARAS_INTERNAL_UTILS_HPP__

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// #include "../kem_gpu/gpu_utilities.hpp"

namespace paras_extension {

inline void trim_inplace(std::string &s) {
  const size_t a = s.find_first_not_of(" \t\n\r");
  if (a == std::string::npos) {
    s.clear();
    return;
  }
  const size_t b = s.find_last_not_of(" \t\n\r");
  s = s.substr(a, b - a + 1);
}

inline bool run_cmd_lines(const char *cmd, std::vector<std::string> &lines) {
  FILE *p = popen(cmd, "r");
  if (!p)
    return false;
  char buf[512];
  while (fgets(buf, sizeof(buf), p)) {
    std::string s(buf);
    trim_inplace(s);
    if (!s.empty())
      lines.push_back(s);
  }
  pclose(p);
  return true;
}

inline bool run_cmd_lines_cached(const char *cmd,
                                 std::vector<std::string> &lines) {
  static std::unordered_map<std::string, std::vector<std::string>> cache;
  static std::mutex m;

  {
    std::lock_guard<std::mutex> lock(m);
    auto it = cache.find(cmd);
    if (it != cache.end()) {
      lines = it->second;
      return true;
    }
  }

  std::vector<std::string> tmp;
  bool ok = run_cmd_lines(cmd, tmp);

  {
    std::lock_guard<std::mutex> lock(m);
    cache.emplace(cmd, tmp);
  }

  lines = std::move(tmp);
  return ok;
}

inline std::string run_cmd_first_line_cached(const char *cmd) {
  std::vector<std::string> lines;
  run_cmd_lines_cached(cmd, lines);
  return lines.empty() ? std::string{} : lines.front();
}

[[noreturn]] inline void paras_selector_error(const char *msg) {
  std::cerr << "ParaS selector error: " << msg << "\n";
  std::exit(1);
}

inline unsigned long parse_ul_or0(const std::string &s) {
  char *end = nullptr;
  unsigned long v = std::strtoul(s.c_str(), &end, 10);
  return (end == s.c_str()) ? 0UL : v;
}

class local_memory_allocator {
public:
  using address = void *;

  static address alloc_bytes(size_t alignment, size_t bytes) {
#if PARAS_GPU_BACKEND

    void *ptr = nullptr;

    cudaError_t err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMallocManaged failed");
    }

    return ptr;

#else

    void *ptr = nullptr;

#if (__cplusplus >= 201703L)
    ptr = std::aligned_alloc(alignment,
                             ((bytes + alignment - 1) / alignment) * alignment);
#else
    posix_memalign(&ptr, alignment, bytes);
#endif

    if (!ptr) {
      throw std::bad_alloc();
    }

    return ptr;

#endif
  }

  template <class T> static address alloc(size_t elements) {
    return alloc_bytes(alignof(T), sizeof(T) * elements);
  }

  static void free(address ptr) noexcept { std::free(ptr); }
};

class local_memory {
public:
  using address = local_memory_allocator::address;
};

} // namespace paras_extension

#endif
