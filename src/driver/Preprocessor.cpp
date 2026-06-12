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

#include "paras/Preprocessor.hpp"

namespace preprocessor {
std::string generate_output_filename() {
  auto now = std::chrono::system_clock::now();
  auto now_sec = std::chrono::system_clock::to_time_t(now);
  std::string filename =
      "/tmp/files/paras_pp_" + std::to_string(now_sec) + ".cpp";
  return filename;
}

void process_file(const std::string &inputfile, const std::string &outputfile,
                  std::vector<std::string> bkend_target) {
  std::ifstream fin(inputfile);
  if (!fin) {
    perror("Error");
    exit(EXIT_FAILURE);
  }
  std::ofstream fout(outputfile);
  if (!fout) {
    std::cerr << "Cannot open output file: " << outputfile << "\n";
    return;
  }

  std::regex sycl_include(R"(^\s*#include\s*[<\"]sycl(\/sycl)?\.hpp[>\"])");

  std::vector<std::string> lines;
  std::string line;
  bool found_sycl = false;

  while (std::getline(fin, line)) {
    if (std::regex_search(line, sycl_include))
      found_sycl = true;

    lines.push_back(line);
  }

  bool inserted = false;

  if (!found_sycl) {
    if (!bkend_target[0].empty()) {
      if (bkend_target[0] == "cuda") {
        fout << "#include \"kem_gpu/gpu_threadpool.hpp\"\n";
        fout << "#define PARASDEVICE 1\n";
      } else if (bkend_target[0] == "hip") {
        fout << "#include \"kem_gpu/rocm_threadpool.hpp\"\n";
        fout << "#define PARASDEVICE 2\n";
      }
    } else {
      fout << "#include \"kem/threadpool.hpp\"\n";
    }
    inserted = true;
  }

  for (const auto &l : lines) {
    fout << l << "\n";

    if (!inserted && std::regex_search(l, sycl_include)) {
      if (!bkend_target[0].empty()) {
        if (bkend_target[0] == "cuda") {
          fout << "#include \"kem_gpu/gpu_threadpool.hpp\"\n";
          fout << "#define PARASDEVICE 1\n";
        } else if (bkend_target[0] == "hip") {
          fout << "#include \"kem_gpu/rocm_threadpool.hpp\"\n";
          fout << "#define PARASDEVICE 2\n";
        }
      } else {
        fout << "#include \"kem/threadpool.hpp\"\n";
      }

      inserted = true;
    }
  }

  fin.close();
  fout.close();
}

} // namespace preprocessor
