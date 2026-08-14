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

#include "paras/Preprocessor.hpp"

namespace preprocessor {
std::string generate_output_filename() {
  std::string model = "/tmp/paras_pp_%%%%%%%%.cpp";
  llvm::SmallString<128> result;
  if (std::error_code EC = llvm::sys::fs::createUniqueFile(model, result)) {
    llvm::errs() << "paras: cannot create temp file: " << EC.message() << "\n";
    exit(EXIT_FAILURE);
  }
  return std::string(result.str());

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
