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

#include "paras/Executor.hpp"
#include "pathfinder.hpp"
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <csignal>
#include <sys/wait.h>

namespace executor {
std::string find_clangxx(const char *argv0) {

  llvm::SmallString<256> exe;
  if (!llvm::sys::fs::real_path(argv0, exe)) {
    llvm::SmallString<256> localClang(llvm::sys::path::parent_path(exe));
    llvm::sys::path::append(localClang, "clang++");
    if (llvm::sys::fs::exists(localClang))
      return std::string(localClang.str());
  }

  if (const char *inst = getenv("LLVM_INSTALL_DIR")) {
    llvm::SmallString<256> p(inst);
    llvm::sys::path::append(p, "bin", "clang++");
    if (llvm::sys::fs::exists(p))
      return std::string(p.str());
  }

  if (!access("clang++", X_OK))
    return "clang++";

  if (!access("/usr/bin/clang++", X_OK))
    return "/usr/bin/clang++";
  if (!access("/usr/local/bin/clang++", X_OK))
    return "/usr/local/bin/clang++";

  return "clang++";
}

namespace {

constexpr int kJobSignals[] = {SIGHUP, SIGINT, SIGQUIT};
bool ignoredAtStart[sizeof(kJobSignals) / sizeof(kJobSignals[0])] = {};
}

void remember_ignored_signals() {
  for (unsigned i = 0; i < sizeof(kJobSignals) / sizeof(kJobSignals[0]); ++i) {
    struct sigaction sa;
    if (sigaction(kJobSignals[i], nullptr, &sa) == 0)
      ignoredAtStart[i] = (sa.sa_handler == SIG_IGN);
  }
}

static void restore_ignored_signals() {
  for (unsigned i = 0; i < sizeof(kJobSignals) / sizeof(kJobSignals[0]); ++i)
    if (ignoredAtStart[i])
      std::signal(kJobSignals[i], SIG_IGN);
}

std::string base_compiler() {
  const char *env = std::getenv("PARAS_CXX");
  return (env && *env) ? std::string(env) : std::string(PARAS_BASE_COMPILER);
}

static void appendRuntimeLinkFlags(std::vector<std::string> &command) {
  std::string parasLibPath = PARAS_INSTALL_PREFIX + std::string("/lib");
  command.push_back("-L");
  command.push_back(parasLibPath);
  command.push_back("-Wl,-rpath," + parasLibPath);
  command.push_back("-lDeviceDiscoveryModule");
  const std::string gccLib = PARAS_GCC_LIBDIR;
  if (!gccLib.empty()) {
    command.push_back("-L" + gccLib);
    command.push_back("-Wl,-rpath," + gccLib);
  }
}

int executor(std::vector<std::string> command,
             std::vector<std::string> bkend_target) {
  std::string program_name = base_compiler();
  command.insert(command.begin(), program_name);

  if (!bkend_target[0].empty()) {
    std::string back_end = "";
    std::string tar_arch = "";
    back_end = bkend_target[0];
    tar_arch = bkend_target[1];

    std::vector<std::string> prefix;

    const bool rdc = std::find(command.begin(), command.end(), "-fgpu-rdc") !=
                     command.end();
    if (bkend_target[3] == "is_link_only" && rdc) {
      prefix.push_back("--offload-arch=" + tar_arch);
      if (back_end == "hip") {
        prefix.push_back("--hip-link");
      } else if (back_end == "cuda") {
        prefix.push_back("--offload-new-driver");
        prefix.push_back("--offload-link");
      }
    }
    if (bkend_target[3] != "is_link_only") {
      if (rdc && back_end == "cuda")
        prefix.push_back("--offload-new-driver");
      prefix.push_back("-x");
      prefix.push_back(back_end);
      prefix.push_back("--offload-arch=" + tar_arch);

      if (back_end == "cuda") {
        bool user_set = false;
        for (const std::string &a : command)
          if (a.find("maxrregcount") != std::string::npos)
            user_set = true;
        long regs = 64;
        if (const char *e = std::getenv("PARAS_CUDA_MAXRREGCOUNT"))
          regs = std::strtol(e, nullptr, 10);
        if (!user_set && regs > 0) {
          prefix.push_back("-Xcuda-ptxas");
          prefix.push_back("--maxrregcount=" + std::to_string(regs));
        }
      }

      if (!std::getenv("PARAS_NO_IMPLICIT_HD_TEMPLATES"))
        prefix.push_back("-foffload-implicit-host-device-templates");

      prefix.push_back("-include");
      prefix.push_back(std::string(PARAS_INSTALL_PREFIX) +
                       "/include/kem_gpu/device_prelude.hpp");
    }
    if (!bkend_target[2].empty()) {
      prefix.push_back("-resource-dir");
      prefix.push_back(bkend_target[2]);
    }
    command.insert(command.begin() + 1, prefix.begin(), prefix.end());
    appendRuntimeLinkFlags(command);
  } else {
    if (!bkend_target[2].empty()) {
      command.insert(command.begin() + 1, "-resource-dir");
      command.insert(command.begin() + 2, bkend_target[2]);
    }
    appendRuntimeLinkFlags(command);
  }

  std::vector<char *> args;
  args.reserve(command.size() + 1);

  for (auto &itr : command)
    args.push_back(const_cast<char *>(itr.c_str()));

  args.push_back(nullptr);

  restore_ignored_signals();
  pid_t pid = fork();
  if (pid < 0) {
    perror("parascc: fork failed");
    return 1;
  }
  if (pid == 0) {
    execvp(program_name.c_str(), args.data());
    perror("parascc: execvp failed");
    _exit(127);
  }

  int status = 0;
  pid_t r;
  do {
    r = waitpid(pid, &status, 0);
  } while (r < 0 && errno == EINTR);
  if (r < 0) {
    perror("parascc: waitpid failed");
    return 1;
  }
  if (WIFSIGNALED(status))
    return 128 + WTERMSIG(status);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
}
