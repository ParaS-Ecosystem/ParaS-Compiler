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
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/CommandLine.h>
#include <sys/wait.h>
#include <algorithm>
#include <cstdio>

namespace executor {
std::string find_clangxx(const char* argv0) {
    try {
        fs::path exe = fs::canonical(argv0);
        fs::path dir = exe.parent_path();
        fs::path localClang = dir / "clang++";
        if (fs::exists(localClang))
            return localClang.string();
    } catch (...) {
    }

    if (const char* inst = getenv("LLVM_INSTALL_DIR")) {
        fs::path p = fs::path(inst) / "bin/clang++";
        if (fs::exists(p))
            return p.string();
    }

    if (!access("clang++", X_OK))
        return "clang++";

    if (!access("/usr/bin/clang++", X_OK))
        return "/usr/bin/clang++";
    if (!access("/usr/local/bin/clang++", X_OK))
        return "/usr/local/bin/clang++";

    return "clang++";
}

int executor(std::vector<std::string> command, std::vector<std::string> bkend_target) {

    std::string program_name = PARAS_BASE_COMPILER;
    command.insert(command.begin(), program_name);

    if (!bkend_target[0].empty()) {
        std::string back_end = "";
        std::string tar_arch = "";
        back_end = bkend_target[0];
        tar_arch = bkend_target[1];
        std::vector<std::string> prefix;
        if (bkend_target[3] != "is_link_only") {
            prefix.push_back("-x");
            prefix.push_back(back_end);
            prefix.push_back("--offload-arch=" + tar_arch);
#if LLVM_VERSION_MAJOR < 21

            if (back_end == "hip")
                prefix.push_back("-Wno-linker-warnings");
#endif
        }
        if (!bkend_target[2].empty()) {
            prefix.push_back("-resource-dir");
            prefix.push_back(bkend_target[2]);
        }
        command.insert(command.begin() + 1, prefix.begin(), prefix.end());
    } else {
        if (!bkend_target[2].empty()) {
            command.insert(command.begin() + 1, "-resource-dir");
            command.insert(command.begin() + 2, bkend_target[2]);
        }
    }

    const bool links = std::none_of(command.begin(), command.end(), [](const std::string& a) {
        return a == "-c" || a == "-S" || a == "-E" || a == "-fsyntax-only";
    });
    if (links) {
        command.push_back("-L");
        command.push_back(PARAS_INSTALL_PREFIX + std::string("/lib"));
        command.push_back("-lDeviceDiscoveryModule");
    }

    std::vector<char*> args;
    args.reserve(command.size() + 1);

    for (auto& itr : command)
        args.push_back(const_cast<char*>(itr.c_str()));

    args.push_back(nullptr);

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
    if (waitpid(pid, &status, 0) < 0) {
        perror("parascc: waitpid failed");
        return 1;
    }
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return 1;
}
} // namespace executor
