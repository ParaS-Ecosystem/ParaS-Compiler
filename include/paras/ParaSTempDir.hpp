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

#ifndef __PARAS_TEMPDIR_HPP__
#define __PARAS_TEMPDIR_HPP__

#include <cstdlib>
#include <string>
#include <system_error>
#include <unistd.h>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

inline const std::string& parasTempDir() {
    static const std::string dir = [] {
        const char* base = std::getenv("TMPDIR");
        std::string d =
            std::string(base && *base ? base : "/tmp") + "/paras-" + std::to_string(::getuid());
        namespace fs = llvm::sys::fs;
        if (std::error_code EC = fs::create_directories(d, true, fs::perms::owner_all)) {
            llvm::errs() << "parascc: cannot create temp directory " << d << ": " << EC.message()
                         << "\n";
            std::exit(EXIT_FAILURE);
        }
        fs::file_status st;
        if (fs::status(d, st) || st.getUser() != ::getuid()) {
            llvm::errs() << "parascc: temp directory " << d
                         << " is not owned by the current user\n";
            std::exit(EXIT_FAILURE);
        }
        return d;
    }();
    return dir;
}

#endif
