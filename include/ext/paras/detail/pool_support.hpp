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

#ifndef __SYCL_EXT_PARAS_POOL_SUPPORT_HPP__
#define __SYCL_EXT_PARAS_POOL_SUPPORT_HPP__

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ext/paras/command_graph_types.hpp"

namespace sycl {
namespace ext {
namespace paras {
namespace detail {

class shared_flag {
public:
    operator bool() const noexcept { return *value_; }
    shared_flag& operator=(bool v) noexcept {
        *value_ = v;
        return *this;
    }

private:
    std::shared_ptr<bool> value_ = std::make_shared<bool>(false);
};

struct record_state {
    std::string error;
    std::vector<std::shared_ptr<void>> host_tasks;
};

}
}
}
}

#endif
