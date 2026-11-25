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

#ifndef __SYCL_EXT_PARAS_POOL_SLOT_HPP__
#define __SYCL_EXT_PARAS_POOL_SLOT_HPP__

#include <memory>
#include <mutex>

#include "ext/paras/command_graph_types.hpp"

namespace sycl {
namespace ext {
namespace paras {
namespace detail {

template <typename Pool>
class pool_slot {
public:
    template <typename Make>
    std::shared_ptr<Pool> get_or_create(Make make) const {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        if (!shared_->owner) shared_->owner = make();
        return shared_->owner;
    }
    std::shared_ptr<Pool> get() const {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        return shared_->owner;
    }
    void detach() { shared_ = std::make_shared<shared>(); }

private:
    struct shared {
        std::mutex mutex;
        std::shared_ptr<Pool> owner;
    };
    std::shared_ptr<shared> shared_ = std::make_shared<shared>();
};

}
}
}
}

#endif
