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

#ifndef __PARAS_SAMPLER_HPP__
#define __PARAS_SAMPLER_HPP__

#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

enum class addressing_mode { mirrored_repeat, repeat, clamp_to_edge, clamp, none };
enum class filtering_mode { nearest, linear };
enum class coordinate_normalization_mode { normalized, unnormalized };

class sampler {
    coordinate_normalization_mode normalization_;
    addressing_mode addressing_;
    filtering_mode filtering_;

public:
    PARAS_KERNEL_HD
    sampler(coordinate_normalization_mode normalizationMode, addressing_mode addressingMode,
            filtering_mode filteringMode)
        : normalization_(normalizationMode), addressing_(addressingMode),
          filtering_(filteringMode) {}

    sampler(const sampler& rhs) = default;
    sampler& operator=(const sampler& rhs) = default;

    PARAS_KERNEL_HD
    addressing_mode get_addressing_mode() const { return addressing_; }
    PARAS_KERNEL_HD
    filtering_mode get_filtering_mode() const { return filtering_; }
    PARAS_KERNEL_HD
    coordinate_normalization_mode get_coordinate_normalization_mode() const {
        return normalization_;
    }

    PARAS_KERNEL_HD
    bool operator==(const sampler& rhs) const {
        return normalization_ == rhs.normalization_ && addressing_ == rhs.addressing_ &&
               filtering_ == rhs.filtering_;
    }
    PARAS_KERNEL_HD
    bool operator!=(const sampler& rhs) const { return !(*this == rhs); }
};

} // namespace sycl

#endif
