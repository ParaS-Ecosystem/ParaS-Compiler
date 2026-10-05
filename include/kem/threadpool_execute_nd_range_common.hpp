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

#ifndef __PARAS_THREADPOOL_EXECUTE_ND_RANGE_COMMON_HPP__
#define __PARAS_THREADPOOL_EXECUTE_ND_RANGE_COMMON_HPP__

#include <cstddef>
#include <string>

#include "kem/host_group_runtime.hpp"
#include "kem/nd_range_validate.hpp"
#include "sycl/device.hpp"
#include "sycl/exception.hpp"
#include "sycl/nd_item.hpp"
#include "sycl/nd_range.hpp"

namespace paras_host_detail {

template <int D>
inline sycl::id<D> delinearize(std::size_t lin, const sycl::range<D>& r) {
    sycl::id<D> out;
    for (int d = D - 1; d >= 0; --d) {
        out[d] = lin % r[d];
        lin /= r[d];
    }
    return out;
}

template <int D, typename Func>
void execute_nd_range(const sycl::nd_range<D>& r, Func& f) {
    paras_detail::validate_nd_range(r);
    static const std::size_t max_wg =
        sycl::device{}.get_info<sycl::info::device::max_work_group_size>();
    if (r.get_local_range().size() > max_wg) {
        throw sycl::exception(
            sycl::make_error_code(sycl::errc::nd_range),
            "nd_range: work-group size " + std::to_string(r.get_local_range().size()) +
                " exceeds the device's max_work_group_size " + std::to_string(max_wg));
    }

    const sycl::range<D> GR = r.get_global_range();
    const sycl::range<D> LR = r.get_local_range();
    const sycl::range<D> GRR = r.get_group_range();

    std::size_t num_groups = 1, local_size = 1;
    for (int d = 0; d < D; ++d) {
        num_groups *= GRR[d];
        local_size *= LR[d];
    }

    run_work_groups(num_groups, local_size, [&](std::size_t group_lin, std::size_t local_lin) {
        const sycl::id<D> grp = delinearize<D>(group_lin, GRR);
        const sycl::id<D> lid = delinearize<D>(local_lin, LR);
        sycl::id<D> gid;
        for (int d = 0; d < D; ++d)
            gid[d] = grp[d] * LR[d] + lid[d];
        sycl::nd_item<D> item(gid, lid, grp, GR, LR, GRR);
        f(item);
    });
}

} // namespace paras_host_detail

#endif
