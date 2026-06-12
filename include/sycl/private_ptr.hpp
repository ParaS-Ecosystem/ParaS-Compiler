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

#ifndef __PARAS_PRIVATE_PTR_HPP__
#define __PARAS_PRIVATE_PTR_HPP__

#include "access.hpp"
#include "multi_ptr.hpp"

namespace sycl {

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using private_ptr =
    multi_ptr<ElementType, access::address_space::private_space, IsDecorated>;

} // namespace sycl

#endif
