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

#ifndef __PARAS_LAMBDA_HANDLER_HPP__
#define __PARAS_LAMBDA_HANDLER_HPP__

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/ADT/DenseSet.h>

#include <string>
#include <utility>
#include <vector>

class ParaSLambdaHandler
    : public clang::ast_matchers::MatchFinder::MatchCallback {

private: 
  clang::Rewriter &rewriter;
  std::vector<std::string> backend_target;
  llvm::DenseSet<const clang::CXXMethodDecl *> processedKernels;

public:
  ParaSLambdaHandler(clang::Rewriter &r, std::vector<std::string> backend)
      : rewriter(r), backend_target(std::move(backend)) {}
      
  void
  run(const clang::ast_matchers::MatchFinder::MatchResult &result) override;
};

#endif
