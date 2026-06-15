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

#ifndef __PARAS_FUNC_PARAMETER_HANDLER_HPP__
#define __PARAS_FUNC_PARAMETER_HANDLER_HPP__

#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Rewrite/Core/Rewriter.h>

class FunctionParameterHandler
    : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  clang::Rewriter &rewriter;

public:
  FunctionParameterHandler(clang::Rewriter &R) : rewriter(R) {}

  void
  run(const clang::ast_matchers::MatchFinder::MatchResult &result) override;

  ~FunctionParameterHandler() = default;
};

#endif
