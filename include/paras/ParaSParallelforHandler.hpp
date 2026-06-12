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

#ifndef __PARAS_PARALLEL_FOR_HANDLER_HPP__
#define __PARAS_PARALLEL_FOR_HANDLER_HPP__

#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <string>
#include <unordered_set>

class ParallelForFunctionCallback
    : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  clang::Rewriter &rewriter;
  static std::unordered_set<std::string> ProcessedFiles;

public:
  ParallelForFunctionCallback(clang::Rewriter &r) : rewriter(r) {}

  void
  run(const clang::ast_matchers::MatchFinder::MatchResult &result) override;

  void HandleHeaders(const clang::SourceManager &SM);
};

#endif
