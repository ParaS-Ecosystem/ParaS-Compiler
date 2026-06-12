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

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <paras/ParaSLambdaHandler.hpp>

void ParaSLambdaHandler::run(
    const clang::ast_matchers::MatchFinder::MatchResult &result) {
  if (const clang::LambdaExpr *Lambda =
          result.Nodes.getNodeAs<clang::LambdaExpr>("lambda")) {
    llvm::outs() << "Found lambda at: ";
    Lambda->getBeginLoc().print(llvm::outs(), *result.SourceManager);
    llvm::outs() << "\n";
  }
}
