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

#include "paras/ParaSFuncParameterHandler.hpp"
#include "clang/AST/Decl.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"

void FunctionParameterHandler::run(
    const clang::ast_matchers::MatchFinder::MatchResult &result) {
  const clang::ParmVarDecl *PVD =
      result.Nodes.getNodeAs<clang::ParmVarDecl>("functionparameter");
  if (!PVD)
    return;

  clang::TypeLoc TL = PVD->getTypeSourceInfo()->getTypeLoc();
  clang::TypeLoc baseType = TL.getUnqualifiedLoc();
  const clang::SourceManager *SM = result.SourceManager;

  rewriter.ReplaceText(baseType.getBeginLoc(),
                       baseType.getSourceRange().printToString(*SM).size(),
                       "threadpool");
}
