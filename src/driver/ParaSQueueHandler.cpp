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

#include "paras/ParaSQueueHandler.hpp"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"

void QueueHandler::run(
    const clang::ast_matchers::MatchFinder::MatchResult &result) {
  const clang::SourceManager &SM = *result.SourceManager;
  const clang::VarDecl *VD =
      result.Nodes.getNodeAs<clang::VarDecl>("queue-definer");
  if (!VD)
    return;

  clang::TypeSourceInfo *TSI = VD->getTypeSourceInfo();
  if (!TSI)
    return;

  clang::TypeLoc TL = TSI->getTypeLoc();
  clang::CharSourceRange range =
      clang::CharSourceRange::getTokenRange(TL.getBeginLoc(), TL.getEndLoc());

  rewriter.ReplaceText(range, "threadpool");
}
