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
