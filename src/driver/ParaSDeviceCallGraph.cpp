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

#include "paras/ParaSDeviceCallGraph.hpp"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"

const clang::FunctionDecl *
ParaSDeviceCallGraph::normalize(const clang::FunctionDecl *FD) const {
  if (!FD)
    return nullptr;

  return FD->getCanonicalDecl();
}

const clang::FunctionDecl *
ParaSDeviceCallGraph::annotationIdentity(const clang::FunctionDecl *FD) const {
  if (!FD)
    return nullptr;

  if (const clang::FunctionDecl *pattern =
          FD->getTemplateInstantiationPattern())
    return pattern->getCanonicalDecl();

  return FD->getCanonicalDecl();
}

bool ParaSDeviceCallGraph::isRewritable(const clang::FunctionDecl *FD,
                                        const clang::SourceManager &SM) const {
  if (!FD)
    return false;

  clang::SourceLocation loc = SM.getSpellingLoc(FD->getBeginLoc());
  if (loc.isInvalid())
    return false;

  if (SM.isInSystemHeader(loc) || SM.isInExternCSystemHeader(loc))
    return false;

  if (FD->hasAttr<clang::CUDAGlobalAttr>())
    return false;

  return true;
}

void ParaSDeviceCallGraph::enqueue(const clang::FunctionDecl *FD) {
  FD = normalize(FD);
  if (!FD)
    return;

  if (visited.insert(FD).second)
    workList.push_back(FD);
}

void ParaSDeviceCallGraph::addKernel(const clang::CXXMethodDecl *callOperator) {
  enqueue(callOperator);
}

void ParaSDeviceCallGraph::annotate(const clang::FunctionDecl *FD,
                                    clang::ASTContext &context) {
  const clang::FunctionDecl *definition = nullptr;
  if (!FD || !FD->hasBody(definition) || !definition)
    return;

  clang::SourceManager &SM = context.getSourceManager();
  if (!isRewritable(definition, SM))
    return;

  if (definition->hasAttr<clang::CUDADeviceAttr>() ||
      definition->hasAttr<clang::CUDAGlobalAttr>())
    return;

  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(definition)) {
    if (method->getParent() && method->getParent()->isLambda())
      return;
  }

  const clang::FunctionDecl *annotationKey = annotationIdentity(definition);
  if (!annotationKey || !annotated.insert(annotationKey).second)
    return;

  bool insertedAny = false;
  for (const clang::FunctionDecl *redecl : annotationKey->redecls()) {
    if (!redecl || !isRewritable(redecl, SM))
      continue;

    if (redecl->hasAttr<clang::CUDADeviceAttr>() ||
        redecl->hasAttr<clang::CUDAGlobalAttr>())
      continue;

    clang::SourceLocation loc = SM.getSpellingLoc(redecl->getBeginLoc());
    if (loc.isInvalid() || loc.isMacroID())
      continue;

    if (rewriter.InsertTextBefore(loc, "__host__ __device__ ")) {
      llvm::errs() << "[ParaS] failed to mark device-callable redeclaration: "
                   << redecl->getQualifiedNameAsString() << "\n";
      continue;
    }

    insertedAny = true;
  }

  if (insertedAny)
    llvm::outs() << "[ParaS] device-callable: "
                 << definition->getQualifiedNameAsString()
                 << " (all redeclarations)\n";
}

void ParaSDeviceCallGraph::process(const clang::FunctionDecl *FD,
                                   clang::ASTContext &context) {
  const clang::FunctionDecl *definition = nullptr;
  if (!FD || !FD->hasBody(definition) || !definition)
    return;

  annotate(definition, context);

  class CalleeVisitor : public clang::RecursiveASTVisitor<CalleeVisitor> {
    ParaSDeviceCallGraph &graph;

  public:
    llvm::DenseSet<const clang::VarDecl *> staticConstexprCandidates;

    explicit CalleeVisitor(ParaSDeviceCallGraph &g) : graph(g) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr *expr) {
      if (!expr)
        return true;

      const auto *var = llvm::dyn_cast<clang::VarDecl>(expr->getDecl());
      if (!var || !var->isStaticLocal() || !var->isConstexpr() ||
          var->getStorageClass() != clang::SC_Static)
        return true;

      staticConstexprCandidates.insert(var->getCanonicalDecl());
      return true;
    }

    bool VisitCallExpr(clang::CallExpr *call) {
      if (const clang::FunctionDecl *callee = call->getDirectCallee())
        graph.enqueue(callee);
      return true;
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
      graph.enqueue(expr->getConstructor());
      return true;
    }
  } visitor(*this);

  visitor.TraverseStmt(const_cast<clang::Stmt *>(definition->getBody()));

  clang::SourceManager &SM = context.getSourceManager();

  for (const clang::VarDecl *VD : visitor.staticConstexprCandidates) {
    if (!VD)
      continue;

    clang::SourceLocation loc = SM.getSpellingLoc(VD->getBeginLoc());
    if (loc.isInvalid() || loc.isMacroID() || SM.isInSystemHeader(loc) ||
        SM.isInExternCSystemHeader(loc))
      continue;

    clang::SourceLocation endLoc = SM.getSpellingLoc(VD->getLocation());
    if (endLoc.isInvalid())
      continue;

    const clang::CharSourceRange prefixRange =
        clang::CharSourceRange::getTokenRange(loc, endLoc);
    const std::string rewrittenPrefix = rewriter.getRewrittenText(prefixRange);

    if (rewrittenPrefix.find("__device__") != std::string::npos)
      continue;

    if (rewriter.InsertTextBefore(loc, "__device__ ")) {
      llvm::errs()
          << "[ParaS] failed to mark static constexpr device variable: "
          << VD->getQualifiedNameAsString() << "\n";
      continue;
    }

    llvm::outs() << "[ParaS] device-visible static constexpr variable: "
                 << VD->getQualifiedNameAsString() << "\n";
  }
}

void ParaSDeviceCallGraph::run(clang::ASTContext &context) {
  while (!workList.empty()) {
    const clang::FunctionDecl *FD = workList.pop_back_val();
    process(FD, context);
  }
}
