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

#include "paras/ParaSConsumer.hpp"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include <cstdlib>
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang::ast_matchers;

std::unordered_set<std::string> SubmitFunctionCallback::ProcessedFiles;
std::unordered_set<std::string> ParallelForFunctionCallback::ProcessedFiles;

ParaSConsumer::ParaSConsumer(clang::Rewriter &r,
                             std::vector<std::string> backend_target)
    : pf_callback(r), s_callback(r, backend_target), qh_callback(r),
      fph_callback(r), vdr_callback(r, backend_target),
      lambda_callback(r, backend_target), rewriter(r) {
  gpuBuild = !backend_target.empty() &&
             (backend_target[0] == "cuda" || backend_target[0] == "hip");

  const char *noQueueRewrite = std::getenv("PARAS_NO_QUEUE_REWRITE");
  const bool rewriteQueues = !(noQueueRewrite && noQueueRewrite[0] == '1');
  if (rewriteQueues) {
  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::callExpr(clang::ast_matchers::callee(
              clang::ast_matchers::memberExpr(
                  clang::ast_matchers::member(
                      clang::ast_matchers::hasName("sycl::queue::submit")))
                  .bind("memberexpr"))))
          .bind("submitCall"),
      &s_callback);

  matchers.addMatcher(
      traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          callExpr(callee(memberExpr(member(hasName("sycl::queue::submit")))
                              .bind("memberexpr-pf")),
                   hasArgument(
                       0, lambdaExpr(hasDescendant(
                              callExpr(callee(functionDecl(hasName(
                                           "sycl::handler::parallel_for"))),
                                       hasDescendant(cxxConstructExpr(hasType(
                                           classTemplateSpecializationDecl(
                                               hasName("sycl::range"),
                                               hasTemplateArgument(
                                                   0, templateArgument().bind(
                                                          "rangeDim")))))))
                                  .bind("parallelforCall"))))))
          .bind("submit-with-parallelfor"),
      &s_callback);

  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::varDecl(
              clang::ast_matchers::hasType(clang::ast_matchers::cxxRecordDecl(
                  clang::ast_matchers::hasName("sycl::queue")))))
          .bind("vardecl-1"),
      &vdr_callback);

  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::varDecl(clang::ast_matchers::hasType(
              clang::ast_matchers::references(clang::ast_matchers::qualType(
                  clang::ast_matchers::hasDeclaration(
                      clang::ast_matchers::recordDecl(
                          clang::ast_matchers::matchesName("sycl::queue"))))))))
          .bind("vardecl-2"),
      &vdr_callback);

  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::fieldDecl(
              clang::ast_matchers::hasType(
                  clang::ast_matchers::qualType(clang::ast_matchers::anyOf(

                      clang::ast_matchers::hasUnqualifiedDesugaredType(
                          clang::ast_matchers::recordType(
                              clang::ast_matchers::hasDeclaration(
                                  clang::ast_matchers::cxxRecordDecl(
                                      clang::ast_matchers::hasName(
                                          "sycl::queue"))))),

                      clang::ast_matchers::references(
                          clang::ast_matchers::hasUnqualifiedDesugaredType(
                              clang::ast_matchers::recordType(
                                  clang::ast_matchers::hasDeclaration(
                                      clang::ast_matchers::cxxRecordDecl(
                                          clang::ast_matchers::hasName(
                                              "sycl::queue"))))))))))
              .bind("vardecl-4")),
      &vdr_callback);

  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::fieldDecl(
              clang::ast_matchers::hasType(
                  clang::ast_matchers::qualType(clang::ast_matchers::pointsTo(
                      clang::ast_matchers::hasUnqualifiedDesugaredType(
                          clang::ast_matchers::recordType(
                              clang::ast_matchers::hasDeclaration(
                                  clang::ast_matchers::cxxRecordDecl(
                                      clang::ast_matchers::hasName(
                                          "sycl::queue")))))))))
              .bind("fielddecl-3")),
      &vdr_callback);
  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::functionDecl(
              clang::ast_matchers::returns(
                  clang::ast_matchers::qualType(clang::ast_matchers::anyOf(

                      clang::ast_matchers::hasUnqualifiedDesugaredType(
                          clang::ast_matchers::recordType(
                              clang::ast_matchers::hasDeclaration(
                                  clang::ast_matchers::cxxRecordDecl(
                                      clang::ast_matchers::hasName(
                                          "sycl::queue"))))),

                      clang::ast_matchers::references(
                          clang::ast_matchers::hasUnqualifiedDesugaredType(
                              clang::ast_matchers::recordType(
                                  clang::ast_matchers::hasDeclaration(
                                      clang::ast_matchers::cxxRecordDecl(
                                          clang::ast_matchers::hasName(
                                              "sycl::queue"))))))))))
              .bind("vardecl-5")),
      &vdr_callback);

  matchers.addMatcher(
      clang::ast_matchers::cxxConstructExpr(
          clang::ast_matchers::hasDeclaration(
              clang::ast_matchers::cxxConstructorDecl(
                  clang::ast_matchers::isCopyConstructor(),
                  clang::ast_matchers::ofClass(
                      clang::ast_matchers::cxxRecordDecl(
                          clang::ast_matchers::hasName("sycl::queue"))))))
          .bind("queue-copy-construct"),
      &vdr_callback);

  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::varDecl(
              clang::ast_matchers::hasType(
                  clang::ast_matchers::qualType(clang::ast_matchers::pointsTo(
                      clang::ast_matchers::hasUnqualifiedDesugaredType(
                          clang::ast_matchers::recordType(
                              clang::ast_matchers::hasDeclaration(
                                  clang::ast_matchers::cxxRecordDecl(
                                      clang::ast_matchers::hasName(
                                          "sycl::queue")))))))))
              .bind("vardecl-6")),
      &vdr_callback);

  matchers.addMatcher(
      clang::ast_matchers::traverse(
          clang::TK_IgnoreUnlessSpelledInSource,
          clang::ast_matchers::cxxNewExpr().bind("new-queue")),
      &vdr_callback);

  }

  matchers.addMatcher(
      callExpr(callee(functionDecl(hasAnyName(
                   "sycl::handler::parallel_for", "sycl::handler::single_task",
                   "sycl::handler::parallel_for_work_group"))),
               argumentCountAtLeast(1))
          .bind("paras-kernel-call"),
      &lambda_callback);

  matchers.addMatcher(
      functionDecl(isDefinition(), hasAttr(clang::attr::CUDADevice),
                   unless(isImplicit()))
          .bind("paras-device-function"),
      &lambda_callback);
}

namespace {

class ExplicitHostInstantiations
    : public clang::RecursiveASTVisitor<ExplicitHostInstantiations> {
public:
  ExplicitHostInstantiations(clang::Rewriter &r, clang::ASTContext &c)
      : rewriter(r), context(c), SM(c.getSourceManager()) {}

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitFunctionDecl(clang::FunctionDecl *FD) {
    if (!FD || FD->getTemplateSpecializationKind() !=
                   clang::TSK_ExplicitInstantiationDefinition)
      return true;

    if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(FD))
      if (!method->isStatic())
        return true;
    const clang::FunctionTemplateDecl *FTD = FD->getPrimaryTemplate();
    if (!FTD)
      return true;
    const clang::FunctionDecl *primary = FTD->getTemplatedDecl();
    if (!primary || primary->isConstexpr() || primary->isConsteval() ||
        primary->hasAttr<clang::CUDAHostAttr>() ||
        primary->hasAttr<clang::CUDADeviceAttr>() ||
        primary->hasAttr<clang::CUDAGlobalAttr>())
      return true;
    if (!done.insert(primary->getCanonicalDecl()).second)
      return true;

    llvm::SmallVector<clang::SourceLocation, 4> sites;
    for (const clang::FunctionDecl *redecl : primary->redecls()) {
      clang::SourceLocation loc = SM.getSpellingLoc(redecl->getInnerLocStart());
      clang::SourceLocation nameLoc = SM.getSpellingLoc(redecl->getLocation());
      if (loc.isInvalid() || loc.isMacroID() || nameLoc.isInvalid() ||
          SM.isInSystemHeader(loc) || SM.isInExternCSystemHeader(loc) ||
          SM.getFileID(loc) != SM.getFileID(nameLoc))
        return true;
      const std::string prefix = rewriter.getRewrittenText(
          clang::CharSourceRange::getTokenRange(loc, nameLoc));

      if (prefix.find("__device__") != std::string::npos ||
          prefix.find("__host__") != std::string::npos)
        return true;
      sites.push_back(loc);
    }
    for (clang::SourceLocation loc : sites)
      rewriter.InsertTextBefore(loc, "__host__ ");
    return true;
  }

private:
  clang::Rewriter &rewriter;
  clang::ASTContext &context;
  clang::SourceManager &SM;
  llvm::DenseSet<const clang::FunctionDecl *> done;
};

}

void ParaSConsumer::HandleTranslationUnit(clang::ASTContext &context) {
  matchers.matchAST(context);

  paras::rewriteKernelAttributes(context, rewriter, gpuBuild);
  if (gpuBuild && !std::getenv("PARAS_NO_IMPLICIT_HD_TEMPLATES")) {
    ExplicitHostInstantiations pass(rewriter, context);
    pass.TraverseDecl(context.getTranslationUnitDecl());
  }
}
