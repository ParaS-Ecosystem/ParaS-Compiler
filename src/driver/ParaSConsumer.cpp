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
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TemplateBase.h"
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
      fph_callback(r), vdr_callback(r, backend_target) {

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
}

void ParaSConsumer::HandleTranslationUnit(clang::ASTContext &context) {
  matchers.matchAST(context);
}
