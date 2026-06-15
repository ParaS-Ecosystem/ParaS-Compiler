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

#include "paras/ParaSVarDeclHandler.hpp"
#include <system_error>

#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <iostream>

void VarDeclReplacer::run(
    const clang::ast_matchers::MatchFinder::MatchResult &result) {

  if (const clang::VarDecl *VD =
          result.Nodes.getNodeAs<clang::VarDecl>("vardecl-1")) {
    int flag = 0;
    const clang::Expr *StrippedExpr = VD->getInit()->IgnoreParenImpCasts();

    if (const auto *EWC =
            llvm::dyn_cast<clang::ExprWithCleanups>(StrippedExpr)) {
      StrippedExpr = EWC->getSubExpr()->IgnoreParenImpCasts();
    }

    if (const auto *MTE =
            llvm::dyn_cast<clang::MaterializeTemporaryExpr>(StrippedExpr)) {
      StrippedExpr = MTE->getSubExpr()->IgnoreParenImpCasts();
    }

    if (const auto *CtorExpr =
            llvm::dyn_cast<clang::CXXConstructExpr>(StrippedExpr)) {
      if (CtorExpr->getNumArgs() == 2) {
        if (const clang::DeclRefExpr *lambda =
                llvm::dyn_cast<clang::DeclRefExpr>(CtorExpr->getArg(0))) {
          if (const clang::VarDecl *selector =
                  llvm::dyn_cast<clang::VarDecl>(lambda->getDecl())) {

            if (selector->getNameAsString() == "cpu_selector_v" &&
                !bkend_target[0].empty()) {
              llvm::errs() << "Error: Found cpu queue but compiling for gpu "
                              "(gave -parasdevice flag)"
                           << "\n";
              exit(1);
            } else if (selector->getNameAsString() == "gpu_selector_v" &&
                       bkend_target[0].empty()) {
              llvm::errs() << "Error: Found gpu queue but no device found!"
                           << "\n";
              exit(1);
            } else if (selector->getNameAsString() == "gpu_selector_v" &&
                       !bkend_target[0].empty()) {
              for (std::string each_bkend_target : bkend_target) {
                if (each_bkend_target == "cuda") {
                  flag = 1;
                } else if (each_bkend_target == "hip") {
                  flag = 2;
                }
              }
            }
          }
        }
      }
    }

    const clang::SourceManager *SM = result.SourceManager;
    clang::SourceLocation type_startloc = VD->getTypeSpecStartLoc();
    clang::SourceLocation type_endloc = VD->getTypeSpecEndLoc();
    if (flag == 1) {
      rewriter.ReplaceText(clang::SourceRange(type_startloc, type_endloc),
                           "cuda_threadpool");
      return;
    }

    else if (flag == 2) {
      rewriter.ReplaceText(clang::SourceRange(type_startloc, type_endloc),
                           "rocm_threadpool");
      return;
    } else {
      if (bkend_target[0].empty()) {
        rewriter.ReplaceText(clang::SourceRange(type_startloc, type_endloc),
                             "threadpool");
      } else {
        for (std::string each_bkend_target : bkend_target) {
          if (each_bkend_target == "cuda") {
            rewriter.ReplaceText(clang::SourceRange(type_startloc, type_endloc),
                                 "cuda_threadpool");
            return;
          } else if (each_bkend_target == "hip") {
            rewriter.ReplaceText(clang::SourceRange(type_startloc, type_endloc),
                                 "rocm_threadpool");
            return;
          }
        }
      }
    }

    return;
  }

  if (const clang::VarDecl *VD =
          result.Nodes.getNodeAs<clang::VarDecl>("vardecl-2")) {
    int flag = 0;
    if (VD->hasInit()) {
      if (const clang::DeclRefExpr *RefVar =
              llvm::dyn_cast<clang::DeclRefExpr>(VD->getInit())) {
        if (const clang::VarDecl *RefVD =
                llvm::dyn_cast<clang::VarDecl>(RefVar->getDecl())) {
          const clang::Expr *StrippedExpr =
              RefVD->getInit()->IgnoreParenImpCasts();

          if (const auto *EWC =
                  llvm::dyn_cast<clang::ExprWithCleanups>(StrippedExpr)) {
            StrippedExpr = EWC->getSubExpr()->IgnoreParenImpCasts();
          }

          if (const auto *MTE = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(
                  StrippedExpr)) {
            StrippedExpr = MTE->getSubExpr()->IgnoreParenImpCasts();
          }

          if (const auto *CtorExpr =
                  llvm::dyn_cast<clang::CXXConstructExpr>(StrippedExpr)) {
            if (CtorExpr->getNumArgs() == 2) {
              if (const clang::DeclRefExpr *lambda =
                      llvm::dyn_cast<clang::DeclRefExpr>(CtorExpr->getArg(0))) {
                if (const clang::VarDecl *selector =
                        llvm::dyn_cast<clang::VarDecl>(lambda->getDecl())) {
                  if (selector->getNameAsString() == "cpu_selector_v" &&
                      !bkend_target[0].empty()) {
                    llvm::errs() << "Error: Found cpu queue but compiling for "
                                    "gpu (gave -parasdevice flag)"
                                 << "\n";
                    exit(1);
                  } else if (selector->getNameAsString() == "gpu_selector_v" &&
                             bkend_target[0].empty()) {
                    llvm::errs()
                        << "Error: Found gpu queue but no device found!"
                        << "\n";
                    exit(1);
                  } else if (selector->getNameAsString() == "gpu_selector_v" &&
                             !bkend_target[0].empty()) {
                    for (std::string each_bkend_target : bkend_target) {
                      if (each_bkend_target == "cuda") {
                        flag = 1;
                      } else if (each_bkend_target == "hip") {
                        flag = 2;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    const clang::SourceManager *SM = result.SourceManager;
    clang::SourceLocation type_startloc = VD->getTypeSpecStartLoc();
    clang::QualType pointeetype = VD->getType()->getPointeeType();
    clang::TypeLoc TL = VD->getTypeSourceInfo()->getTypeLoc();
    clang::ReferenceTypeLoc RTL = TL.castAs<clang::ReferenceTypeLoc>();
    clang::TypeLoc InnerTL = RTL.getPointeeLoc();
    clang::SourceRange replaceRange = InnerTL.getSourceRange();

    if (flag == 1) {
      rewriter.ReplaceText(replaceRange, "cuda_threadpool");
      return;
    }

    else if (flag == 2) {
      rewriter.ReplaceText(replaceRange, "rocm_threadpool");
      return;
    } else {
      if (bkend_target[0].empty()) {
        rewriter.ReplaceText(replaceRange, "threadpool");
      } else {
        for (std::string each_bkend_target : bkend_target) {
          if (each_bkend_target == "cuda") {
            rewriter.ReplaceText(replaceRange, "cuda_threadpool");
            return;
          } else if (each_bkend_target == "hip") {
            rewriter.ReplaceText(replaceRange, "rocm_threadpool");
            return;
          }
        }
      }
    }
  }

  if (const clang::VarDecl *VD =
          result.Nodes.getNodeAs<clang::VarDecl>("vardecl-3")) {
  }

  if (const clang::FieldDecl *FD =
          result.Nodes.getNodeAs<clang::FieldDecl>("vardecl-4")) {
    int flag = 0;

    if (!FD->getType()->isReferenceType()) {

      if (FD->hasInClassInitializer()) {

        const clang::Expr *StrippedExpr =
            FD->getInClassInitializer()->IgnoreParenImpCasts();

        if (const auto *EWC =
                llvm::dyn_cast<clang::ExprWithCleanups>(StrippedExpr)) {
          StrippedExpr = EWC->getSubExpr()->IgnoreParenImpCasts();
        }

        if (const auto *MTE =
                llvm::dyn_cast<clang::MaterializeTemporaryExpr>(StrippedExpr)) {
          StrippedExpr = MTE->getSubExpr()->IgnoreParenImpCasts();
        }

        if (const auto *CtorExpr =
                llvm::dyn_cast<clang::CXXConstructExpr>(StrippedExpr)) {

          if (CtorExpr->getNumArgs() == 2) {

            if (const clang::DeclRefExpr *lambda =
                    llvm::dyn_cast<clang::DeclRefExpr>(CtorExpr->getArg(0))) {

              if (const clang::VarDecl *selector =
                      llvm::dyn_cast<clang::VarDecl>(lambda->getDecl())) {

                if (selector->getNameAsString() == "cpu_selector_v" &&
                    !bkend_target[0].empty()) {

                  llvm::errs()
                      << "Error: Found cpu queue but compiling for gpu "
                         "(gave -parasdevice flag)\n";
                  exit(1);
                }

                else if (selector->getNameAsString() == "gpu_selector_v" &&
                         bkend_target[0].empty()) {

                  llvm::errs()
                      << "Error: Found gpu queue but no device found!\n";
                  exit(1);
                }

                else if (selector->getNameAsString() == "gpu_selector_v" &&
                         !bkend_target[0].empty()) {

                  for (std::string each_bkend_target : bkend_target) {

                    if (each_bkend_target == "cuda") {
                      flag = 1;
                    }

                    else if (each_bkend_target == "hip") {
                      flag = 2;
                    }
                  }
                }
              }
            }
          }
        }
      }

      clang::TypeLoc TL = FD->getTypeSourceInfo()->getTypeLoc();

      clang::SourceRange replaceRange = TL.getSourceRange();

      if (flag == 1) {

        rewriter.ReplaceText(replaceRange, "cuda_threadpool");

        return;
      }

      else if (flag == 2) {

        rewriter.ReplaceText(replaceRange, "rocm_threadpool");

        return;
      }

      else {

        if (bkend_target[0].empty()) {

          rewriter.ReplaceText(replaceRange, "threadpool");
          return;
        }

        else {

          for (std::string each_bkend_target : bkend_target) {

            if (each_bkend_target == "cuda") {

              rewriter.ReplaceText(replaceRange, "cuda_threadpool");

              return;
            }

            else if (each_bkend_target == "hip") {

              rewriter.ReplaceText(replaceRange, "rocm_threadpool");

              return;
            }
          }
        }
      }
    }

    else {

      clang::TypeLoc TL = FD->getTypeSourceInfo()->getTypeLoc();

      if (auto RTL = TL.getAs<clang::ReferenceTypeLoc>()) {
        TL = RTL.getPointeeLoc();
      }

      clang::SourceRange replaceRange = TL.getSourceRange();

      if (flag == 1) {

        rewriter.ReplaceText(replaceRange, "cuda_threadpool");

        return;
      }

      else if (flag == 2) {

        rewriter.ReplaceText(replaceRange, "rocm_threadpool");

        return;
      }

      else {

        if (bkend_target[0].empty()) {

          rewriter.ReplaceText(replaceRange, "threadpool");
          return;
        }

        else {

          for (std::string each_bkend_target : bkend_target) {

            if (each_bkend_target == "cuda") {

              rewriter.ReplaceText(replaceRange, "cuda_threadpool");

              return;
            }

            else if (each_bkend_target == "hip") {

              rewriter.ReplaceText(replaceRange, "rocm_threadpool");

              return;
            }
          }
        }
      }
    }
  }

  if (const clang::FieldDecl *FD =
          result.Nodes.getNodeAs<clang::FieldDecl>("fielddecl-3")) {

    clang::TypeSourceInfo *TSI = FD->getTypeSourceInfo();

    if (!TSI)
      return;

    clang::TypeLoc TL = TSI->getTypeLoc();

    if (auto PTL = TL.getAs<clang::PointerTypeLoc>()) {

      TL = PTL.getPointeeLoc();
    }

    clang::SourceRange replaceRange = TL.getSourceRange();

    if (!replaceRange.isValid())
      return;

    for (std::string each_bkend_target : bkend_target) {

      if (each_bkend_target == "cuda") {

        rewriter.ReplaceText(replaceRange, "cuda_threadpool");

        return;
      }

      else if (each_bkend_target == "hip") {

        rewriter.ReplaceText(replaceRange, "roc_threadpool");

        return;
      }
    }

    if (bkend_target[0].empty()) {

      rewriter.ReplaceText(replaceRange, "threadpool");

      return;
    }
  }

  if (const clang::FunctionDecl *FD =
          result.Nodes.getNodeAs<clang::FunctionDecl>("vardecl-5")) {

    clang::TypeSourceInfo *TSI = FD->getTypeSourceInfo();

    if (!TSI)
      return;

    clang::TypeLoc TL = TSI->getTypeLoc();

    auto FTL = TL.getAs<clang::FunctionTypeLoc>();

    if (!FTL)
      return;

    clang::TypeLoc ReturnTL = FTL.getReturnLoc();

    if (auto RTL = ReturnTL.getAs<clang::ReferenceTypeLoc>()) {

      ReturnTL = RTL.getPointeeLoc();
    }

    clang::SourceRange replaceRange = ReturnTL.getSourceRange();

    if (!replaceRange.isValid())
      return;

    for (std::string each_bkend_target : bkend_target) {

      if (each_bkend_target == "cuda") {

        rewriter.ReplaceText(replaceRange, "cuda_threadpool");

        return;
      }

      else if (each_bkend_target == "hip") {

        rewriter.ReplaceText(replaceRange, "roc_threadpool");

        return;
      }
    }

    if (bkend_target[0].empty()) {

      rewriter.ReplaceText(replaceRange, "threadpool");

      return;
    }
  }

  if (const clang::VarDecl *VD =
          result.Nodes.getNodeAs<clang::VarDecl>("vardecl-6")) {

    clang::TypeSourceInfo *TSI = VD->getTypeSourceInfo();

    if (!TSI)
      return;

    clang::TypeLoc TL = TSI->getTypeLoc();

    if (auto PTL = TL.getAs<clang::PointerTypeLoc>()) {

      TL = PTL.getPointeeLoc();
    }

    clang::SourceRange replaceRange = TL.getSourceRange();

    if (!replaceRange.isValid())
      return;

    for (std::string each_bkend_target : bkend_target) {

      if (each_bkend_target == "cuda") {

        rewriter.ReplaceText(replaceRange, "cuda_threadpool");

        return;
      }

      else if (each_bkend_target == "hip") {

        rewriter.ReplaceText(replaceRange, "roc_threadpool");

        return;
      }
    }

    if (bkend_target[0].empty()) {

      rewriter.ReplaceText(replaceRange, "threadpool");

      return;
    }
  }
}
