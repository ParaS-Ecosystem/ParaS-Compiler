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

#include "paras/ParaSSubmitHandler.hpp"

#include <clang/AST/AST.h>
#include <clang/Lex/Lexer.h>

static bool isAssignedToSyclEvent(const clang::CallExpr *CE,
                                  const clang::SourceManager &SM) {
  if (!CE)
    return false;

  clang::SourceLocation beginLoc = CE->getBeginLoc();
  if (beginLoc.isInvalid())
    return false;

  clang::FileID fileID = SM.getFileID(beginLoc);
  unsigned offset = SM.getFileOffset(beginLoc);

  llvm::StringRef code = SM.getBufferData(fileID);
  if (offset >= code.size())
    return false;

  size_t lineStart = code.rfind('\n', offset);
  if (lineStart == llvm::StringRef::npos)
    lineStart = 0;
  else
    lineStart++;

  llvm::StringRef beforeCall = code.slice(lineStart, offset);

  return beforeCall.contains("sycl::event");
}

void SubmitFunctionCallback::run(
    const clang::ast_matchers::MatchFinder::MatchResult &result) {
  std::string tpool_func_Call = "";

  if (const clang::TemplateArgument *DimArg =
          result.Nodes.getNodeAs<clang::TemplateArgument>("rangeDim")) {
    unsigned N = DimArg->getAsIntegral().getExtValue();

    if (N == 1)
      tpool_func_Call = "spawn_1D";
    else
      tpool_func_Call = "spawn_ND";
  }

  if (const clang::CallExpr *CE =
          result.Nodes.getNodeAs<clang::CallExpr>("submit-with-parallelfor")) {
    auto &SM = result.SourceManager;
    const clang::Expr *Args = CE->getArg(0);
    auto langopts = result.Context->getLangOpts();

    clang::CharSourceRange args_range =
        clang::CharSourceRange::getTokenRange(Args->getSourceRange());

    llvm::StringRef kernel_function_text =
        clang::Lexer::getSourceText(args_range, *SM, langopts);

    if (kernel_function_text.empty())
      return;

    const clang::MemberExpr *ME =
        result.Nodes.getNodeAs<clang::MemberExpr>("memberexpr-pf");

    if (!ME)
      return;

    clang::Expr *base = ME->getBase();
    if (!base)
      return;

    clang::CharSourceRange baseRange =
        clang::CharSourceRange::getTokenRange(base->getSourceRange());

    llvm::StringRef objectName =
        clang::Lexer::getSourceText(baseRange, *SM, langopts);

    if (objectName.empty())
      return;

    if (tpool_func_Call == "spawn_1D" && isAssignedToSyclEvent(CE, *SM)) {
      tpool_func_Call = "spawn_1D_event";
    }

    std::string new_text = objectName.str() + "." + tpool_func_Call + "(" +
                           kernel_function_text.str() + ")";

    rewriter.ReplaceText(CE->getSourceRange(), new_text);
    HandleHeaders(*SM);
    return;
  }

  if (const clang::CallExpr *CE =
          result.Nodes.getNodeAs<clang::CallExpr>("submitCall")) {
    if (!CE || CE->getNumArgs() == 0)
      return;

    const clang::MemberExpr *ME =
        result.Nodes.getNodeAs<clang::MemberExpr>("memberexpr");

    if (!ME)
      return;

    const clang::Expr *base = ME->getBase();
    if (!base)
      return;

    auto &SM = result.SourceManager;
    const clang::Expr *Args = CE->getArg(0);
    auto langopts = result.Context->getLangOpts();

    clang::CharSourceRange args_range =
        clang::CharSourceRange::getTokenRange(Args->getSourceRange());

    llvm::StringRef lambda_function_text =
        clang::Lexer::getSourceText(args_range, *SM, langopts);

    if (lambda_function_text.empty())
      return;

    clang::CharSourceRange baseRange =
        clang::CharSourceRange::getTokenRange(base->getSourceRange());

    llvm::StringRef objectName =
        clang::Lexer::getSourceText(baseRange, *SM, langopts);

    if (objectName.empty())
      return;

    std::string submit_func =
        isAssignedToSyclEvent(CE, *SM) ? "spawn_1D_event" : "spawn_1D";

    std::string new_text = objectName.str() + "." + submit_func + "(" +
                           lambda_function_text.str() + ")";

    rewriter.ReplaceText(CE->getSourceRange(), new_text);
    HandleHeaders(*SM);
  }
}

void SubmitFunctionCallback::HandleHeaders(const clang::SourceManager &SM) {
  clang::FileID mainFileID = SM.getMainFileID();
  auto entry = SM.getFileEntryRefForID(mainFileID);

  if (!entry)
    return;

  llvm::StringRef filepath = entry->getName();
  const clang::FileEntry &FE = entry->getFileEntry();
  llvm::StringRef fullpath = FE.tryGetRealPathName();

  if (!fullpath.empty())
    filepath = fullpath;

  std::string filepathStr = filepath.str();

  if (ProcessedFiles.count(filepathStr))
    return;

  ProcessedFiles.insert(filepathStr);

  std::string new_header = "";

  for (std::string each_bkend_target : bkend_target) {
    if (each_bkend_target == "cuda") {
      new_header = "#include <kem_gpu/gpu_threadpool.hpp>";
      llvm::StringRef code = SM.getBufferData(mainFileID);

      if (code.contains(new_header) ||
          code.contains("#include \"kem_gpu/gpu_threadpool.hpp\""))
        return;

      size_t insertPosition = code.rfind("#include");
      clang::SourceLocation insertlocation =
          SM.getLocForStartOfFile(mainFileID);

      if (insertPosition != llvm::StringRef::npos) {
        unsigned line = 1;

        for (size_t i = 0; i < insertPosition; i++)
          if (code[i] == '\n')
            line++;

        insertlocation = SM.translateLineCol(mainFileID, line + 1, 1);
      }

      rewriter.InsertText(insertlocation,
                          "#include <kem_gpu/gpu_threadpool.hpp>\n");

      return;
    } else if (each_bkend_target == "hip") {
      new_header = "#include <kem_gpu/rocm_threadpool.hpp>";
      llvm::StringRef code = SM.getBufferData(mainFileID);

      if (code.contains(new_header) ||
          code.contains("#include \"kem_gpu/rocm_threadpool.hpp\""))
        return;

      size_t insertPosition = code.rfind("#include");
      clang::SourceLocation insertlocation =
          SM.getLocForStartOfFile(mainFileID);

      if (insertPosition != llvm::StringRef::npos) {
        unsigned line = 1;

        for (size_t i = 0; i < insertPosition; i++)
          if (code[i] == '\n')
            line++;

        insertlocation = SM.translateLineCol(mainFileID, line + 1, 1);
      }

      rewriter.InsertText(insertlocation,
                          "#include <kem_gpu/HIPThreadpool.hpp>\n");

      return;
    }
  }

  llvm::StringRef code = SM.getBufferData(mainFileID);

  if (code.contains("#include <kem/threadpool.hpp>") ||
      code.contains("#include \"kem/threadpool.hpp\""))
    return;

  size_t insertPosition = code.rfind("#include");
  clang::SourceLocation insertlocation = SM.getLocForStartOfFile(mainFileID);

  if (insertPosition != llvm::StringRef::npos) {
    unsigned line = 1;

    for (size_t i = 0; i < insertPosition; i++)
      if (code[i] == '\n')
        line++;

    insertlocation = SM.translateLineCol(mainFileID, line + 1, 1);
  }

  rewriter.InsertText(insertlocation, "#include <kem/threadpool.hpp>\n");
}
