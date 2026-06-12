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

#include "paras/ParaSParallelforHandler.hpp"

#include <clang/Lex/Lexer.h>

void ParallelForFunctionCallback::run(
    const clang::ast_matchers::MatchFinder::MatchResult &result) {
  if (const clang::CallExpr *CE =
          result.Nodes.getNodeAs<clang::CallExpr>("parallelforCall")) {
    if (!CE || CE->getNumArgs() == 0)
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

    std::string new_text =
        "std::thread(" + lambda_function_text.str() + ").join()";

    rewriter.ReplaceText(CE->getSourceRange(), new_text);
    HandleHeaders(*SM);
  }
}

void ParallelForFunctionCallback::HandleHeaders(
    const clang::SourceManager &SM) {
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

  llvm::StringRef code = SM.getBufferData(mainFileID);
  if (code.contains("#include <thread>"))
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

  rewriter.InsertText(insertlocation, "#include <thread>");
}
