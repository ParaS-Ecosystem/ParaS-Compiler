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

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/AttrIterator.h"
#include "clang/AST/Attrs.inc"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include <algorithm>
#include <chrono>
#include <clang/AST/AST.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Program.h>
#include <memory>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <vector>

#include "paras/Executor.hpp"
#include "paras/ParaSConsumer.hpp"
#include "paras/Preprocessor.hpp"

#include "pathfinder.hpp"

bool is_link_only = false;
std::string GlobalOutFile = "";
std::vector<std::string> markForDeletion;
std::vector<std::string> transformedSourceFiles;
std::vector<std::string> finalCompilerArgs;

static bool isCppSourceFile(const std::string &arg) {
  auto endsWith = [](const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  return endsWith(arg, ".cpp") || endsWith(arg, ".cc") || endsWith(arg, ".cxx");
}

static llvm::cl::OptionCategory Tooling("paras-expfinder");

static llvm::cl::opt<bool>
    VersionOption("pversion", llvm::cl::desc("Display version information"),
                  llvm::cl::init(false), llvm::cl::cat(Tooling));

static llvm::cl::opt<bool>
    OnlyTransform("otf", llvm::cl::desc("Only transform the code"),
                  llvm::cl::init(false), llvm::cl::cat(Tooling));

static llvm::cl::opt<std::string>
    WriteToFile("outfile", llvm::cl::desc("Specify an output file"),
                llvm::cl::value_desc("filename"), llvm::cl::cat(Tooling));

class ParasFrontendAction : public clang::ASTFrontendAction {
private:
  clang::Rewriter rewriter;
  std::vector<std::string> compilerFlags;
  std::vector<std::string> bkend_target;

  struct TempFileGuard {
    std::string Path;
    bool Keep = false;
    TempFileGuard(std::string p) : Path(std::move(p)) {}
    ~TempFileGuard() {
      llvm::outs() << "[TempFileGuard] Removing: " << Path << "\n";
    }
    void release() { Keep = true; }
  };

  void compileTransformedCode(std::string tempFileName) {
    std::vector<std::string> args;
    args.push_back("clang++-20-tests");
    args.push_back(tempFileName);
    for (const auto &f : compilerFlags)
      args.push_back(f);

    TempFileGuard guard(tempFileName);

    std::string Program = "clang++-20";
    int ExitCode;
    std::vector<llvm::StringRef> ArgRefs;
    ArgRefs.reserve(args.size());
    for (const auto &s : args)
      ArgRefs.emplace_back(s);

    ExitCode = llvm::sys::ExecuteAndWait(
        Program, llvm::ArrayRef<llvm::StringRef>(ArgRefs), std::nullopt,
        std::nullopt, 0, 0);

    if (ExitCode != 0) {
      llvm::errs() << Program << " returned non-zero exit code: " << ExitCode
                   << "\n";
      return;
    } else
      llvm::outs() << Program << " succeeded.\n";

    guard.release();
  }

public:
  ParasFrontendAction() {}

  ParasFrontendAction(const std::vector<std::string> &cf,
                      std::vector<std::string> &bkend_target)
      : compilerFlags(cf), bkend_target(bkend_target) {}

  void ExecuteAction() override { clang::ASTFrontendAction::ExecuteAction(); }

  void EndSourceFileAction() override {
    auto &SM = rewriter.getSourceMgr();
    clang::FileID mainID = SM.getMainFileID();

    std::string updatedFileContents;
    llvm::raw_string_ostream updatedFileStream(updatedFileContents);

    rewriter.getEditBuffer(mainID).write(updatedFileStream);

    if (!bkend_target[0].empty()) {
      std::string back_end = bkend_target[0];
      std::string tar_arch = bkend_target[1];
    }

    if (!WriteToFile.empty()) {
      std::error_code EC;
      llvm::raw_fd_ostream outfile(GlobalOutFile, EC);
      outfile << updatedFileContents;
      return;
    }

    updatedFileStream.flush();
    if (OnlyTransform) {
      llvm::outs() << "\n=========== Transformed code =========== \n "
                   << updatedFileContents << "\n";
      return;
    }

    std::string tmpModel = "/tmp/paras_tmp_%%%%%%%%.cpp";
    llvm::SmallString<128> tempPathSV;
    std::error_code EC;
    if ((EC = llvm::sys::fs::createUniqueFile(tmpModel, tempPathSV))) {
      llvm::errs() << "Issue with creating temp file: " << EC.message() << "\n";
      return;
    }

    std::string tempFile(tempPathSV.str());

    llvm::raw_fd_ostream outFile(tempFile, EC);
    if (EC) {
      llvm::errs() << "Issue with creating " << tempFile << "\n";
      llvm::errs() << "Exited with error code: " << EC.message() << "\n";
      return;
    }
    outFile << updatedFileContents;
    outFile.flush();
    outFile.close();

    transformedSourceFiles.push_back(tempFile);
    markForDeletion.push_back(tempFile);
  }

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef infile) override {
    rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<ParaSConsumer>(rewriter, bkend_target);
  }
};

class ParasFrontendActionFactory
    : public clang::tooling::FrontendActionFactory {
private:
  std::vector<std::string> flags;
  std::vector<std::string> bkend_target;

public:
  ParasFrontendActionFactory(const std::vector<std::string> &F,
                             const std::vector<std::string> &bkend_target)
      : flags(F), bkend_target(bkend_target) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<ParasFrontendAction>(flags, bkend_target);
  }
};

std::vector<std::vector<std::string>> parseCommandLineArgs(int argc,
                                                           const char **argv) {
  std::vector<std::vector<std::string>> result(4);
  std::vector<std::string> args;

  std::string parasLibPath = PARAS_INSTALL_PREFIX + std::string("/lib/");
  std::string parasIncludePath =
      PARAS_INSTALL_PREFIX + std::string("/include/");

  bool have_cpp = false, have_ofile = false;

  for (int i = 0; i < argc; i++) {
    args.push_back(argv[i]);
  }

  std::vector<std::string> bkend_target(4);
  bool found = false;
  bool llvmResDirFound = false;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "-resource-dir")
      llvmResDirFound = true;
    if (args[i] == "-parasdevice") {
      if (found) {
        std::cerr << "Error: multiple -parasdevice options not allowed\n";
        exit(1);
      }
      found = true;
      if (i + 1 < args.size()) {
        std::string value = args[i + 1];
        size_t pos = value.find(':');
        if (pos == std::string::npos) {
          std::cerr << "Error: -parasdevice requires format <backend>:<arch>\n";
          exit(1);
        }

        std::string backend = value.substr(0, pos);
        std::string arch = value.substr(pos + 1);

        if (backend != "cuda" && backend != "hip") {
          std::cerr << "Error: backend must be 'cuda' or 'hip', got '"
                    << backend << "'\n";
          exit(1);
        }
        if (arch.empty()) {
          std::cerr << "Error: architecture part cannot be empty\n";
          exit(1);
        }
        args.push_back("-DPARASDEVICE=1");
        bkend_target[0] = backend;
        bkend_target[1] = arch;
        args.erase(args.begin() + i, args.begin() + i + 2);
        i--;
      }
    }
  }

  auto endsWith = [](const std::string &str,
                     const std::string &suffix) -> bool {
    if (str.length() < suffix.length())
      return false;
    return str.compare(str.length() - suffix.length(), suffix.length(),
                       suffix) == 0;
  };

  std::string outfile_flag;

  for (int i = 0; i < args.size(); i++) {
    const std::string &arg = args[i];
    if (arg.rfind("--outfile=", 0) == 0) {
      outfile_flag = arg.substr(strlen("--outfile="));
      GlobalOutFile = outfile_flag;
      args.erase(args.begin() + i);
      i--;
      continue;
    }
    if (arg == "--outfile") {
      if (i + 1 < args.size()) {
        outfile_flag = args[i + 1];
        GlobalOutFile = outfile_flag;

        args.erase(args.begin() + i);
        args.erase(args.begin() + i);
        i--;
      }
      continue;
    }
  }

  std::vector<std::string> sourceFiles;
  std::vector<std::string> preprocessedSourceFiles;

  for (size_t i = 1; i < args.size(); ++i) {
    if (isCppSourceFile(args[i]))
      sourceFiles.push_back(args[i]);
  }

  std::vector<std::string> originalSourceDirs;
  for (const auto &sourceFile : sourceFiles) {
    std::error_code pathEc;
    std::filesystem::path absSource =
        std::filesystem::absolute(sourceFile, pathEc);
    if (pathEc)
      continue;

    std::string sourceDir = absSource.parent_path().string();
    if (!sourceDir.empty() &&
        std::find(originalSourceDirs.begin(), originalSourceDirs.end(),
                  sourceDir) == originalSourceDirs.end()) {
      originalSourceDirs.push_back(std::move(sourceDir));
    }
  }

  for (const auto &sourceFile : sourceFiles) {
    std::string ppoutputfile = preprocessor::generate_output_filename();
    preprocessor::process_file(sourceFile, ppoutputfile, bkend_target);
    preprocessedSourceFiles.push_back(ppoutputfile);
    markForDeletion.push_back(ppoutputfile);
  }

  std::vector<std::string> row0, row1, row2;
  row0.push_back(args[0]);

  for (const auto &ppSource : preprocessedSourceFiles)
    row0.push_back(ppSource);

  if (sourceFiles.empty())
    is_link_only = true;

  std::vector<std::string> tool_flags;
  for (size_t i = 1; i < args.size();) {
    if (args[i] == "-h" || args[i] == "--pversion" || args[i] == "--otf") {
      tool_flags.push_back(args[i]);
      args.erase(args.begin() + i);
    } else {
      ++i;
    }
  }

  if (!outfile_flag.empty()) {
    tool_flags.push_back("--outfile");
    tool_flags.push_back(outfile_flag);
  }

  row0.insert(row0.end(), tool_flags.begin(), tool_flags.end());
  row0.push_back("--");
  for (const auto &sourceDir : originalSourceDirs) {
    row0.push_back("-iquote");
    row0.push_back(sourceDir);
  }
  
  if (found) {
    row0.push_back("-x");
    row0.push_back(bkend_target[0]);
    row0.push_back("--offload-arch=" + bkend_target[1]);
  }

  if (!llvmResDirFound) {
    std::string cmd =
        std::string(PARAS_BASE_COMPILER) + " --print-resource-dir";
    const char *cmd_str = cmd.c_str();
    FILE *pipe = popen(cmd_str, "r");
    if (!pipe) {
      llvm::errs() << "WARNING: Unable to get the LLVM Resource Directory\n";
    }
    std::string result;
    char chunk[128];
    while (fgets(chunk, sizeof(chunk), pipe)) {
      result += chunk;
    }
    int status = pclose(pipe);
    if (status == -1) {
      llvm::errs() << "WARNING: Unable to get the LLVM Resource Directory\n";
    }
    if (!result.empty() && result.back() == '\n')
      result.pop_back();
    row0.push_back("-resource-dir");
    row0.push_back(result);
    bkend_target[2] = result;
  }

  row0.push_back("-I");
  row0.push_back(parasIncludePath);

  for (int i = 1; i < args.size(); i++) {
    const std::string &current = args[i];
    if (isCppSourceFile(current)) {
      have_cpp = true;
      continue;
    }
    if (endsWith(current, ".o")) {
      have_ofile = true;
    }

    row0.push_back(current);
    row1.push_back(current);
    row2.push_back(current);
  }

  if (!have_cpp && have_ofile) {
    is_link_only = true;
  }

  finalCompilerArgs.assign(args.begin() + 1, args.end());
  finalCompilerArgs.push_back("-I");
  finalCompilerArgs.push_back(parasIncludePath);

  for (const auto &sourceDir : originalSourceDirs) {
    finalCompilerArgs.push_back("-iquote");
    finalCompilerArgs.push_back(sourceDir);
  }

  row2.push_back("-I");
  row2.push_back(parasIncludePath);

  for (const auto &sourceDir : originalSourceDirs) {
    row2.push_back("-iquote");
    row2.push_back(sourceDir);
  }

  if (is_link_only)
    bkend_target[3] = "is_link_only";

  result[0] = row0;
  result[1] = row1;
  result[2] = row2;
  result[3] = bkend_target;

  return result;
}

int main(int argc, const char **argv) {
  std::vector<std::vector<std::string>> compilerFlags =
      parseCommandLineArgs(argc, argv);
  if (is_link_only) {
    compilerFlags[1].push_back("-L");
    std::string libpath = PARAS_INSTALL_PREFIX + std::string("/lib/");
    compilerFlags[1].push_back(libpath);
    compilerFlags[1].push_back("-lDeviceDiscoveryModule");
    executor::executor(compilerFlags[1], compilerFlags[3]);
    return 0;
  }

  const char **fakeargs = new const char *[compilerFlags[0].size() + 1];
  for (size_t i = 0; i < compilerFlags[0].size(); i++)
    fakeargs[i] = compilerFlags[0][i].c_str();
  fakeargs[compilerFlags[0].size()] = nullptr;
  int fakeargc = compilerFlags[0].size();

  auto ExpectedParser = clang::tooling::CommonOptionsParser::create(
      fakeargc, fakeargs, Tooling, llvm::cl::ZeroOrMore);
  if (!ExpectedParser) {
    llvm::outs() << llvm::toString(ExpectedParser.takeError()) << "\n";
    return -1;
  }

  if (VersionOption) {
    llvm::outs() << "Paras Driver version 1.0.0 (LLVM/Clang v"
                 << LLVM_VERSION_STRING << ")\n";
    return 0;
  }

  clang::tooling::CommonOptionsParser &options = ExpectedParser.get();
  clang::tooling::ClangTool tool(options.getCompilations(),
                                 options.getSourcePathList());

  int transformStatus = tool.run(std::make_unique<ParasFrontendActionFactory>(
                                     compilerFlags[2], compilerFlags[3])
                                     .get());
  if (transformStatus != 0)
    return transformStatus;

  if (OnlyTransform || !WriteToFile.empty())
    return 0;

  size_t transformedIndex = 0;
  std::vector<std::string> finalCommand;
  finalCommand.reserve(finalCompilerArgs.size());

  for (const auto &arg : finalCompilerArgs) {
    if (isCppSourceFile(arg)) {
      if (transformedIndex >= transformedSourceFiles.size()) {
        llvm::errs()
            << "parascc: internal error: missing transformed source for " << arg
            << "\n";
        return 1;
      }
      finalCommand.push_back(transformedSourceFiles[transformedIndex++]);
    } else {
      finalCommand.push_back(arg);
    }
  }

  if (transformedIndex != transformedSourceFiles.size()) {
    llvm::errs()
        << "parascc: internal error: transformed source count mismatch\n";
    return 1;
  }

  executor::executor(finalCommand, compilerFlags[3]);
  return 0;
}
