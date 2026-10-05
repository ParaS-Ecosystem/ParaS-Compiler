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

#include "clang/AST/Attr.h"
#include "clang/AST/AttrIterator.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include <chrono>
#include <clang/AST/AST.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <cstdlib>
#include <fstream>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/ADT/SmallString.h>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <sys/stat.h>

#include "paras/ParaSConsumer.hpp"
#include "paras/Executor.hpp"
#include "paras/Preprocessor.hpp"
#include "paras/ParaSLog.hpp"
#include "paras/ParaSTempDir.hpp"

#include "pathfinder.hpp"

bool is_link_only = false;
std::string GlobalOutFile = "";
std::vector<std::string> markForDeletion;

static void cleanupTempFiles() {
    if (std::getenv("PARAS_KEEP_TEMPS"))
        return;
    for (const auto& f : markForDeletion)
        ::unlink(f.c_str());
    markForDeletion.clear();
}

struct ParaSRewrittenHeader {
    std::string originalPath;
    std::string rewrittenPath;
};

std::vector<ParaSRewrittenHeader> rewrittenHeaders;

static std::string yamlQuote(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('\'');
    for (char c : value) {
        if (c == '\'')
            result += "''";
        else
            result.push_back(c);
    }
    result.push_back('\'');
    return result;
}

static std::string createVFSOverlay() {
    if (rewrittenHeaders.empty())
        return {};

    std::string model = parasTempDir() + "/paras_vfs_%%%%%%%%.yaml";
    llvm::SmallString<128> overlayPath;
    std::error_code EC = llvm::sys::fs::createUniqueFile(model, overlayPath);
    if (EC) {
        llvm::errs() << "parascc: cannot create VFS overlay: " << EC.message() << "\n";
        return {};
    }

    llvm::raw_fd_ostream out(overlayPath, EC);
    if (EC) {
        llvm::errs() << "parascc: cannot write VFS overlay: " << EC.message() << "\n";
        return {};
    }

    out << "{\n";
    out << "  'version': 0,\n";
    out << "  'case-sensitive': 'true',\n";
    out << "  'roots': [\n";

    for (size_t i = 0; i < rewrittenHeaders.size(); ++i) {
        const auto& H = rewrittenHeaders[i];
        out << "    { 'type': 'file', 'name': " << yamlQuote(H.originalPath)
            << ", 'external-contents': " << yamlQuote(H.rewrittenPath) << " }";
        if (i + 1 != rewrittenHeaders.size())
            out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
    out.close();

    std::string result(overlayPath.str());
    markForDeletion.push_back(result);
    return result;
}

static void writeRewrittenHeaders(clang::Rewriter& rewriter, clang::SourceManager& SM,
                                  clang::FileID mainID) {
    for (auto it = rewriter.buffer_begin(); it != rewriter.buffer_end(); ++it) {
        clang::FileID FID = it->first;
        if (FID == mainID)
            continue;

        auto fileRef = SM.getFileEntryRefForID(FID);
        if (!fileRef)
            continue;

        std::string originalPath = fileRef->getName().str();

        std::error_code parasPathEC;
        std::filesystem::path parasInclude = std::filesystem::absolute(
            std::filesystem::path(PARAS_INSTALL_PREFIX) / "include", parasPathEC);
        std::error_code headerPathEC;
        std::filesystem::path headerPath = std::filesystem::absolute(originalPath, headerPathEC);

        if (!parasPathEC && !headerPathEC) {
            const std::string parasIncludeStr = parasInclude.lexically_normal().string();
            const std::string headerPathStr = headerPath.lexically_normal().string();
            const std::string prefix = parasIncludeStr + "/";

            if (headerPathStr == parasIncludeStr ||
                headerPathStr.compare(0, prefix.size(), prefix) == 0) {
                continue;
            }
        }

        std::error_code pathEC;
        std::filesystem::path absolutePath = std::filesystem::absolute(originalPath, pathEC);
        if (!pathEC)
            originalPath = absolutePath.lexically_normal().string();

        std::string headerModel = parasTempDir() + "/paras_hdr_%%%%%%%%.hpp";
        llvm::SmallString<128> rewrittenPathSV;
        std::error_code EC = llvm::sys::fs::createUniqueFile(headerModel, rewrittenPathSV);
        if (EC) {
            llvm::errs() << "parascc: cannot create rewritten header for " << originalPath << ": "
                         << EC.message() << "\n";
            continue;
        }

        std::string rewrittenContents;
        llvm::raw_string_ostream rewrittenStream(rewrittenContents);
        it->second.write(rewrittenStream);
        rewrittenStream.flush();

        llvm::raw_fd_ostream rewrittenFile(rewrittenPathSV, EC);
        if (EC) {
            llvm::errs() << "parascc: cannot write rewritten header for " << originalPath << ": "
                         << EC.message() << "\n";
            continue;
        }
        rewrittenFile << rewrittenContents;
        rewrittenFile.close();

        std::string rewrittenPath(rewrittenPathSV.str());

        auto existing = std::find_if(
            rewrittenHeaders.begin(), rewrittenHeaders.end(),
            [&](const ParaSRewrittenHeader& H) { return H.originalPath == originalPath; });
        if (existing != rewrittenHeaders.end()) {
            existing->rewrittenPath = rewrittenPath;
        } else {
            rewrittenHeaders.push_back({originalPath, rewrittenPath});
        }

        markForDeletion.push_back(rewrittenPath);
        parasLog() << "[ParaS] rewrote header: " << originalPath << " -> " << rewrittenPath << "\n";
    }
}

static llvm::cl::OptionCategory Tooling("paras-expfinder");

static llvm::cl::opt<bool> VersionOption("pversion", llvm::cl::desc("Display version information"),
                                         llvm::cl::init(false), llvm::cl::cat(Tooling));

static llvm::cl::opt<bool> OnlyTransform("otf", llvm::cl::desc("Only transform the code"),
                                         llvm::cl::init(false), llvm::cl::cat(Tooling));

static llvm::cl::opt<std::string> WriteToFile("outfile", llvm::cl::desc("Specify an output file"),
                                              llvm::cl::value_desc("filename"),
                                              llvm::cl::cat(Tooling));

class ParasFrontendAction : public clang::ASTFrontendAction {
private:
    clang::Rewriter rewriter;
    clang::Rewriter headerRewriter;
    std::vector<std::string> compilerFlags;
    std::vector<std::string> bkend_target;

public:
    ParasFrontendAction() {}

    ParasFrontendAction(const std::vector<std::string>& cf, std::vector<std::string>& bkend_target)
        : compilerFlags(cf), bkend_target(bkend_target) {}

    void ExecuteAction() override { clang::ASTFrontendAction::ExecuteAction(); }

    void EndSourceFileAction() override {
        auto& SM = rewriter.getSourceMgr();
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
            llvm::outs() << "\n=========== Transformed code =========== \n " << updatedFileContents
                         << "\n";
            return;
        }

        std::string tmpModel = parasTempDir() + "/paras_tmp_%%%%%%%%.cpp";
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

        std::vector<std::string> command = compilerFlags;
        command.push_back(tempFile);

        writeRewrittenHeaders(headerRewriter, SM, mainID);
        std::string overlayPath = createVFSOverlay();
        if (!overlayPath.empty()) {
            command.push_back("-ivfsoverlay");
            command.push_back(overlayPath);
            parasLog() << "[ParaS] using rewritten-header VFS overlay: " << overlayPath << "\n";
        }

        markForDeletion.push_back(tempFile);
        int rc = executor::executor(command, bkend_target);
        cleanupTempFiles();
        std::exit(rc);
    }

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& CI,
                                                          llvm::StringRef infile) override {
        rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        headerRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<ParaSConsumer>(rewriter, headerRewriter, bkend_target);
    }
};

class ParasFrontendActionFactory : public clang::tooling::FrontendActionFactory {
private:
    std::vector<std::string> flags;
    std::vector<std::string> bkend_target;

public:
    ParasFrontendActionFactory(const std::vector<std::string>& F,
                               const std::vector<std::string>& bkend_target)
        : flags(F), bkend_target(bkend_target) {}

    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<ParasFrontendAction>(flags, bkend_target);
    }
};

std::vector<std::vector<std::string>> parseCommandLineArgs(int argc, const char** argv) {
    std::vector<std::vector<std::string>> result(4);
    std::vector<std::string> args;

    std::string parasLibPath = PARAS_INSTALL_PREFIX + std::string("/lib/");
    std::string parasIncludePath = PARAS_INSTALL_PREFIX + std::string("/include/");

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
                    std::cerr << "Error: backend must be 'cuda' or 'hip', got '" << backend
                              << "'\n";
                    exit(1);
                }
                if (arch.empty()) {
                    std::cerr << "Error: architecture part cannot be empty\n";
                    exit(1);
                }
                args.push_back(backend == "hip" ? "-DPARASDEVICE=2" : "-DPARASDEVICE=1");
                args.push_back("-DPARAS_OFFLOAD_ARCH=\"" + arch + "\"");
                if (backend == "hip") {
                    bool userSet = false, wave64 = false;
                    for (const std::string& a : args) {
                        if (a.rfind("-DPARAS_AMD_WAVEFRONT_SIZE", 0) == 0)
                            userSet = true;
                        if (a == "-mwavefrontsize64")
                            wave64 = true;
                        if (a == "-mno-wavefrontsize64")
                            wave64 = false;
                    }
                    int wave = 0;
                    std::stringstream archList(arch);
                    for (std::string one; std::getline(archList, one, ',');) {
                        if (one.rfind("gfx", 0) != 0) {
                            wave = -1;
                            break;
                        }
                        std::string id =
                            one.substr(3, one.find(':') == std::string::npos ? std::string::npos
                                                                             : one.find(':') - 3);
                        size_t dash = id.find('-');
                        std::string d = dash != std::string::npos
                                            ? id.substr(0, dash)
                                            : id.substr(0, id.size() >= 4 ? 2 : 1);
                        if (d.empty() || !std::all_of(d.begin(), d.end(),
                                                      [](unsigned char c) { return isdigit(c); })) {
                            wave = -1;
                            break;
                        }
                        int major = std::stoi(d);
                        int w = (major >= 10 && !wave64) ? 32 : 64;
                        if (wave != 0 && wave != w) {
                            wave = -1;
                            break;
                        }
                        wave = w;
                    }
                    if (!userSet && wave > 0)
                        args.push_back("-DPARAS_AMD_WAVEFRONT_SIZE=" + std::to_string(wave));
                }
                bkend_target[0] = backend;
                bkend_target[1] = arch;
                args.erase(args.begin() + i, args.begin() + i + 2);
                i--;
            }
        }
    }

    auto endsWith = [](const std::string& str, const std::string& suffix) -> bool {
        if (str.length() < suffix.length())
            return false;
        return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
    };

    std::string outfile_flag;

    for (int i = 0; i < args.size(); i++) {
        const std::string& arg = args[i];
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

    std::string sourceFile;
    std::string origSourceDir;
    std::string origSourceName;
    auto it = std::find_if(args.begin() + 1, args.end(), [&](const std::string& arg) {
        return endsWith(arg, ".cpp") || endsWith(arg, ".cc") || endsWith(arg, ".cxx");
    });

    if (it != args.end()) {
        sourceFile = *it;
        int nsrc = std::count_if(args.begin() + 1, args.end(), [&](const std::string& a) {
            return endsWith(a, ".cpp") || endsWith(a, ".cc") || endsWith(a, ".cxx");
        });
        if (nsrc > 1) {
            std::cerr
                << "parascc: error: " << nsrc
                << " source files given; parascc handles one "
                   "source per invocation. Use 'parascc -c' per file, then link the .o files.\n";
            exit(1);
        }
    }
    origSourceName = sourceFile;
    if (!sourceFile.empty()) {
        llvm::SmallString<256> absSource(sourceFile);
        llvm::sys::fs::make_absolute(absSource);
        origSourceDir = std::string(llvm::sys::path::parent_path(absSource));

        std::string ppoutputfile = preprocessor::generate_output_filename();
        preprocessor::process_file(sourceFile, ppoutputfile, bkend_target);
        sourceFile = ppoutputfile;
        markForDeletion.push_back(sourceFile);
    }

    std::vector<std::string> row0, row1, row2;
    row0.push_back(args[0]);

    if (!sourceFile.empty())
        row0.push_back(sourceFile);
    else
        is_link_only = true;

    std::vector<std::string> tool_flags;
    for (std::string itr : args) {
        if (itr == "-h" || itr == "--pversion" || itr == "--otf") {
            tool_flags.push_back(itr);
            args.erase(std::remove(args.begin(), args.end(), itr), args.end());
        }
    }
    if (!outfile_flag.empty()) {
        tool_flags.push_back("--outfile");
        tool_flags.push_back(outfile_flag);
    }

    row0.insert(row0.end(), tool_flags.begin(), tool_flags.end());

    row0.push_back("--");
    if (found) {
        row0.push_back("-x");
        row0.push_back(bkend_target[0]);
        row0.push_back("--offload-arch=" + bkend_target[1]);
    }

    if (!llvmResDirFound) {
        std::string cmd = std::string(PARAS_BASE_COMPILER) + " --print-resource-dir";
        const char* cmd_str = cmd.c_str();
        FILE* pipe = popen(cmd_str, "r");
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

    if (!origSourceDir.empty()) {
        row0.push_back("-I");
        row0.push_back(origSourceDir);
    }

    for (int i = 1; i < args.size(); i++) {
        const std::string& current = args[i];
        if (current == sourceFile || endsWith(current, ".cpp") || endsWith(current, ".cc") ||
            endsWith(current, ".cxx")) {
            have_cpp = true;
            continue;
        }
        if (endsWith(current, ".o")) {
            have_ofile = true;
        }

        row1.push_back(current);
        row2.push_back(current);
        if (current.rfind("-L", 0) == 0 || current.rfind("-Wl,", 0) == 0 ||
            (current.rfind("-l", 0) == 0 && current.size() > 2)) {
            if ((current == "-L") && i + 1 < args.size()) {
                ++i;
                row1.push_back(args[i]);
                row2.push_back(args[i]);
            }
            continue;
        }
        if (current == "-Xlinker" && i + 1 < args.size()) {
            ++i;
            row1.push_back(args[i]);
            row2.push_back(args[i]);
            continue;
        }
        row0.push_back(current);
    }

    if (!have_cpp && have_ofile) {
        is_link_only = true;
    }

    row2.push_back("-I");
    row2.push_back(parasIncludePath);

    if (!origSourceDir.empty()) {
        row2.push_back("-I");
        row2.push_back(origSourceDir);
    }

    if (is_link_only)
        bkend_target[3] = "is_link_only";

    {
        bool has_c = std::find(row2.begin(), row2.end(), "-c") != row2.end();
        bool has_o = std::find(row2.begin(), row2.end(), "-o") != row2.end();
        if (has_c && !has_o && !origSourceDir.empty()) {
            std::string stem = std::string(llvm::sys::path::stem(origSourceName));
            row2.push_back("-o");
            row2.push_back(stem + ".o");
        }
    }

    result[0] = row0;
    result[1] = row1;
    result[2] = row2;
    result[3] = bkend_target;

    return result;
}

int main(int argc, const char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pversion" || a == "-pversion") {
            llvm::outs() << "Paras Driver version 1.0.0 (LLVM/Clang v" << LLVM_VERSION_STRING
                         << ")\n";
            return 0;
        }
    }
    std::vector<std::vector<std::string>> compilerFlags = parseCommandLineArgs(argc, argv);
    if (is_link_only) {
        compilerFlags[1].push_back("-L");
        std::string libpath = PARAS_INSTALL_PREFIX + std::string("/lib/");
        compilerFlags[1].push_back(libpath);
        compilerFlags[1].push_back("-lDeviceDiscoveryModule");
        return executor::executor(compilerFlags[1], compilerFlags[3]);
    }

    const char** fakeargs = new const char*[compilerFlags[0].size() + 1];
    for (size_t i = 0; i < compilerFlags[0].size(); i++)
        fakeargs[i] = compilerFlags[0][i].c_str();
    fakeargs[compilerFlags[0].size()] = nullptr;
    int fakeargc = compilerFlags[0].size();

    auto ExpectedParser = clang::tooling::CommonOptionsParser::create(fakeargc, fakeargs, Tooling,
                                                                      llvm::cl::ZeroOrMore);
    if (!ExpectedParser) {
        llvm::outs() << llvm::toString(ExpectedParser.takeError()) << "\n";
        return -1;
    }

    if (VersionOption) {
        llvm::outs() << "Paras Driver version 1.0.0 (LLVM/Clang v" << LLVM_VERSION_STRING << ")\n";
        return 0;
    }

    clang::tooling::CommonOptionsParser& options = ExpectedParser.get();
    clang::tooling::ClangTool tool(options.getCompilations(), options.getSourcePathList());

    int toolRc = tool.run(
        std::make_unique<ParasFrontendActionFactory>(compilerFlags[2], compilerFlags[3]).get());
    cleanupTempFiles();
    return toolRc;
}
