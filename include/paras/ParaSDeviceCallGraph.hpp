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

#ifndef __PARAS_DEVICE_CALL_GRAPH_HPP__
#define __PARAS_DEVICE_CALL_GRAPH_HPP__

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/SourceManager.h"

class ParaSDeviceCallGraph {
private:
    clang::Rewriter& rewriter;
    clang::Rewriter& headerRewriter;

    clang::Rewriter& rewriterFor(clang::SourceLocation loc, const clang::SourceManager& SM) {
        return SM.isInMainFile(loc) ? rewriter : headerRewriter;
    }
    llvm::DenseSet<const clang::FunctionDecl*> visited;
    llvm::DenseSet<const clang::FunctionDecl*> annotated;
    llvm::SmallVector<const clang::FunctionDecl*, 64> workList;

    const clang::FunctionDecl* normalize(const clang::FunctionDecl* FD) const;
    const clang::FunctionDecl* annotationIdentity(const clang::FunctionDecl* FD) const;
    bool isRewritable(const clang::FunctionDecl* FD, const clang::SourceManager& SM) const;
    void annotate(const clang::FunctionDecl* FD, clang::ASTContext& context);
    void process(const clang::FunctionDecl* FD, clang::ASTContext& context);

public:
    ParaSDeviceCallGraph(clang::Rewriter& mainRw, clang::Rewriter& headerRw)
        : rewriter(mainRw), headerRewriter(headerRw) {}

    void addKernel(const clang::CXXMethodDecl* callOperator);
    void enqueue(const clang::FunctionDecl* FD);
    void run(clang::ASTContext& context);
};

#endif
