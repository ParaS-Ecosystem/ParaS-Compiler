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

#include "paras/ParaSDeviceFunctionMarker.hpp"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/SourceManager.h>

#include <set>
#include <string>
#include <vector>

namespace {

bool isSyclKernelLauncher(const clang::CXXMethodDecl* MD) {
    if (!MD || !MD->getIdentifier())
        return false;
    const llvm::StringRef name = MD->getName();
    if (name != "parallel_for" && name != "single_task" && name != "parallel_for_work_group")
        return false;
    const std::string cls = MD->getParent()->getQualifiedNameAsString();
    return llvm::StringRef(cls).starts_with("sycl::");
}

class KernelRootFinder : public clang::RecursiveASTVisitor<KernelRootFinder> {
public:
    std::vector<const clang::FunctionDecl*> roots;

    bool shouldVisitTemplateInstantiations() const { return true; }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* CE) {
        if (!isSyclKernelLauncher(CE->getMethodDecl()) || CE->getNumArgs() == 0)
            return true;
        addKernelObject(CE->getArg(CE->getNumArgs() - 1));
        return true;
    }

private:
    void addKernelObject(const clang::Expr* E) {
        E = E->IgnoreImplicit()->IgnoreParens();
        if (const auto* LE = llvm::dyn_cast<clang::LambdaExpr>(E)) {
            roots.push_back(LE->getCallOperator());
            return;
        }
        const clang::CXXRecordDecl* RD = E->getType()->getAsCXXRecordDecl();
        if (!RD || !RD->hasDefinition())
            return;
        RD = RD->getDefinition();
        for (const clang::CXXMethodDecl* M : RD->methods())
            if (M->getOverloadedOperator() == clang::OO_Call)
                roots.push_back(M);
        for (const clang::Decl* D : RD->decls())
            if (const auto* FTD = llvm::dyn_cast<clang::FunctionTemplateDecl>(D))
                for (const clang::FunctionDecl* S : FTD->specializations())
                    if (S->getOverloadedOperator() == clang::OO_Call)
                        roots.push_back(S);
    }
};

class CalleeCollector : public clang::RecursiveASTVisitor<CalleeCollector> {
public:
    std::vector<const clang::FunctionDecl*> callees;

    bool VisitCallExpr(clang::CallExpr* CE) {
        if (const clang::FunctionDecl* FD = CE->getDirectCallee())
            callees.push_back(FD);
        return true;
    }
    bool VisitCXXConstructExpr(clang::CXXConstructExpr* CE) {
        if (const clang::CXXConstructorDecl* C = CE->getConstructor())
            callees.push_back(C);
        return true;
    }
    bool VisitLambdaExpr(clang::LambdaExpr* LE) {
        callees.push_back(LE->getCallOperator());
        return true;
    }
};

bool hasCudaTargetAttr(const clang::FunctionDecl* FD) {
    return FD->hasAttr<clang::CUDAHostAttr>() || FD->hasAttr<clang::CUDADeviceAttr>() ||
           FD->hasAttr<clang::CUDAGlobalAttr>();
}

bool isLambdaCallOperator(const clang::FunctionDecl* FD) {
    const auto* MD = llvm::dyn_cast<clang::CXXMethodDecl>(FD);
    return MD && MD->getParent()->isLambda();
}

} // namespace

void markKernelReachableFunctions(clang::ASTContext& ctx, clang::Rewriter& rw) {
    clang::SourceManager& SM = ctx.getSourceManager();

    KernelRootFinder finder;
    finder.TraverseDecl(ctx.getTranslationUnitDecl());

    std::set<const clang::FunctionDecl*> visited;
    std::set<unsigned> inserted;
    std::vector<const clang::FunctionDecl*> work(finder.roots.begin(), finder.roots.end());

    while (!work.empty()) {
        const clang::FunctionDecl* FD = work.back();
        work.pop_back();
        if (!FD || !visited.insert(FD->getCanonicalDecl()).second)
            continue;

        if (const clang::FunctionDecl* def = FD->getDefinition())
            FD = def;
        const clang::FunctionDecl* pattern = FD->getTemplateInstantiationPattern();
        const clang::FunctionDecl* srcDecl = pattern ? pattern : FD;
        const clang::SourceLocation srcLoc = SM.getExpansionLoc(srcDecl->getLocation());
        if (!SM.isInMainFile(srcLoc))
            continue;

        if (!isLambdaCallOperator(srcDecl) && !srcDecl->isConstexpr() &&
            !hasCudaTargetAttr(srcDecl) && !srcDecl->isImplicit()) {
            for (const clang::FunctionDecl* R : srcDecl->redecls()) {
                clang::SourceLocation L = R->getInnerLocStart();
                if (L.isMacroID() || !SM.isInMainFile(L))
                    continue;
                if (inserted.insert(L.getRawEncoding()).second)
                    rw.InsertTextBefore(L, "__host__ __device__ ");
            }
        }

        const clang::FunctionDecl* withBody = nullptr;
        if (FD->hasBody(withBody) && withBody && withBody->getBody()) {
            CalleeCollector cc;
            cc.TraverseStmt(withBody->getBody());
            work.insert(work.end(), cc.callees.begin(), cc.callees.end());
        }
    }
}
