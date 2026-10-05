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

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <paras/ParaSDeviceCallGraph.hpp>
#include <paras/ParaSLog.hpp>
#include <paras/ParaSLambdaHandler.hpp>

namespace {

const clang::Expr* stripKernelExpr(const clang::Expr* expr) {
    if (!expr)
        return nullptr;

    expr = expr->IgnoreParenImpCasts();

    bool changed = true;
    while (changed && expr) {
        changed = false;

        if (const auto* cleanups = llvm::dyn_cast<clang::ExprWithCleanups>(expr)) {
            expr = cleanups->getSubExpr()->IgnoreParenImpCasts();
            changed = true;
        } else if (const auto* materialized =
                       llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
            expr = materialized->getSubExpr()->IgnoreParenImpCasts();
            changed = true;
        } else if (const auto* bound = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
            expr = bound->getSubExpr()->IgnoreParenImpCasts();
            changed = true;
        }
    }

    return expr;
}

const clang::CXXMethodDecl* lambdaCallOperatorFromExpr(const clang::Expr* expr) {
    expr = stripKernelExpr(expr);
    if (!expr)
        return nullptr;

    clang::QualType type = expr->getType();
    if (type->isReferenceType())
        type = type->getPointeeType();

    const clang::CXXRecordDecl* closure = type->getAsCXXRecordDecl();
    if (!closure || !closure->isLambda())
        return nullptr;

    return closure->getLambdaCallOperator();
}

class ReturnedLambdaFinder : public clang::RecursiveASTVisitor<ReturnedLambdaFinder> {
public:
    const clang::CXXMethodDecl* callOperator = nullptr;

    bool VisitReturnStmt(clang::ReturnStmt* stmt) {
        if (callOperator || !stmt)
            return !callOperator;

        const clang::Expr* value = stripKernelExpr(stmt->getRetValue());
        const auto* lambda = llvm::dyn_cast_or_null<clang::LambdaExpr>(value);
        if (!lambda)
            return true;

        callOperator = lambda->getCallOperator();
        return false;
    }
};

const clang::CXXMethodDecl* returnedLambdaCallOperator(const clang::FunctionDecl* function) {
    if (!function)
        return nullptr;

    const clang::FunctionDecl* definition = nullptr;
    if (!function->hasBody(definition) || !definition || !definition->getBody())
        return nullptr;

    ReturnedLambdaFinder finder;
    finder.TraverseStmt(const_cast<clang::Stmt*>(definition->getBody()));
    return finder.callOperator;
}

const clang::CXXMethodDecl*
resolveKernelCallOperator(const clang::Expr* expr, llvm::DenseSet<const clang::Expr*>& seenExprs,
                          llvm::DenseSet<const clang::FunctionDecl*>& seenFunctions) {
    expr = stripKernelExpr(expr);
    if (!expr || !seenExprs.insert(expr).second)
        return nullptr;

    if (const clang::CXXMethodDecl* op = lambdaCallOperatorFromExpr(expr))
        return op;

    if (const auto* ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        if (const auto* var = llvm::dyn_cast<clang::VarDecl>(ref->getDecl())) {
            if (var->hasInit()) {
                if (const clang::CXXMethodDecl* op =
                        resolveKernelCallOperator(var->getInit(), seenExprs, seenFunctions))
                    return op;
            }
        }
    }

    if (const auto* call = llvm::dyn_cast<clang::CallExpr>(expr)) {
        for (const clang::Expr* arg : call->arguments()) {
            if (const clang::CXXMethodDecl* op =
                    resolveKernelCallOperator(arg, seenExprs, seenFunctions))
                return op;
        }

        if (const clang::FunctionDecl* callee = call->getDirectCallee()) {
            callee = callee->getCanonicalDecl();
            if (seenFunctions.insert(callee).second) {
                if (const clang::CXXMethodDecl* op = returnedLambdaCallOperator(callee))
                    return op;
            }
        }
    }

    if (const auto* ctor = llvm::dyn_cast<clang::CXXConstructExpr>(expr)) {
        for (const clang::Expr* arg : ctor->arguments()) {
            if (const clang::CXXMethodDecl* op =
                    resolveKernelCallOperator(arg, seenExprs, seenFunctions))
                return op;
        }
    }

    return nullptr;
}

} // namespace

void ParaSLambdaHandler::run(const clang::ast_matchers::MatchFinder::MatchResult& result) {
    if (const clang::LambdaExpr* Lambda = result.Nodes.getNodeAs<clang::LambdaExpr>("lambda")) {
        parasLog() << "[ParaS] lambda at: ";
        Lambda->getBeginLoc().print(parasLog(), *result.SourceManager);
        parasLog() << "\n";
    }

    if (backend_target.empty() || (backend_target[0] != "cuda" && backend_target[0] != "hip"))
        return;

    const clang::Expr* kernelExpr = result.Nodes.getNodeAs<clang::Expr>("paras-kernel-callable");
    if (!kernelExpr)
        return;

    llvm::DenseSet<const clang::Expr*> seenExprs;
    llvm::DenseSet<const clang::FunctionDecl*> seenFunctions;
    const clang::CXXMethodDecl* callOperator =
        resolveKernelCallOperator(kernelExpr, seenExprs, seenFunctions);

    if (!callOperator)
        return;

    const clang::CXXMethodDecl* kernelKey = callOperator->getCanonicalDecl();
    if (!processedKernels.insert(kernelKey).second)
        return;

    parasLog() << "[ParaS] Found SYCL kernel callable\n";
    parasLog() << "[ParaS] Kernel lambda call operator: "
               << callOperator->getQualifiedNameAsString() << "\n";

    ParaSDeviceCallGraph deviceGraph(rewriter, headerRewriter);
    deviceGraph.addKernel(callOperator);
    deviceGraph.run(*result.Context);
}
