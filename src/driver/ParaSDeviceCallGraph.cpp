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

#include "paras/ParaSVerbose.hpp"
#include "paras/ParaSDeviceCallGraph.hpp"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

const clang::FunctionDecl *
ParaSDeviceCallGraph::normalize(const clang::FunctionDecl *FD) const {
  if (!FD)
    return nullptr;

  return FD->getCanonicalDecl();
}

const clang::FunctionDecl *
ParaSDeviceCallGraph::annotationIdentity(const clang::FunctionDecl *FD) const {
  if (!FD)
    return nullptr;

  if (const clang::FunctionDecl *pattern =
          FD->getTemplateInstantiationPattern())
    return pattern->getCanonicalDecl();

  return FD->getCanonicalDecl();
}

bool ParaSDeviceCallGraph::isRewritable(const clang::FunctionDecl *FD,
                                        const clang::SourceManager &SM) const {
  if (!FD)
    return false;

  clang::SourceLocation loc = SM.getSpellingLoc(FD->getBeginLoc());
  if (loc.isInvalid())
    return false;

  if (SM.isInSystemHeader(loc) || SM.isInExternCSystemHeader(loc))
    return false;

  if (FD->hasAttr<clang::CUDAGlobalAttr>())
    return false;

  return true;
}

void ParaSDeviceCallGraph::enqueue(const clang::FunctionDecl *FD) {
  FD = normalize(FD);
  if (!FD)
    return;

  if (visited.insert(FD).second)
    workList.push_back(FD);
}

void ParaSDeviceCallGraph::enqueueDestructorOf(clang::QualType type) {
  if (type.isNull())
    return;
  const clang::Type *t = type->getBaseElementTypeUnsafe();
  if (!t)
    return;
  if (const clang::CXXRecordDecl *rd = t->getAsCXXRecordDecl())
    if (rd->hasDefinition())
      if (const clang::CXXDestructorDecl *dtor = rd->getDestructor())
        enqueue(dtor);
}

void ParaSDeviceCallGraph::addKernel(const clang::CXXMethodDecl *callOperator) {
  enqueue(callOperator);
  enqueueParameterClasses(callOperator);
}

void ParaSDeviceCallGraph::enqueueRecordSpecialMembers(
    const clang::CXXRecordDecl *record) {
  if (!record || !record->hasDefinition() || record->isLambda())
    return;
  record = record->getDefinition();
  for (const clang::CXXConstructorDecl *ctor : record->ctors())
    if (ctor && !ctor->isImplicit())
      enqueue(ctor);

  for (const clang::Decl *decl : record->decls())
    if (const auto *tmpl = llvm::dyn_cast<clang::FunctionTemplateDecl>(decl))
      if (llvm::isa<clang::CXXConstructorDecl>(tmpl->getTemplatedDecl()))
        for (const clang::FunctionDecl *spec : tmpl->specializations())
          enqueue(spec);
  if (const clang::CXXDestructorDecl *dtor = record->getDestructor())
    if (!dtor->isImplicit())
      enqueue(dtor);
}

void ParaSDeviceCallGraph::enqueueParameterClasses(
    const clang::FunctionDecl *FD) {
  if (!FD)
    return;
  for (const clang::ParmVarDecl *param : FD->parameters()) {
    if (!param)
      continue;
    const clang::QualType type =
        param->getType().getNonReferenceType().getUnqualifiedType();
    enqueueRecordSpecialMembers(type->getAsCXXRecordDecl());
  }
}

void ParaSDeviceCallGraph::addFunctor(const clang::CXXRecordDecl *record) {
  if (!record || !record->hasDefinition() || record->isLambda())
    return;
  record = record->getDefinition();

  auto isCallOperator = [](const clang::FunctionDecl *FD) {
    return FD && FD->getOverloadedOperator() == clang::OO_Call;
  };

  for (const clang::CXXMethodDecl *method : record->methods()) {
    if (isCallOperator(method)) {
      enqueue(method);
      enqueueParameterClasses(method);
    }
  }

  for (const clang::Decl *decl : record->decls()) {
    const auto *tmpl = llvm::dyn_cast<clang::FunctionTemplateDecl>(decl);
    if (!tmpl || !isCallOperator(tmpl->getTemplatedDecl()))
      continue;
    for (const clang::FunctionDecl *spec : tmpl->specializations()) {
      enqueue(spec);
      enqueueParameterClasses(spec);
    }
  }

  enqueueRecordSpecialMembers(record);
}

void ParaSDeviceCallGraph::annotate(const clang::FunctionDecl *FD,
                                    clang::ASTContext &context) {
  const clang::FunctionDecl *definition = nullptr;
  if (!FD || !FD->hasBody(definition) || !definition)
    return;

  clang::SourceManager &SM = context.getSourceManager();
  if (!isRewritable(definition, SM))
    return;

  if (definition->hasAttr<clang::CUDADeviceAttr>() ||
      definition->hasAttr<clang::CUDAGlobalAttr>())
    return;

  if (definition->isImplicit())
    return;

  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(definition)) {
    if (method->getParent() && method->getParent()->isLambda())
      return;
  }

  const clang::FunctionDecl *annotationKey = annotationIdentity(definition);
  if (!annotationKey || !annotated.insert(annotationKey).second)
    return;

  bool insertedAny = false;
  auto annotateFamily = [&](const clang::FunctionDecl *familyKey) {
  for (const clang::FunctionDecl *redecl : familyKey->redecls()) {
    if (!redecl || !isRewritable(redecl, SM))
      continue;

    if (redecl->hasAttr<clang::CUDADeviceAttr>() ||
        redecl->hasAttr<clang::CUDAGlobalAttr>())
      continue;

    clang::SourceLocation loc = SM.getSpellingLoc(redecl->getInnerLocStart());
    if (loc.isInvalid() || loc.isMacroID())
      continue;

    {
      const auto dec = SM.getDecomposedLoc(loc);
      const llvm::StringRef text = SM.getBufferData(dec.first);
      unsigned off = dec.second;
      while (off + 1 < text.size() && text[off] == '[' && text[off + 1] == '[') {
        int depth = 0;
        unsigned i = off;
        for (; i < text.size(); ++i) {
          if (text[i] == '[')
            ++depth;
          else if (text[i] == ']' && --depth == 0)
            break;
        }
        if (i >= text.size())
          break;
        off = i + 1;
        while (off < text.size() &&
               (text[off] == ' ' || text[off] == '\t' || text[off] == '\n' ||
                text[off] == '\r'))
          ++off;
      }
      loc = SM.getComposedLoc(dec.first, off);
    }

    if (!annotatedAt.insert(loc).second)
      continue;
    {
      clang::SourceLocation nameLoc = SM.getSpellingLoc(redecl->getLocation());
      if (nameLoc.isValid() && !nameLoc.isMacroID() &&
          SM.getFileID(nameLoc) == SM.getFileID(loc) &&
          rewriter.getRewrittenText(
                      clang::CharSourceRange::getTokenRange(loc, nameLoc))
                  .find("__device__") != std::string::npos)
        continue;
    }

    if (rewriter.InsertTextBefore(loc, "__host__ __device__ ")) {
      llvm::errs() << "[ParaS] failed to mark device-callable redeclaration: "
                   << redecl->getQualifiedNameAsString() << "\n";
      continue;
    }

    insertedAny = true;
  }
  };
  annotateFamily(annotationKey);

  const clang::FunctionTemplateDecl *FTD =
      annotationKey->getDescribedFunctionTemplate();
  if (!FTD)
    FTD = annotationKey->getPrimaryTemplate();
  if (FTD) {
    if (const clang::FunctionDecl *primary = FTD->getTemplatedDecl()) {
      const clang::FunctionDecl *key = primary->getCanonicalDecl();
      if (isRewritable(key, SM) && !key->hasAttr<clang::CUDADeviceAttr>() &&
          annotated.insert(key).second)
        annotateFamily(key);
    }
    for (const clang::FunctionDecl *spec :
         const_cast<clang::FunctionTemplateDecl *>(FTD)->specializations()) {
      if (!spec || spec->getTemplateSpecializationKind() !=
                       clang::TSK_ExplicitSpecialization)
        continue;
      const clang::FunctionDecl *key = spec->getCanonicalDecl();
      if (isRewritable(key, SM) && !key->hasAttr<clang::CUDADeviceAttr>() &&
          annotated.insert(key).second)
        annotateFamily(key);
    }
  }

  if (insertedAny)
    if (parasVerbose())
      llvm::outs() << "[ParaS] device-callable: "
                   << definition->getQualifiedNameAsString()
                   << " (all redeclarations)\n";
}

void ParaSDeviceCallGraph::process(const clang::FunctionDecl *FD,
                                   clang::ASTContext &context) {
  const clang::FunctionDecl *definition = nullptr;
  if (!FD || !FD->hasBody(definition) || !definition)
    return;

  annotate(definition, context);

  class CalleeVisitor : public clang::RecursiveASTVisitor<CalleeVisitor> {
    ParaSDeviceCallGraph &graph;

  public:
    llvm::DenseSet<const clang::VarDecl *> staticConstexprCandidates;
    llvm::DenseMap<const clang::VarDecl *,
                   llvm::SmallVector<clang::SourceLocation, 4>>
        staticRefs;

    explicit CalleeVisitor(ParaSDeviceCallGraph &g) : graph(g) {}

    bool shouldVisitImplicitCode() const { return true; }

    bool VisitDeclRefExpr(clang::DeclRefExpr *expr) {
      if (!expr)
        return true;

      const auto *var = llvm::dyn_cast<clang::VarDecl>(expr->getDecl());

      if (!var || !var->isStaticLocal() ||
          var->getStorageClass() != clang::SC_Static)
        return true;
      if (!var->isConstexpr() &&
          !(var->getType().isConstQualified() &&
            var->isUsableInConstantExpressions(var->getASTContext())))
        return true;

      staticConstexprCandidates.insert(var->getCanonicalDecl());
      staticRefs[var->getCanonicalDecl()].push_back(expr->getLocation());
      return true;
    }

    bool VisitCallExpr(clang::CallExpr *call) {
      if (const clang::FunctionDecl *callee = call->getDirectCallee())
        graph.enqueue(callee);
      return true;
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
      graph.enqueue(expr->getConstructor());

      if (const clang::CXXRecordDecl *record =
              expr->getType()->getAsCXXRecordDecl())
        if (record->hasDefinition())
          if (const clang::CXXDestructorDecl *dtor =
                  record->getDefinition()->getDestructor())
            graph.enqueue(dtor);
      return true;
    }

    bool VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
      if (const clang::CXXTemporary *temp = expr->getTemporary())
        graph.enqueue(temp->getDestructor());
      return true;
    }

    bool VisitVarDecl(clang::VarDecl *var) {
      if (var && var->hasLocalStorage())
        graph.enqueueDestructorOf(var->getType());
      return true;
    }
  } visitor(*this);

  visitor.TraverseStmt(const_cast<clang::Stmt *>(definition->getBody()));

  if (const auto *ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(definition))
    for (const clang::CXXCtorInitializer *init : ctor->inits())
      if (init && init->getInit())
        visitor.TraverseStmt(init->getInit());

  if (const auto *dtor = llvm::dyn_cast<clang::CXXDestructorDecl>(definition))
    if (const clang::CXXRecordDecl *rd = dtor->getParent()) {
      for (const clang::FieldDecl *field : rd->fields())
        enqueueDestructorOf(field->getType());
      for (const clang::CXXBaseSpecifier &base : rd->bases())
        enqueueDestructorOf(base.getType());
    }

  clang::SourceManager &SM = context.getSourceManager();

  for (const clang::VarDecl *VD : visitor.staticConstexprCandidates) {
    if (!VD)
      continue;

    if (const auto *parent = llvm::dyn_cast_or_null<clang::FunctionDecl>(
            VD->getParentFunctionOrMethod()))
      if (parent->getCanonicalDecl() != definition->getCanonicalDecl() &&
          !visited.count(normalize(parent))) {
        hoistHostStatic(VD, parent, visitor.staticRefs.lookup(VD), context);
        continue;
      }

    clang::SourceLocation loc = SM.getSpellingLoc(VD->getBeginLoc());
    if (loc.isInvalid() || loc.isMacroID() || SM.isInSystemHeader(loc) ||
        SM.isInExternCSystemHeader(loc))
      continue;

    clang::SourceLocation endLoc = SM.getSpellingLoc(VD->getLocation());
    if (endLoc.isInvalid())
      continue;

    const clang::CharSourceRange prefixRange =
        clang::CharSourceRange::getTokenRange(loc, endLoc);
    const std::string rewrittenPrefix = rewriter.getRewrittenText(prefixRange);

    if (rewrittenPrefix.find("__device__") != std::string::npos)
      continue;

    if (rewriter.InsertTextBefore(loc, "__device__ ")) {
      llvm::errs()
          << "[ParaS] failed to mark static constexpr device variable: "
          << VD->getQualifiedNameAsString() << "\n";
      continue;
    }

    if (parasVerbose())
      llvm::outs() << "[ParaS] device-visible static constexpr variable: "
                   << VD->getQualifiedNameAsString() << "\n";
  }
}

void ParaSDeviceCallGraph::hoistHostStatic(
    const clang::VarDecl *VD, const clang::FunctionDecl *parent,
    llvm::ArrayRef<clang::SourceLocation> refs, clang::ASTContext &context) {
  clang::SourceManager &SM = context.getSourceManager();
  if (parent->isTemplated() || parent->isTemplateInstantiation() ||
      VD->getDeclContext()->isDependentContext())
    return;
  const clang::QualType type = VD->getType().getUnqualifiedType();
  const auto *builtin = type->getAs<clang::BuiltinType>();
  if (!builtin || !(type->isIntegerType() || type->isRealFloatingType()))
    return;
  const clang::APValue *value = VD->evaluateValue();
  if (!value)
    return;

  std::string literal;
  const std::string typeName = type.getAsString(context.getPrintingPolicy());
  if (value->isInt()) {
    const llvm::APSInt &v = value->getInt();
    if (type->isBooleanType())
      literal = v.getBoolValue() ? "true" : "false";
    else
      literal = "static_cast<" + typeName + ">(0x" +
                llvm::toString(v.extOrTrunc(64), 16, false) + "ull)";
  } else if (value->isFloat()) {
    const llvm::APFloat &f = value->getFloat();
    const llvm::APInt bits = f.bitcastToAPInt();
    if (builtin->getKind() == clang::BuiltinType::Float)
      literal = "__builtin_bit_cast(float, 0x" + llvm::toString(bits, 16, false) + "u)";
    else if (builtin->getKind() == clang::BuiltinType::Double)
      literal = "__builtin_bit_cast(double, 0x" + llvm::toString(bits, 16, false) + "ull)";
  }
  if (literal.empty())
    return;

  const clang::Decl *top = parent;
  while (top && !top->getDeclContext()->isFileContext())
    top = llvm::dyn_cast<clang::Decl>(top->getDeclContext());
  if (!top)
    return;
  if (const auto *fd = llvm::dyn_cast<clang::FunctionDecl>(top))
    if (const clang::FunctionTemplateDecl *ftd = fd->getDescribedFunctionTemplate())
      top = ftd;
  clang::SourceLocation at = SM.getExpansionLoc(top->getBeginLoc());
  if (at.isInvalid() || SM.isInSystemHeader(at))
    return;

  llvm::SmallVector<clang::SourceLocation, 4> sites;
  for (clang::SourceLocation ref : refs) {
    if (ref.isMacroID()) {
      if (!SM.isMacroArgExpansion(ref))
        return;
      ref = SM.getSpellingLoc(ref);
    }
    if (ref.isInvalid() || SM.isInSystemHeader(ref))
      return;
    sites.push_back(ref);
  }

  const clang::SourceLocation declLoc = SM.getSpellingLoc(VD->getLocation());
  const std::string twin =
      "paras_sv_" + std::to_string(SM.getFileOffset(declLoc)) + "_" +
      llvm::utohexstr(llvm::hash_value(SM.getFilename(declLoc)) & 0xffffff);
  const clang::CharSourceRange atRange =
      clang::CharSourceRange::getTokenRange(at, at);
  if (rewriter.getRewrittenText(atRange).find(twin) == std::string::npos) {
    std::string text =
        "\n#ifndef PARAS_SV\n"
        "#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)\n"
        "#define PARAS_SV(host, device) device\n"
        "#else\n"
        "#define PARAS_SV(host, device) host\n"
        "#endif\n"
        "#endif\n"
        "__device__ const " + typeName + " " + twin + " = " + literal + ";\n";
    if (rewriter.InsertTextBefore(at, text))
      return;
  }
  const std::string name = VD->getName().str();
  for (clang::SourceLocation ref : sites) {
    const clang::CharSourceRange refRange = clang::CharSourceRange::getCharRange(
        ref, ref.getLocWithOffset(static_cast<int>(name.size())));
    if (rewriter.getRewrittenText(refRange) != name) {
      if (parasVerbose())
        llvm::outs() << "[ParaS]   ref " << ref.printToString(SM) << " is '"
                     << rewriter.getRewrittenText(refRange) << "'\n";
      continue;
    }
    const bool failed = rewriter.ReplaceText(ref, name.size(),
                         "PARAS_SV(" + name + ", " + twin + ")");
    if (parasVerbose())
      llvm::outs() << "[ParaS]   ref " << ref.printToString(SM)
                   << (failed ? " (not rewritable)" : "") << "\n";
  }
  if (parasVerbose())
    llvm::outs() << "[ParaS] device twin " << twin << " for static "
                 << VD->getQualifiedNameAsString() << "\n";
}

void ParaSDeviceCallGraph::run(clang::ASTContext &context) {
  while (!workList.empty()) {
    const clang::FunctionDecl *FD = workList.pop_back_val();
    process(FD, context);
  }
}
