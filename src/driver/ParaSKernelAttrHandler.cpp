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

#include "paras/ParaSKernelAttrHandler.hpp"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "pathfinder.hpp"

namespace {

struct AttrSpan {
  unsigned begin = 0;
  unsigned end = 0;
  std::vector<std::string> props;
};

bool isSpace(char c) { return std::isspace(static_cast<unsigned char>(c)); }

std::string trim(const std::string &s) {
  std::size_t b = 0, e = s.size();
  while (b < e && isSpace(s[b]))
    ++b;
  while (e > b && isSpace(s[e - 1]))
    --e;
  return s.substr(b, e - b);
}

std::vector<std::string> splitArgs(const std::string &s) {
  std::vector<std::string> out;
  int depth = 0;
  std::string cur;
  for (char c : s) {
    if (c == '(' || c == '[' || c == '{')
      ++depth;
    else if (c == ')' || c == ']' || c == '}')
      --depth;
    if (c == ',' && depth == 0) {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur += c;
    }
  }
  if (!trim(cur).empty())
    out.push_back(trim(cur));
  return out;
}

std::string wrapArgs(const std::vector<std::string> &args) {
  std::string r;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (i)
      r += ", ";
    r += "(" + args[i] + ")";
  }
  return r;
}

bool parseSpecifier(const std::string &body, std::vector<std::string> &props) {

  for (const std::string &attr : splitArgs(body)) {
    std::size_t lp = attr.find('(');
    std::string name = trim(attr.substr(0, lp));
    std::string args;
    if (lp != std::string::npos) {
      std::size_t rp = attr.rfind(')');
      if (rp == std::string::npos || rp < lp)
        return false;
      args = attr.substr(lp + 1, rp - lp - 1);
    }

    std::string compact;
    for (char c : name)
      if (!isSpace(c))
        compact += c;
    const auto list = splitArgs(args);
    if (compact == "sycl::device_has")
      props.push_back("::sycl::detail::kp_device_has<" + wrapArgs(list) + ">");
    else if (compact == "sycl::reqd_work_group_size" && !list.empty())
      props.push_back("::sycl::detail::kp_work_group_size<" + wrapArgs(list) +
                      ">");
    else if (compact == "sycl::reqd_sub_group_size" && list.size() == 1)
      props.push_back("::sycl::detail::kp_sub_group_size<" + wrapArgs(list) +
                      ">");
    else
      return false;
  }
  return true;
}

bool specifierAt(const llvm::StringRef text, unsigned pos, AttrSpan &span) {
  if (pos + 1 >= text.size() || text[pos] != '[' || text[pos + 1] != '[')
    return false;
  int depth = 0;
  for (unsigned i = pos; i < text.size(); ++i) {
    if (text[i] == '[')
      ++depth;
    else if (text[i] == ']') {
      --depth;
      if (depth == 0) {

        if (i < pos + 3 || text[i - 1] != ']')
          return false;
        span.begin = pos;
        span.end = i + 1;
        const std::string body =
            text.substr(pos + 2, i - 1 - (pos + 2)).str();
        span.props.clear();
        return parseSpecifier(body, span.props);
      }
    }
  }
  return false;
}

std::vector<AttrSpan> scanRange(const llvm::StringRef text, unsigned from,
                                unsigned to) {
  std::vector<AttrSpan> out;
  for (unsigned i = from; i + 1 < to; ++i) {
    if (text[i] != '[' || text[i + 1] != '[')
      continue;
    AttrSpan span;
    if (specifierAt(text, i, span) && span.end <= to) {
      out.push_back(span);
      i = span.end - 1;
    }
  }
  return out;
}

std::vector<AttrSpan> specifiersBefore(const llvm::StringRef text,
                                       unsigned pos) {
  std::vector<AttrSpan> out;
  unsigned cur = pos;
  while (true) {
    unsigned i = cur;
    while (i > 0 && isSpace(text[i - 1]))
      --i;
    if (i < 2 || text[i - 1] != ']' || text[i - 2] != ']')
      break;

    int depth = 0;
    unsigned j = i;
    bool found = false;
    while (j > 0) {
      --j;
      if (text[j] == ']')
        ++depth;
      else if (text[j] == '[') {
        --depth;
        if (depth == 0) {

          found = j + 1 < text.size() && text[j + 1] == '[';
          break;
        }
      }
    }
    if (!found)
      break;
    AttrSpan span;
    if (!specifierAt(text, j, span) || span.end != i)
      break;
    out.insert(out.begin(), span);
    cur = j;
  }
  return out;
}

std::string propsList(const std::vector<std::string> &props) {
  std::string r = "::sycl::detail::kp_list<";
  for (std::size_t i = 0; i < props.size(); ++i) {
    if (i)
      r += ", ";
    r += props[i];
  }
  return r + ">";
}

bool isHalfType(clang::QualType T) {
  if (T.isNull())
    return false;
  T = T.getNonReferenceType();
  while (const clang::ArrayType *AT = T->getAsArrayTypeUnsafe())
    T = AT->getElementType();
  T = T.getCanonicalType().getUnqualifiedType();
  const clang::CXXRecordDecl *RD = T->getAsCXXRecordDecl();
  if (!RD)
    return false;
  const std::string name = RD->getQualifiedNameAsString();
  if (name == "sycl::half" || name == "sycl::detail::half_impl::half")
    return true;
  if (const auto *spec =
          llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(RD)) {
    if ((name == "sycl::vec" || name == "sycl::marray") &&
        spec->getTemplateArgs().size() > 0 &&
        spec->getTemplateArgs()[0].getKind() == clang::TemplateArgument::Type)
      return isHalfType(spec->getTemplateArgs()[0].getAsType());
  }
  return false;
}

class HalfUseFinder : public clang::RecursiveASTVisitor<HalfUseFinder> {
public:
  bool found = false;
  bool VisitExpr(clang::Expr *E) {
    if (!E->isTypeDependent() && isHalfType(E->getType()))
      found = true;
    return !found;
  }
  bool VisitVarDecl(clang::VarDecl *VD) {
    if (isHalfType(VD->getType()))
      found = true;
    return !found;
  }
};

bool bodyUsesHalf(clang::Stmt *body) {
  if (!body)
    return false;
  HalfUseFinder f;
  f.TraverseStmt(body);
  return f.found;
}

const char *const kFp16Prop =
    "::sycl::detail::kp_device_has<(::sycl::aspect::fp16)>";

std::string calleeMemberName(const clang::CallExpr *CE) {
  const clang::Expr *callee = CE->getCallee()->IgnoreParenImpCasts();
  if (const auto *ME = llvm::dyn_cast<clang::MemberExpr>(callee))
    return ME->getMemberNameInfo().getAsString();
  if (const auto *DME = llvm::dyn_cast<clang::CXXDependentScopeMemberExpr>(callee))
    return DME->getMemberNameInfo().getAsString();
  if (const auto *UME = llvm::dyn_cast<clang::UnresolvedMemberExpr>(callee))
    return UME->getMemberNameInfo().getAsString();
  return {};
}

class KernelAttrVisitor : public clang::RecursiveASTVisitor<KernelAttrVisitor> {
public:
  KernelAttrVisitor(clang::ASTContext &ctx, clang::Rewriter &rw)
      : ctx(ctx), SM(ctx.getSourceManager()), rw(rw) {
    parasInclude = std::string(PARAS_INSTALL_PREFIX) + "/include/";
  }

  bool shouldVisitTemplateInstantiations() const { return false; }
  bool shouldVisitImplicitCode() const { return false; }

  bool VisitCXXMethodDecl(clang::CXXMethodDecl *MD) {
    if (MD->getOverloadedOperator() != clang::OO_Call)
      return true;
    const clang::CXXRecordDecl *RD = MD->getParent();
    if (!RD || RD->isLambda())
      return true;

    const clang::CXXMethodDecl *declMD = MD;
    if (MD->isOutOfLine())
      declMD = llvm::cast<clang::CXXMethodDecl>(MD->getCanonicalDecl());
    if (declMD->isOutOfLine())
      return true;
    clang::SourceLocation loc = declMD->getBeginLoc();
    if (const auto *FTD = declMD->getDescribedFunctionTemplate())
      loc = FTD->getBeginLoc();
    unsigned off;
    clang::FileID fid;
    if (!userFileLoc(loc, fid, off))
      return true;
    const llvm::StringRef text = SM.getBufferData(fid);
    auto specs = specifiersBefore(text, off);
    const bool usesHalf = MD->doesThisDeclarationHaveABody() &&
                          bodyUsesHalf(MD->getBody());
    if (MD != declMD)
      specs.clear();
    if (specs.empty() && !usesHalf)
      return true;

    auto &entry = classProps[RD->getCanonicalDecl()];
    if (usesHalf) {
      if (std::find(entry.props.begin(), entry.props.end(), kFp16Prop) ==
          entry.props.end())
        entry.props.push_back(kFp16Prop);
      if (entry.insertAt.isInvalid())
        entry.insertAt = SM.getComposedLoc(
            fid, specs.empty() ? off : specs.front().begin);
    }
    for (const auto &sp : specs) {
      if (!done.insert(SM.getComposedLoc(fid, sp.begin)).second)
        continue;
      for (const auto &p : sp.props)
        if (std::find(entry.props.begin(), entry.props.end(), p) ==
            entry.props.end())
          entry.props.push_back(p);
      rw.RemoveText(clang::SourceRange(SM.getComposedLoc(fid, sp.begin),
                                       SM.getComposedLoc(fid, sp.end - 1)));
      if (entry.insertAt.isInvalid())
        entry.insertAt = SM.getComposedLoc(fid, specs.front().begin);
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *CE) {
    const std::string name = calleeMemberName(CE);
    if (name != "single_task" && name != "parallel_for" &&
        name != "parallel_for_work_group")
      return true;
    if (CE->getNumArgs() == 0)
      return true;
    const clang::Expr *last = CE->getArg(CE->getNumArgs() - 1);
    if (!last)
      return true;
    last = last->IgnoreImplicit();
    while (true) {
      if (const auto *MT = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(last))
        last = MT->getSubExpr()->IgnoreImplicit();
      else if (const auto *BT = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(last))
        last = BT->getSubExpr()->IgnoreImplicit();
      else if (const auto *CC = llvm::dyn_cast<clang::CXXConstructExpr>(last);
               CC && CC->getNumArgs() == 1)
        last = CC->getArg(0)->IgnoreImplicit();
      else
        break;
    }
    if (const auto *LE = llvm::dyn_cast<clang::LambdaExpr>(last))
      kernelLambdas.insert(LE);
    return true;
  }

  bool VisitLambdaExpr(clang::LambdaExpr *LE) {
    const clang::CompoundStmt *body = LE->getCompoundStmtBody();
    if (!body)
      return true;
    clang::SourceLocation from = LE->getIntroducerRange().getEnd();
    if (const clang::TypeSourceInfo *TSI =
            LE->getCallOperator()->getTypeSourceInfo())
      if (auto FTL = TSI->getTypeLoc()
                         .IgnoreParens()
                         .getAs<clang::FunctionTypeLoc>())
        if (FTL.getRParenLoc().isValid())
          from = FTL.getRParenLoc();
    clang::SourceLocation to = body->getLBracLoc();
    if (from.isMacroID() || to.isMacroID() || LE->getBeginLoc().isMacroID() ||
        LE->getEndLoc().isMacroID())
      return true;
    unsigned fromOff, toOff;
    clang::FileID fid, fid2;
    if (!userFileLoc(from, fid, fromOff) || !userFileLoc(to, fid2, toOff) ||
        fid != fid2 || toOff <= fromOff)
      return true;
    const llvm::StringRef text = SM.getBufferData(fid);
    auto specs = scanRange(text, fromOff, toOff);
    const bool usesHalf = kernelLambdas.count(LE) &&
                          bodyUsesHalf(const_cast<clang::CompoundStmt *>(body));
    if (specs.empty() && !usesHalf)
      return true;
    if (!done.insert(LE->getBeginLoc()).second)
      return true;
    std::vector<std::string> props;
    if (usesHalf)
      props.push_back(kFp16Prop);
    for (const auto &sp : specs) {
      if (!done.insert(SM.getComposedLoc(fid, sp.begin)).second)
        return true;
      props.insert(props.end(), sp.props.begin(), sp.props.end());
      rw.RemoveText(clang::SourceRange(SM.getComposedLoc(fid, sp.begin),
                                       SM.getComposedLoc(fid, sp.end - 1)));
    }
    rw.InsertTextBefore(LE->getBeginLoc(),
                        "::sycl::detail::with_kernel_props<" +
                            propsList(props) + ">(");
    rw.InsertTextAfterToken(LE->getEndLoc(), ")");
    return true;
  }

  void finish() {
    for (auto &kv : classProps) {
      if (kv.second.insertAt.isInvalid() || kv.second.props.empty())
        continue;
      rw.InsertTextBefore(kv.second.insertAt,
                          "using paras_kernel_props = " +
                              propsList(kv.second.props) + "; ");
    }
  }

private:
  bool userFileLoc(clang::SourceLocation loc, clang::FileID &fid,
                   unsigned &off) {
    if (loc.isInvalid() || loc.isMacroID())
      return false;
    if (SM.isInSystemHeader(loc) || SM.isInExternCSystemHeader(loc))
      return false;
    auto decomposed = SM.getDecomposedLoc(loc);
    fid = decomposed.first;
    off = decomposed.second;
    auto entry = SM.getFileEntryRefForID(fid);
    if (!entry)
      return false;
    const std::string name = entry->getName().str();
    if (name.compare(0, parasInclude.size(), parasInclude) == 0)
      return false;
    return true;
  }

  struct ClassEntry {
    clang::SourceLocation insertAt;
    std::vector<std::string> props;
  };

  clang::ASTContext &ctx;
  clang::SourceManager &SM;
  clang::Rewriter &rw;
  std::string parasInclude;
  llvm::DenseMap<const clang::CXXRecordDecl *, ClassEntry> classProps;
  llvm::DenseSet<clang::SourceLocation> done;
  llvm::DenseSet<const clang::LambdaExpr *> kernelLambdas;
};

}

namespace paras {
void rewriteKernelAttributes(clang::ASTContext &ctx, clang::Rewriter &rw) {
  KernelAttrVisitor v(ctx, rw);
  v.TraverseDecl(ctx.getTranslationUnitDecl());
  v.finish();
}
}
