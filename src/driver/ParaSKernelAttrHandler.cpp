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
#include "clang/AST/QualTypeNames.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cctype>
#include <set>
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

std::vector<AttrSpan> specifiersFrom(const llvm::StringRef text, unsigned pos) {
  std::vector<AttrSpan> out;
  unsigned i = pos;
  while (true) {
    while (i < text.size() && isSpace(text[i]))
      ++i;
    AttrSpan span;
    if (!specifierAt(text, i, span))
      break;
    out.push_back(span);
    i = span.end;
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
  KernelAttrVisitor(clang::ASTContext &ctx, clang::Rewriter &rw, bool gpu)
      : ctx(ctx), SM(ctx.getSourceManager()), rw(rw), gpu(gpu) {
    parasInclude = std::string(PARAS_INSTALL_PREFIX) + "/include/";
  }

  bool shouldVisitTemplateInstantiations() const { return false; }
  bool shouldVisitImplicitCode() const { return false; }

  bool VisitVarDecl(clang::VarDecl *VD) {
    if (!gpu || !VD || !VD->isConstexpr() || !VD->getType()->isReferenceType() ||
        !VD->isLocalVarDecl() || VD->isStaticLocal() ||
        llvm::isa<clang::ParmVarDecl>(VD))
      return true;
    if (const auto *FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(
            VD->getParentFunctionOrMethod()))
      if (FD->isConstexpr())
        return true;
    clang::SourceLocation loc = VD->getBeginLoc();
    clang::FileID fid;
    unsigned off = 0;
    if (!userFileLoc(loc, fid, off))
      return true;
    if (!staticRefs.insert(loc).second)
      return true;
    rw.InsertTextBefore(loc, "static __device__ ");
    return true;
  }

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
    if (specs.empty())
      specs = specifiersFrom(text, off);
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
      removeChars(fid, sp.begin, sp.end - sp.begin);
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
    clang::SourceLocation lb = LE->getBeginLoc(), le = LE->getEndLoc();

    auto argText = [&](clang::SourceLocation l) {
      while (l.isMacroID() && SM.isMacroArgExpansion(l))
        l = SM.getImmediateSpellingLoc(l);
      return l;
    };
    {
      const clang::SourceLocation alb = argText(lb), ale = argText(le),
                                  afrom = argText(from), ato = argText(to);
      if (alb.isFileID() && ale.isFileID() && afrom.isFileID() &&
          ato.isFileID()) {
        lb = alb;
        le = ale;
        from = afrom;
        to = ato;
      }
    }
    if (lb.isMacroID() && from.isMacroID() && to.isMacroID() && le.isMacroID())
      return rewriteMacroLambda(LE, from, to);
    if (from.isMacroID() || to.isMacroID() || lb.isMacroID() || le.isMacroID())
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
    std::vector<std::string> specUses;
    if (kernelLambdas.count(LE))
      specUses = specConstantsIn(const_cast<clang::CompoundStmt *>(body));
    if (specs.empty() && !usesHalf && specUses.empty())
      return true;
    if (!done.insert(lb).second)
      return true;
    std::vector<std::string> props;
    if (usesHalf)
      props.push_back(kFp16Prop);
    for (const std::string &u : specUses)
      props.push_back("::sycl::detail::kp_spec_use<" + u + ">");
    for (const auto &sp : specs) {
      if (!done.insert(SM.getComposedLoc(fid, sp.begin)).second)
        return true;
      props.insert(props.end(), sp.props.begin(), sp.props.end());
      removeChars(fid, sp.begin, sp.end - sp.begin);
    }
    rw.InsertTextBefore(lb, "::sycl::detail::with_kernel_props<" +
                            propsList(props) + ">(");
    rw.InsertTextAfterToken(le, ")");
    return true;
  }

  bool rewriteMacroLambda(clang::LambdaExpr *LE, clang::SourceLocation from,
                          clang::SourceLocation to) {
    const clang::SourceLocation sb = SM.getSpellingLoc(LE->getBeginLoc());
    const clang::SourceLocation se = SM.getSpellingLoc(LE->getEndLoc());
    const clang::SourceLocation sf = SM.getSpellingLoc(from);
    const clang::SourceLocation st = SM.getSpellingLoc(to);
    clang::FileID fid, fid2, fid3, fid4;
    unsigned ob, oe, of, ot;
    if (!userFileLoc(sb, fid, ob) || !userFileLoc(se, fid2, oe) ||
        !userFileLoc(sf, fid3, of) || !userFileLoc(st, fid4, ot) ||
        fid != fid2 || fid != fid3 || fid != fid4 || !(ob < of && of < ot && ot <= oe))
      return true;
    const llvm::StringRef def = SM.getBufferData(fid);

    const std::string param = trim(def.substr(of + 1, ot - of - 1).str());
    if (param.empty() ||
        !std::all_of(param.begin(), param.end(), [](char c) {
          return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
        }) ||
        std::isdigit(static_cast<unsigned char>(param[0])) ||
        param == "mutable" || param == "constexpr" || param == "noexcept")
      return true;

    clang::SourceLocation body = LE->getBeginLoc();
    while (body.isMacroID() && SM.isMacroArgExpansion(body))
      body = SM.getImmediateSpellingLoc(body);
    const clang::CharSourceRange inv = SM.getImmediateExpansionRange(body);
    clang::FileID ifid, ifid2;
    unsigned ib, ie;
    if (!userFileLoc(SM.getSpellingLoc(inv.getBegin()), ifid, ib) ||
        !userFileLoc(SM.getSpellingLoc(inv.getEnd()), ifid2, ie) ||
        ifid != ifid2 || ie <= ib)
      return true;
    const llvm::StringRef itext = SM.getBufferData(ifid);
    const auto specs = scanRange(itext, ib, std::min<unsigned>(ie + 1, itext.size()));
    bool ok = true;
    for (const auto &sp : specs)
      ok = ok && !sp.props.empty();
    if (!ok)
      return true;

    const clang::SourceLocation defKey = SM.getComposedLoc(fid, ob);
    if (macroLambdas.insert(defKey).second) {
      const unsigned pb = of + 1 + static_cast<unsigned>(
                                       def.substr(of + 1).find(param));
      removeChars(fid, pb, static_cast<unsigned>(param.size()));
      rw.InsertTextBefore(defKey, "::sycl::detail::with_kernel_props_fn<void " +
                                      param + ">(");
      rw.InsertTextAfterToken(SM.getComposedLoc(fid, oe), ")");
    }

    if (!specs.empty() &&
        macroInvocations.insert(SM.getComposedLoc(ifid, ib)).second) {
      std::vector<std::string> props;
      for (const auto &sp : specs)
        props.insert(props.end(), sp.props.begin(), sp.props.end());
      for (std::size_t i = 0; i < specs.size(); ++i) {
        const unsigned len = specs[i].end - specs[i].begin;
        if (i == 0)
          rw.ReplaceText(SM.getComposedLoc(ifid, specs[i].begin), len,
                         "(" + propsList(props) + ")");
        else
          removeChars(ifid, specs[i].begin, len);
      }
    }
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

  std::vector<std::string> specConstantsIn(clang::Stmt *body) {
    struct Finder : clang::RecursiveASTVisitor<Finder> {
      std::vector<const clang::TemplateArgumentLoc *> args;
      std::vector<std::pair<const clang::TemplateArgumentLoc *, unsigned>> lists;
      void take(llvm::StringRef name, const clang::TemplateArgumentLoc *a,
                unsigned n) {
        if (name == "get_specialization_constant" && a && n == 1)
          args.push_back(a);
      }
      bool VisitMemberExpr(clang::MemberExpr *E) {
        if (E->hasExplicitTemplateArgs())
          take(E->getMemberNameInfo().getAsString(), E->getTemplateArgs(),
               E->getNumTemplateArgs());
        return true;
      }
      bool VisitCXXDependentScopeMemberExpr(clang::CXXDependentScopeMemberExpr *E) {
        if (E->hasExplicitTemplateArgs())
          take(E->getMemberNameInfo().getAsString(), E->getTemplateArgs(),
               E->getNumTemplateArgs());
        return true;
      }
      bool VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *E) {
        if (E->hasExplicitTemplateArgs())
          take(E->getMemberNameInfo().getAsString(), E->getTemplateArgs(),
               E->getNumTemplateArgs());
        return true;
      }
    } f;
    f.TraverseStmt(body);
    std::vector<std::string> out;
    for (const clang::TemplateArgumentLoc *a : f.args) {
      clang::SourceRange r = a->getSourceRange();
      clang::SourceLocation b = SM.getSpellingLoc(r.getBegin());
      clang::SourceLocation e = SM.getSpellingLoc(r.getEnd());
      if (b.isInvalid() || e.isInvalid() || SM.getFileID(b) != SM.getFileID(e))
        continue;
      const std::string t =
          clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(b, e),
                                      SM, ctx.getLangOpts())
              .str();
      if (t.empty() || std::find(out.begin(), out.end(), t) != out.end())
        continue;
      out.push_back(t);
    }
    return out;
  }

  void removeChars(clang::FileID fid, unsigned off, unsigned len) {
    clang::Rewriter::RewriteOptions opts;
    opts.IncludeInsertsAtBeginOfRange = false;
    opts.IncludeInsertsAtEndOfRange = false;
    rw.RemoveText(SM.getComposedLoc(fid, off), len, opts);
  }

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
  llvm::DenseSet<clang::SourceLocation> staticRefs;
  llvm::DenseSet<clang::SourceLocation> macroLambdas;
  llvm::DenseSet<clang::SourceLocation> macroInvocations;
  bool gpu = false;
};

class KernelUseScanner : public clang::RecursiveASTVisitor<KernelUseScanner> {
public:
  KernelUseScanner(clang::ASTContext &ctx, clang::Rewriter &rw)
      : ctx(ctx), SM(ctx.getSourceManager()), rw(rw) {
    parasInclude = std::string(PARAS_INSTALL_PREFIX) + "/include/";
  }

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitCallExpr(clang::CallExpr *CE) {
    const clang::FunctionDecl *FD = CE->getDirectCallee();
    if (!FD || FD->isDependentContext())
      return true;
    const std::string name = FD->getNameAsString();
    if (name != "single_task" && name != "parallel_for" &&
        name != "parallel_for_work_group")
      return true;
    const auto *MD = llvm::dyn_cast<clang::CXXMethodDecl>(FD);
    if (!MD || !MD->getParent())
      return true;
    const std::string cls = MD->getParent()->getQualifiedNameAsString();
    if (cls != "sycl::handler" && cls != "sycl::queue")
      return true;
    const clang::TemplateArgumentList *targs = FD->getTemplateSpecializationArgs();
    if (!targs || targs->size() == 0 ||
        targs->get(0).getKind() != clang::TemplateArgument::Type)
      return true;
    const clang::QualType kn = targs->get(0).getAsType();
    if (kn.isNull() || kn->isVoidType() || kn->isDependentType() ||
        !nameable(kn))
      return true;
    if (CE->getNumArgs() == 0)
      return true;
    const clang::Expr *last = CE->getArg(CE->getNumArgs() - 1);
    if (!last)
      return true;
    const clang::CXXRecordDecl *RD =
        last->getType().getNonReferenceType()->getAsCXXRecordDecl();
    if (!RD)
      return true;

    std::set<std::string> aspects, specs;
    specOut = &specs;
    llvm::DenseSet<const clang::FunctionDecl *> seen;
    if (RD->isLambda()) {
      if (const clang::CXXMethodDecl *op = RD->getLambdaCallOperator())
        analyze(op, 0, aspects, seen);
    } else {
      for (const clang::CXXMethodDecl *m : RD->methods())
        if (m->getOverloadedOperator() == clang::OO_Call)
          analyze(m, 0, aspects, seen);

      for (const clang::Decl *d : RD->decls())
        if (const auto *ftd = llvm::dyn_cast<clang::FunctionTemplateDecl>(d))
          for (const clang::FunctionDecl *spec : ftd->specializations())
            if (spec->getOverloadedOperator() == clang::OO_Call)
              analyze(spec, 0, aspects, seen);
    }
    specOut = nullptr;
    if (aspects.empty() && specs.empty())
      return true;

    clang::PrintingPolicy pp(ctx.getLangOpts());
    pp.SuppressTagKeyword = true;
    pp.FullyQualifiedName = true;
    pp.SuppressUnwrittenScope = true;
    const std::string kname =
        clang::TypeName::getFullyQualifiedName(kn, ctx, pp, true);
    std::string list;
    for (const std::string &a : aspects)
      list += (list.empty() ? "" : ", ") + a;
    std::string slist;
    for (const std::string &a : specs)
      slist += (slist.empty() ? "" : ", ") + a;
    const std::string key = kname + "|" + list + "|" + slist;
    if (!emitted.insert(key).second)
      return true;
    if (!list.empty())
      out += "static const bool paras_kreq_" + std::to_string(counter++) +
             " = ::sycl::detail::paras_add_kernel_aspects<" + kname + ">({" +
             list + "});\n";
    if (!slist.empty())
      out += "static const bool paras_kreq_" + std::to_string(counter++) +
             " = ::sycl::detail::paras_add_kernel_spec_uses<" + kname +
             ">({" + slist + "});\n";
    return true;
  }

  void finish() {
    if (out.empty())
      return;
    const clang::FileID main = SM.getMainFileID();
    rw.InsertTextAfter(SM.getLocForEndOfFile(main),
                       "\n// ParaS: optional features used by instantiated "
                       "kernels\n" + out);
  }

private:
  bool userCode(const clang::FunctionDecl *FD) {
    clang::SourceLocation loc = SM.getSpellingLoc(FD->getLocation());
    if (loc.isInvalid() || SM.isInSystemHeader(loc) ||
        SM.isInExternCSystemHeader(loc))
      return false;
    auto entry = SM.getFileEntryRefForID(SM.getFileID(loc));
    if (!entry)
      return false;
    const std::string n = entry->getName().str();
    return n.compare(0, parasInclude.size(), parasInclude) != 0;
  }

  bool nameable(clang::QualType T, int depth = 0) {
    if (depth > 16 || T.isNull())
      return false;
    T = T.getCanonicalType();
    if (T->isBuiltinType())
      return true;
    if (const auto *PT = T->getAs<clang::PointerType>())
      return nameable(PT->getPointeeType(), depth + 1);
    if (const auto *ET = T->getAs<clang::EnumType>()) {
      const clang::EnumDecl *ED = ET->getDecl();
      return ED && !ED->getParentFunctionOrMethod();
    }
    const clang::CXXRecordDecl *RD = T->getAsCXXRecordDecl();
    if (!RD || RD->isLambda() || RD->getParentFunctionOrMethod() ||
        !RD->getIdentifier())
      return false;
    if (const auto *spec =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(RD)) {
      for (const clang::TemplateArgument &a : spec->getTemplateArgs().asArray()) {
        if (a.getKind() == clang::TemplateArgument::Type) {
          if (!nameable(a.getAsType(), depth + 1))
            return false;
        } else if (a.getKind() == clang::TemplateArgument::Integral) {
          if (!nameable(a.getIntegralType(), depth + 1))
            return false;
        } else {
          return false;
        }
      }
    }
    return true;
  }

  void declaredAspects(const clang::FunctionDecl *FD, std::set<std::string> &out) {

    std::vector<const clang::FunctionDecl *> decls;
    if (const clang::FunctionTemplateDecl *ftd = FD->getPrimaryTemplate()) {
      for (const clang::RedeclarableTemplateDecl *r : ftd->redecls())
        if (const auto *f = llvm::dyn_cast<clang::FunctionTemplateDecl>(r))
          if (f->getTemplatedDecl())
            decls.push_back(f->getTemplatedDecl());
    } else {
      const clang::FunctionDecl *pattern = FD->getTemplateInstantiationPattern();
      for (const clang::FunctionDecl *r : (pattern ? pattern : FD)->redecls())
        decls.push_back(r);
    }
    for (const clang::FunctionDecl *pattern : decls)
      declaredAspectsAt(FD, pattern, out);
  }

  void declaredAspectsAt(const clang::FunctionDecl *FD,
                         const clang::FunctionDecl *pattern,
                         std::set<std::string> &out) {

    clang::SourceLocation start = pattern->getOuterLocStart();
    if (const clang::FunctionTemplateDecl *t =
            pattern->getDescribedFunctionTemplate())
      start = t->getBeginLoc();
    const clang::SourceLocation b = SM.getExpansionLoc(start);
    const clang::SourceLocation n = SM.getExpansionLoc(pattern->getLocation());
    if (b.isInvalid() || n.isInvalid())
      return;
    const auto db = SM.getDecomposedLoc(b), dn = SM.getDecomposedLoc(n);
    if (db.first != dn.first || dn.second <= db.second)
      return;
    const llvm::StringRef text = SM.getBufferData(db.first);
    std::vector<AttrSpan> specs = scanRange(text, db.second, dn.second);

    unsigned pos = db.second;
    while (true) {
      unsigned i = pos;
      while (i > 0 && isSpace(text[i - 1]))
        --i;
      unsigned j = i;
      while (j > 0 && (std::isalnum(static_cast<unsigned char>(text[j - 1])) ||
                       text[j - 1] == '_'))
        --j;
      if (j < i) {
        pos = j;
        continue;
      }
      const auto before = specifiersBefore(text, i);
      if (before.empty())
        break;
      specs.insert(specs.end(), before.begin(), before.end());
      pos = before.front().begin;
    }
    for (const AttrSpan &sp : specs) {
      const std::string body =
          text.substr(sp.begin + 2, sp.end - sp.begin - 4).str();
      for (const std::string &attr : splitArgs(body)) {
        std::string compact;
        for (char c : attr)
          if (!isSpace(c))
            compact += c;
        const std::string pre = "sycl::device_has(";
        if (compact.compare(0, pre.size(), pre) != 0 || compact.back() != ')')
          continue;
        const std::string args =
            compact.substr(pre.size(), compact.size() - pre.size() - 1);
        for (std::string a : splitArgs(args)) {
          while (a.size() > 2 && a.front() == '(' && a.back() == ')')
            a = a.substr(1, a.size() - 2);
          const std::size_t sep = a.rfind("aspect::");
          if (sep != std::string::npos) {
            out.insert("::sycl::aspect::" + a.substr(sep + 8));
            continue;
          }

          const clang::FunctionTemplateDecl *ftd = FD->getPrimaryTemplate();
          const clang::TemplateArgumentList *targs =
              FD->getTemplateSpecializationArgs();
          if (!ftd || !targs)
            continue;
          const clang::TemplateParameterList *tpl = ftd->getTemplateParameters();
          for (unsigned i = 0; i < tpl->size() && i < targs->size(); ++i) {
            const clang::NamedDecl *p = tpl->getParam(i);
            if (p && p->getNameAsString() == a &&
                targs->get(i).getKind() == clang::TemplateArgument::Integral)
              out.insert("static_cast<::sycl::aspect>(" +
                         std::to_string(targs->get(i).getAsIntegral().getExtValue()) +
                         ")");
          }
        }
      }
    }
  }

  class BodyScan : public clang::RecursiveASTVisitor<BodyScan> {
  public:
    bool half = false;
    std::vector<const clang::FunctionDecl *> callees;
    bool shouldVisitImplicitCode() const { return true; }
    bool VisitExpr(clang::Expr *E) {
      if (!E->isTypeDependent() && isHalfType(E->getType()))
        half = true;
      return true;
    }
    bool VisitVarDecl(clang::VarDecl *VD) {
      if (isHalfType(VD->getType()))
        half = true;
      return true;
    }
    std::vector<const clang::VarDecl *> specIds;
    bool VisitCallExpr(clang::CallExpr *C) {
      if (const clang::FunctionDecl *F = C->getDirectCallee()) {
        callees.push_back(F);

        if (F->getNameAsString() == "get_specialization_constant")
          if (const clang::TemplateArgumentList *ta =
                  F->getTemplateSpecializationArgs())
            for (const clang::TemplateArgument &a : ta->asArray())
              if (a.getKind() == clang::TemplateArgument::Declaration)
                if (const auto *vd =
                        llvm::dyn_cast<clang::VarDecl>(a.getAsDecl()))
                  specIds.push_back(vd);
      }
      return true;
    }
    bool VisitCXXConstructExpr(clang::CXXConstructExpr *C) {
      if (C->getConstructor())
        callees.push_back(C->getConstructor());
      return true;
    }
  };

  std::set<std::string> *specOut = nullptr;

  void analyze(const clang::FunctionDecl *FD, int depth,
               std::set<std::string> &aspects,
               llvm::DenseSet<const clang::FunctionDecl *> &seen) {
    if (!FD || depth > 8 || !seen.insert(FD->getCanonicalDecl()).second)
      return;
    if (!userCode(FD))
      return;
    if (depth > 0)
      declaredAspects(FD, aspects);
    const clang::FunctionDecl *def = nullptr;
    if (!FD->hasBody(def) || !def || !def->getBody())
      return;
    BodyScan scan;
    scan.TraverseStmt(def->getBody());
    if (scan.half)
      aspects.insert("::sycl::aspect::fp16");
    if (specOut)
      for (const clang::VarDecl *vd : scan.specIds)
        if (vd->hasGlobalStorage() && !vd->isStaticLocal() &&
            !vd->getParentFunctionOrMethod() && !vd->isInAnonymousNamespace()) {

          std::string name;
          llvm::raw_string_ostream os(name);
          clang::PrintingPolicy pp(ctx.getLangOpts());
          pp.SuppressTagKeyword = true;
          pp.FullyQualifiedName = true;
          vd->getNameForDiagnostic(os, pp,  true);
          os.flush();
          specOut->insert("&::" + name);
        }
    for (const clang::FunctionDecl *c : scan.callees)
      analyze(c, depth + 1, aspects, seen);
  }

  clang::ASTContext &ctx;
  clang::SourceManager &SM;
  clang::Rewriter &rw;
  std::string parasInclude;
  std::set<std::string> emitted;
  std::string out;
  int counter = 0;
};

}

namespace paras {
void rewriteKernelAttributes(clang::ASTContext &ctx, clang::Rewriter &rw,
                             bool gpuBuild) {
  KernelAttrVisitor v(ctx, rw, gpuBuild);
  v.TraverseDecl(ctx.getTranslationUnitDecl());
  v.finish();
  KernelUseScanner u(ctx, rw);
  u.TraverseDecl(ctx.getTranslationUnitDecl());
  u.finish();
}
}
