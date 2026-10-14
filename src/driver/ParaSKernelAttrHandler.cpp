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
#include "paras/ParaSLog.hpp"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"

#include <cctype>
#include <map>
#include <string>
#include <vector>

namespace {

struct FunctorAttrs {
    const clang::CXXMethodDecl* firstMethod = nullptr;
    clang::SourceLocation insertLoc;
    std::vector<std::string> deviceHas;
    std::vector<std::string> reqdWorkGroup;
    std::vector<std::string> reqdSubGroup;
};

std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
        ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        --e;
    return s.substr(b, e - b);
}

std::vector<std::string> splitTopLevel(const std::string& s) {
    std::vector<std::string> out;
    int depth = 0;
    std::string cur;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{' || c == '<')
            ++depth;
        else if (c == ')' || c == ']' || c == '}' || c == '>')
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

void parseAttributeList(const std::string& inner, FunctorAttrs& info) {
    for (const std::string& attr : splitTopLevel(inner)) {
        const std::size_t lp = attr.find('(');
        std::string name = trim(attr.substr(0, lp));
        std::string compact;
        for (char c : name)
            if (!std::isspace(static_cast<unsigned char>(c)))
                compact += c;
        if (lp == std::string::npos || attr.back() != ')')
            continue;
        const std::vector<std::string> args =
            splitTopLevel(attr.substr(lp + 1, attr.size() - lp - 2));
        if (compact == "sycl::device_has") {
            for (const auto& a : args)
                info.deviceHas.push_back(a);
        } else if (compact == "sycl::reqd_work_group_size") {
            if (info.reqdWorkGroup.empty())
                info.reqdWorkGroup = args;
        } else if (compact == "sycl::reqd_sub_group_size") {
            if (info.reqdSubGroup.empty())
                info.reqdSubGroup = args;
        }
    }
}

std::size_t readLeadingAttributes(llvm::StringRef buf, std::size_t declStart,
                                  FunctorAttrs& info) {
    std::size_t first = std::string::npos;
    auto skipSpaceBack = [&](std::size_t i) {
        while (i > 0 && std::isspace(static_cast<unsigned char>(buf[i - 1])))
            --i;
        return i;
    };
    std::size_t i = skipSpaceBack(declStart);
    while (i >= 4 && buf[i - 1] == ']' && buf[i - 2] == ']') {
        int depth = 0;
        std::size_t j = i;
        bool found = false;
        while (j > 0) {
            --j;
            if (buf[j] == ']')
                ++depth;
            else if (buf[j] == '[' && --depth == 0) {
                found = (j > 0 && buf[j - 1] == '[');
                break;
            }
        }
        if (!found)
            break;
        const std::size_t open = j - 1;
        parseAttributeList(buf.substr(open + 2, (i - 2) - (open + 2)).str(), info);
        first = open;
        i = skipSpaceBack(open);
    }
    std::size_t k = declStart;
    while (k + 1 < buf.size() && buf[k] == '[' && buf[k + 1] == '[') {
        int depth = 0;
        std::size_t j = k;
        for (; j < buf.size(); ++j) {
            if (buf[j] == '[')
                ++depth;
            else if (buf[j] == ']' && --depth == 0)
                break;
        }
        if (j >= buf.size())
            break;
        parseAttributeList(buf.substr(k + 2, (j - 1) - (k + 2)).str(), info);
        if (first == std::string::npos || k < first)
            first = k;
        k = j + 1;
        while (k < buf.size() && std::isspace(static_cast<unsigned char>(buf[k])))
            ++k;
    }
    return first;
}

class SyclKernelAttrCollector : public clang::RecursiveASTVisitor<SyclKernelAttrCollector> {
public:
    explicit SyclKernelAttrCollector(clang::ASTContext& ctx) : ctx_(ctx) {}

    bool shouldVisitTemplateInstantiations() const { return false; }

    bool VisitCXXMethodDecl(clang::CXXMethodDecl* MD) {
        if (MD->getOverloadedOperator() != clang::OO_Call || MD->isImplicit())
            return true;
        const clang::CXXRecordDecl* RD = MD->getParent();
        if (!RD || RD->isLambda() || MD->isOutOfLine())
            return true;

        const clang::SourceManager& SM = ctx_.getSourceManager();
        clang::SourceLocation begin = MD->getBeginLoc();
        if (begin.isInvalid() || begin.isMacroID() || SM.isInSystemHeader(begin))
            return true;

        bool invalid = false;
        const clang::FileID fid = SM.getFileID(begin);
        llvm::StringRef buf = SM.getBufferData(fid, &invalid);
        if (invalid)
            return true;

        FunctorAttrs found;
        const std::size_t first = readLeadingAttributes(buf, SM.getFileOffset(begin), found);
        if (first == std::string::npos ||
            (found.deviceHas.empty() && found.reqdWorkGroup.empty() && found.reqdSubGroup.empty()))
            return true;

        clang::SourceLocation at = SM.getLocForStartOfFile(fid).getLocWithOffset(first);
        if (const auto* FTD = MD->getDescribedFunctionTemplate()) {
            clang::SourceLocation t = FTD->getBeginLoc();
            if (t.isValid() && !t.isMacroID() && SM.getFileID(t) == fid &&
                SM.isBeforeInTranslationUnit(t, at))
                at = t;
        }

        FunctorAttrs& info = byClass_[RD->getCanonicalDecl()];
        if (!info.firstMethod || SM.isBeforeInTranslationUnit(at, info.insertLoc)) {
            info.firstMethod = MD;
            info.insertLoc = at;
        }
        for (auto& a : found.deviceHas)
            info.deviceHas.push_back(a);
        if (info.reqdWorkGroup.empty())
            info.reqdWorkGroup = found.reqdWorkGroup;
        if (info.reqdSubGroup.empty())
            info.reqdSubGroup = found.reqdSubGroup;
        return true;
    }

    std::map<const clang::CXXRecordDecl*, FunctorAttrs> byClass_;

private:
    clang::ASTContext& ctx_;
};

std::string joinArgs(const std::vector<std::string>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i)
            out += ", ";
        out += v[i];
    }
    return out;
}

const char* accessText(clang::AccessSpecifier AS) {
    switch (AS) {
    case clang::AS_private:
        return "private:";
    case clang::AS_protected:
        return "protected:";
    default:
        return "public:";
    }
}

}

void recordSyclKernelAttributes(clang::ASTContext& context, clang::Rewriter& mainRewriter,
                                clang::Rewriter& headerRewriter) {
    SyclKernelAttrCollector collector(context);
    collector.TraverseDecl(context.getTranslationUnitDecl());

    const clang::SourceManager& SM = context.getSourceManager();
    for (const auto& [RD, info] : collector.byClass_) {
        std::string text = "public: ";
        if (!info.deviceHas.empty())
            text += "using paras_device_has = ::sycl::paras_kb::aspect_list<" +
                    joinArgs(info.deviceHas) + ">; ";
        if (!info.reqdWorkGroup.empty())
            text += "using paras_reqd_work_group_size = ::sycl::paras_kb::size_list<" +
                    joinArgs(info.reqdWorkGroup) + ">; ";
        if (!info.reqdSubGroup.empty())
            text += "using paras_reqd_sub_group_size = ::sycl::paras_kb::size_list<" +
                    joinArgs(info.reqdSubGroup) + ">; ";
        text += accessText(info.firstMethod->getAccess());
        text += " ";

        clang::Rewriter& rw = SM.isInMainFile(info.insertLoc) ? mainRewriter : headerRewriter;
        if (rw.InsertTextBefore(info.insertLoc, text)) {
            llvm::errs() << "[ParaS] could not record SYCL kernel attributes of "
                         << RD->getQualifiedNameAsString() << "\n";
            continue;
        }
        parasLog() << "[ParaS] recorded SYCL kernel attributes of "
                   << RD->getQualifiedNameAsString() << "\n";
    }
}
