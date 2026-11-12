#ifndef __PARAS_VERSION_HPP__
#define __PARAS_VERSION_HPP__

#include <string>
#include "paras/ParaSVersionConfig.hpp"

using std::string;

namespace paras {
    namespace sycl {
        namespace detail {

            static string version_string() {

                string prefix_zero = (PARAS_VERSION_MINOR < 10 ? "0" : "") ;
                string version = std::to_string(PARAS_VERSION_MAJOR)
                    + "." + prefix_zero + std::to_string(PARAS_VERSION_MINOR)
                    + std::to_string(PARAS_VERSION_PATCH)
                    + PARAS_VERSION_SUFFIX ;

                return "ParaS Compiler: " + version ;
            }
        }   // end of namespace detail
    }   // end of namespace sycl
}   // end of namespace paras

#endif